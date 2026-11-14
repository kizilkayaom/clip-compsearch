import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import CLIPModel, CLIPProcessor

from src.data.cache_embeddings import save_set
from src.eval.validation import (
    CompExample,
    ValidationSet,
    compositional_correct,
    encode_texts,
    evaluate,
    load_validation_set,
    main,
    select_comp_examples,
)
from src.features import load_clip

CAPTIONS = {
    10: ["A dog runs on the grass.", "A brown dog on a lawn."],
    20: ["A red car parked on a street."],
    30: ["A plate of pasta on a table."],
}
RECORDS = [
    {"image_id": 10, "caption_id": 1, "caption": "A dog runs on the grass.",
     "negative": "A grass runs on the dog.", "swap_type": "noun"},
    {"image_id": 10, "caption_id": 2, "caption": "A brown dog on a lawn.",
     "negative": "A brown lawn on a dog.", "swap_type": "noun"},
    {"image_id": 20, "caption_id": 3, "caption": "A red car parked on a street.",
     "negative": "A red street parked on a car.", "swap_type": "noun"},
]


@pytest.fixture(scope="module")
def clip() -> tuple[CLIPModel, CLIPProcessor]:
    return load_clip("cpu")


def test_compositional_correct_compares_caption_and_negative() -> None:
    images = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    captions = torch.tensor([[2.0, 0.1], [0.0, 1.0], [1.0, 1.0]])
    negatives = torch.tensor([[0.0, 1.0], [5.0, 0.0], [1.0, 1.0]])

    correct = compositional_correct(images, captions, negatives)

    assert correct.tolist() == [True, False, False]


def test_select_comp_examples_takes_one_seeded_caption_per_image() -> None:
    examples = select_comp_examples(RECORDS, seed=0)

    assert [example.image_id for example in examples] == [10, 20]
    assert select_comp_examples(list(reversed(RECORDS)), seed=0) == examples
    assert examples[1] == CompExample(
        image_id=20,
        caption_id=3,
        caption="A red car parked on a street.",
        negative="A red street parked on a car.",
    )


def test_select_comp_examples_changes_with_the_seed() -> None:
    records = [
        {**RECORDS[0], "caption_id": index, "image_id": image_id}
        for image_id in range(20)
        for index in range(5)
    ]

    first = select_comp_examples(records, seed=0)

    assert select_comp_examples(records, seed=1) != first


def test_encode_texts_keeps_order_without_gradients(
    clip: tuple[CLIPModel, CLIPProcessor],
) -> None:
    model, processor = clip
    texts = ["A dog.", "A red car parked on a street.", "Pasta."]

    batched = encode_texts(model, processor, texts, batch_size=2)
    single = encode_texts(model, processor, texts[::-1], batch_size=1)

    assert batched.shape == (3, 512)
    assert not batched.requires_grad
    assert torch.allclose(batched, single.flip(0), atol=1e-5)


def test_evaluate_gives_perfect_recall_when_images_match_captions(
    clip: tuple[CLIPModel, CLIPProcessor],
) -> None:
    model, processor = clip
    captions = ["A dog runs on the grass.", "A red car.", "A plate of pasta."]
    image_features = encode_texts(model, processor, captions, batch_size=3)
    data = ValidationSet(
        image_ids=(10, 20, 30),
        image_features=image_features,
        captions=tuple(captions),
        caption_image_ids=(10, 20, 30),
        comp_examples=(CompExample(10, 1, captions[0], captions[0]),),
    )

    metrics = evaluate(model, processor, data, batch_size=2)

    assert metrics["t2i_r1"] == 1.0
    assert metrics["t2i_r10"] == 1.0
    assert metrics["comp_accuracy"] == 0.0
    assert metrics["comp_examples"] == 1


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    embeddings = tmp_path / "val.pt"
    save_set(embeddings, [10, 20, 30], torch.randn(3, 512))
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    entries = [
        {"id": 100 * image_id + index, "image_id": image_id, "caption": text}
        for image_id, texts in CAPTIONS.items()
        for index, text in enumerate(texts)
    ]
    entries.append({"id": 999, "image_id": 77, "caption": "Not in the slice."})
    (annotations / "captions_train2017.json").write_text(
        json.dumps({"images": [], "annotations": entries})
    )
    negatives = tmp_path / "val.jsonl"
    negatives.write_text("".join(json.dumps(record) + "\n" for record in RECORDS))
    return embeddings, annotations, negatives


def test_load_validation_set_uses_only_slice_captions(tmp_path: Path) -> None:
    embeddings, annotations, negatives = _write_inputs(tmp_path)

    data = load_validation_set(
        embeddings, annotations / "captions_train2017.json", negatives, seed=0
    )

    assert data.image_ids == (10, 20, 30)
    assert len(data.captions) == 4
    assert data.caption_image_ids == (10, 10, 20, 30)
    assert [example.image_id for example in data.comp_examples] == [10, 20]


def test_load_validation_set_rejects_negatives_outside_the_slice(
    tmp_path: Path,
) -> None:
    embeddings, annotations, negatives = _write_inputs(tmp_path)
    outside = {**RECORDS[0], "image_id": 77}
    negatives.write_text(json.dumps(outside) + "\n")

    with pytest.raises(ValueError):
        load_validation_set(
            embeddings, annotations / "captions_train2017.json", negatives, seed=0
        )


def test_main_writes_baseline_report(tmp_path: Path) -> None:
    embeddings, annotations, negatives = _write_inputs(tmp_path)
    report_path = tmp_path / "results" / "val_baseline.json"

    main(
        [
            "--embeddings", str(embeddings),
            "--annotations-dir", str(annotations),
            "--negatives", str(negatives),
            "--report", str(report_path),
            "--device", "cpu",
        ]
    )

    report = json.loads(report_path.read_text())
    assert report["counts"] == {"images": 3, "captions": 4, "comp_examples": 2}
    assert set(report["metrics"]) == {
        "comp_accuracy", "comp_examples", "t2i_r1", "t2i_r5", "t2i_r10"
    }
    negatives_hash = hashlib.sha256(negatives.read_bytes()).hexdigest()
    assert report["inputs"]["negatives_sha256"] == negatives_hash
    assert len(report["git_commit"]) == 40
