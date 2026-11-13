import csv
import hashlib
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from src.data.build_negatives import (
    Caption,
    assign_sets,
    draw_grammar_sample,
    load_captions,
    main,
    pairs_sha256,
    process_set,
)
from src.data.negatives import nlp

CAPTIONS = [
    Caption(caption_id=1, image_id=10, text="The cat chased the dog."),
    Caption(caption_id=2, image_id=10, text="A small red car."),
    Caption(caption_id=3, image_id=20, text="A dog is running."),
    Caption(caption_id=4, image_id=30, text="A dog and a cat sit on a sofa."),
]
CONTRADICTION = [1.0, 0.0, 0.0]
ENTAILMENT = [0.0, 1.0, 0.0]


def contradiction_scores(pairs: list[tuple[str, str]]) -> list[list[float]]:
    return [CONTRADICTION for _ in pairs]


def entail_red_scores(pairs: list[tuple[str, str]]) -> list[list[float]]:
    return [ENTAILMENT if "red" in caption else CONTRADICTION for caption, _ in pairs]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    entries = [
        {"id": c.caption_id, "image_id": c.image_id, "caption": c.text}
        for c in CAPTIONS
    ]
    (annotations / "captions_train2017.json").write_text(
        json.dumps({"images": [], "annotations": entries})
    )
    splits = tmp_path / "splits"
    splits.mkdir()
    for name, ids in {"train": [10, 30], "val": [20], "test": [99]}.items():
        (splits / f"{name}_ids.json").write_text(json.dumps(ids))
    return annotations, splits


def test_load_captions_collapses_whitespace_and_sorts(tmp_path: Path) -> None:
    path = tmp_path / "captions.json"
    entries = [
        {"id": 5, "image_id": 2, "caption": " A  dog\nruns.\n"},
        {"id": 3, "image_id": 1, "caption": "A cat."},
        {"id": 4, "image_id": 2, "caption": "A bird."},
    ]
    path.write_text(json.dumps({"images": [], "annotations": entries}))

    assert load_captions(path) == [
        Caption(caption_id=3, image_id=1, text="A cat."),
        Caption(caption_id=4, image_id=2, text="A bird."),
        Caption(caption_id=5, image_id=2, text="A dog runs."),
    ]


def test_clip_tokenizer_ignores_extra_whitespace() -> None:
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    spaced = tokenizer("A dog  runs on\nthe grass.")["input_ids"]

    assert spaced == tokenizer("A dog runs on the grass.")["input_ids"]


def test_pairs_sha256_ignores_nli_scores() -> None:
    record = {
        "image_id": 10,
        "caption_id": 1,
        "caption": "The cat chased the dog.",
        "negative": "The dog chased the cat.",
        "swap_type": "noun",
        "nli_scores": CONTRADICTION,
    }

    rescored = {**record, "nli_scores": ENTAILMENT}
    changed = {**record, "negative": "A dog."}

    assert pairs_sha256([record]) == pairs_sha256([rescored])
    assert pairs_sha256([record]) != pairs_sha256([changed])


def test_assign_sets_groups_captions_by_image() -> None:
    grouped = assign_sets(CAPTIONS, {"train": {10, 30}, "val": {20}})

    assert [c.caption_id for c in grouped["train"]] == [1, 2, 4]
    assert [c.caption_id for c in grouped["val"]] == [3]


def test_assign_sets_rejects_image_outside_all_sets() -> None:
    with pytest.raises(ValueError):
        assign_sets(CAPTIONS, {"train": {10}, "val": {20}})


def test_process_set_counts_each_outcome() -> None:
    _, summary = process_set(CAPTIONS, nlp, score_fn=entail_red_scores)

    assert summary == {
        "total_captions": 4,
        "noun_swaps": 2,
        "adjective_swaps": 1,
        "no_swap": 1,
        "rule_rejected": 1,
        "rule_rejected_by_relation": {"and": 1},
        "kept_pairs": 2,
        "nli_entailment_in_kept": 1,
        "images": 3,
        "images_with_kept_pair": 1,
    }


def test_process_set_writes_record_fields() -> None:
    records, _ = process_set(CAPTIONS[:1], nlp, score_fn=contradiction_scores)

    assert records == [
        {
            "image_id": 10,
            "caption_id": 1,
            "caption": "The cat chased the dog.",
            "negative": "The dog chased the cat.",
            "swap_type": "noun",
            "nli_scores": CONTRADICTION,
        }
    ]


def test_process_set_scores_only_kept_pairs_in_chunks() -> None:
    calls: list[list[tuple[str, str]]] = []

    def record_calls(pairs: list[tuple[str, str]]) -> list[list[float]]:
        calls.append(pairs)
        return contradiction_scores(pairs)

    process_set(CAPTIONS, nlp, score_fn=record_calls, chunk_size=1)

    assert [len(pairs) for pairs in calls] == [1, 1]
    assert calls[0][0] == ("The cat chased the dog.", "The dog chased the cat.")


def test_draw_grammar_sample_is_seeded() -> None:
    records = [{"caption_id": index} for index in range(10)]

    first = draw_grammar_sample(records, size=3, seed=0)

    assert len(first) == 3
    assert draw_grammar_sample(records, size=3, seed=0) == first


def test_draw_grammar_sample_returns_all_when_too_few() -> None:
    records = [{"caption_id": index} for index in range(10)]

    assert len(draw_grammar_sample(records, size=50, seed=0)) == 10


def test_main_writes_pairs_report_and_sample(tmp_path: Path) -> None:
    annotations, splits = _write_inputs(tmp_path)
    out_dir = tmp_path / "negatives"
    report_path = tmp_path / "results" / "negatives_report.json"
    sample_path = tmp_path / "results" / "grammar_sample.csv"

    main(
        [
            "--annotations-dir", str(annotations),
            "--splits-dir", str(splits),
            "--out-dir", str(out_dir),
            "--report", str(report_path),
            "--grammar-sample", str(sample_path),
        ],
        score_fn=contradiction_scores,
    )

    train_path = out_dir / "train.jsonl"
    lines = train_path.read_text().splitlines()
    assert [json.loads(line)["caption_id"] for line in lines] == [1, 2]
    assert (out_dir / "val.jsonl").read_text() == ""

    report = json.loads(report_path.read_text())
    assert report["train"]["kept_pairs"] == 2
    assert report["train"]["rule_rejected_by_relation"] == {"and": 1}
    assert report["val"]["no_swap"] == 1
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    assert report["train"]["sha256"] == train_hash

    with sample_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(row["label"] == "" and row["break_type"] == "" for row in rows)


def test_main_pairs_hash_is_stable_when_nli_scores_change(tmp_path: Path) -> None:
    annotations, splits = _write_inputs(tmp_path)
    reports = []
    for name, score_fn in (("a", contradiction_scores), ("b", entail_red_scores)):
        report_path = tmp_path / name / "report.json"
        main(
            [
                "--annotations-dir", str(annotations),
                "--splits-dir", str(splits),
                "--out-dir", str(tmp_path / name / "negatives"),
                "--report", str(report_path),
                "--grammar-sample", str(tmp_path / name / "sample.csv"),
            ],
            score_fn=score_fn,
        )
        reports.append(json.loads(report_path.read_text())["train"])

    assert reports[0]["pairs_sha256"] == reports[1]["pairs_sha256"]
    assert reports[0]["sha256"] != reports[1]["sha256"]


def test_main_limit_processes_first_captions_of_each_set(tmp_path: Path) -> None:
    annotations, splits = _write_inputs(tmp_path)
    out_dir = tmp_path / "negatives"

    main(
        [
            "--annotations-dir", str(annotations),
            "--splits-dir", str(splits),
            "--out-dir", str(out_dir),
            "--report", str(tmp_path / "report.json"),
            "--grammar-sample", str(tmp_path / "sample.csv"),
            "--limit", "1",
        ],
        score_fn=contradiction_scores,
    )

    lines = (out_dir / "train.jsonl").read_text().splitlines()
    assert [json.loads(line)["caption_id"] for line in lines] == [1]
