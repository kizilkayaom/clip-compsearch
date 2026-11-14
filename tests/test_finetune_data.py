import json
from pathlib import Path

import pytest
import torch

from src.data.cache_embeddings import save_set
from src.finetune.data import TrainPair, epoch_batches, load_train_data

PAIRS_BY_IMAGE = {
    image_id: tuple(
        TrainPair(image_id, 10 * image_id + k, f"caption {image_id}-{k}", f"neg {k}")
        for k in range(3)
    )
    for image_id in range(1, 11)
}


def test_epoch_batches_use_each_image_at_most_once_with_one_caption() -> None:
    batches = epoch_batches(PAIRS_BY_IMAGE, seed=0, epoch=1, batch_size=4)

    images = [pair.image_id for batch in batches for pair in batch]
    assert len(batches) == 2
    assert all(len(batch) == 4 for batch in batches)
    assert len(images) == len(set(images)) == 8


def test_epoch_batches_depend_only_on_seed_and_epoch() -> None:
    first = epoch_batches(PAIRS_BY_IMAGE, seed=0, epoch=1, batch_size=4)
    reordered = dict(reversed(list(PAIRS_BY_IMAGE.items())))

    assert epoch_batches(reordered, seed=0, epoch=1, batch_size=4) == first
    assert epoch_batches(PAIRS_BY_IMAGE, seed=0, epoch=2, batch_size=4) != first
    assert epoch_batches(PAIRS_BY_IMAGE, seed=1, epoch=1, batch_size=4) != first


def test_epoch_batches_sample_different_captions_across_epochs() -> None:
    captions = {
        pair.caption_id
        for epoch in range(1, 6)
        for batch in epoch_batches(PAIRS_BY_IMAGE, seed=0, epoch=epoch, batch_size=5)
        for pair in batch
    }

    assert len(captions) > len(PAIRS_BY_IMAGE)


def test_load_train_data_groups_pairs_and_maps_rows(tmp_path: Path) -> None:
    records = [
        {"image_id": 7, "caption_id": 2, "caption": "b", "negative": "nb"},
        {"image_id": 5, "caption_id": 1, "caption": "a", "negative": "na"},
        {"image_id": 7, "caption_id": 1, "caption": "c", "negative": "nc"},
    ]
    pairs_path = tmp_path / "train.jsonl"
    pairs_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    embeddings = tmp_path / "train.pt"
    features = torch.randn(3, 512)
    save_set(embeddings, [5, 6, 7], features)

    data = load_train_data(pairs_path, embeddings)

    assert sorted(data.pairs_by_image) == [5, 7]
    assert [p.caption_id for p in data.pairs_by_image[7]] == [1, 2]
    assert torch.equal(data.image_features_for([7, 5]), features[[2, 0]])


def test_load_train_data_limit_keeps_first_images(tmp_path: Path) -> None:
    records = [
        {"image_id": i, "caption_id": i, "caption": "a", "negative": "b"}
        for i in (3, 1, 2)
    ]
    pairs_path = tmp_path / "train.jsonl"
    pairs_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    embeddings = tmp_path / "train.pt"
    save_set(embeddings, [1, 2, 3], torch.randn(3, 512))

    data = load_train_data(pairs_path, embeddings, limit_images=2)

    assert sorted(data.pairs_by_image) == [1, 2]


def test_load_train_data_rejects_pairs_without_embedding(tmp_path: Path) -> None:
    pairs_path = tmp_path / "train.jsonl"
    record = {"image_id": 9, "caption_id": 1, "caption": "a", "negative": "b"}
    pairs_path.write_text(json.dumps(record) + "\n")
    embeddings = tmp_path / "train.pt"
    save_set(embeddings, [1], torch.randn(1, 512))

    with pytest.raises(ValueError):
        load_train_data(pairs_path, embeddings)
