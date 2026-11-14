import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.cache_embeddings import load_set


@dataclass(frozen=True)
class TrainPair:
    image_id: int
    caption_id: int
    caption: str
    negative: str


@dataclass(frozen=True)
class TrainData:
    pairs_by_image: Mapping[int, tuple[TrainPair, ...]]
    image_features: torch.Tensor
    row_of: Mapping[int, int]

    def image_features_for(self, image_ids: Sequence[int]) -> torch.Tensor:
        return self.image_features[[self.row_of[image_id] for image_id in image_ids]]


def load_train_data(
    pairs_path: Path, embeddings_path: Path, limit_images: int | None = None
) -> TrainData:
    grouped: dict[int, list[TrainPair]] = defaultdict(list)
    with pairs_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pair = TrainPair(
                record["image_id"],
                record["caption_id"],
                record["caption"],
                record["negative"],
            )
            grouped[pair.image_id].append(pair)

    image_ids, features = load_set(embeddings_path)
    row_of = {image_id: row for row, image_id in enumerate(image_ids)}
    missing = sorted(grouped.keys() - row_of.keys())
    if missing:
        raise ValueError(f"{len(missing)} pair image IDs have no cached embedding")

    selected = sorted(grouped)[:limit_images]
    pairs_by_image = {
        image_id: tuple(sorted(grouped[image_id], key=lambda p: p.caption_id))
        for image_id in selected
    }
    return TrainData(pairs_by_image, features, row_of)


def epoch_batches(
    pairs_by_image: Mapping[int, Sequence[TrainPair]],
    seed: int,
    epoch: int,
    batch_size: int,
) -> list[list[TrainPair]]:
    rng = random.Random(f"{seed}:{epoch}")
    picks = [rng.choice(pairs_by_image[i]) for i in sorted(pairs_by_image)]
    rng.shuffle(picks)
    full = len(picks) - len(picks) % batch_size
    return [picks[start : start + batch_size] for start in range(0, full, batch_size)]
