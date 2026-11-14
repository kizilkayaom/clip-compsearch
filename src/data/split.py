import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CAPTIONS = "captions_train2017.json"
TEST_CAPTIONS = "captions_val2017.json"
GIT_COMMIT_ENV = "CLIP_COMPSEARCH_GIT_COMMIT"
GIT_DIRTY_ENV = "CLIP_COMPSEARCH_GIT_DIRTY"


@dataclass(frozen=True)
class Split:
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def sets(self) -> dict[str, tuple[int, ...]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def load_caption_counts(path: Path) -> dict[int, int]:
    with path.open() as f:
        data = json.load(f)
    counts = Counter(annotation["image_id"] for annotation in data["annotations"])
    return {image["id"]: counts[image["id"]] for image in data["images"]}


def make_split(
    train_pool: list[int], test_ids: list[int], val_size: int, seed: int
) -> Split:
    pool = sorted(set(train_pool))
    if val_size > len(pool):
        raise ValueError(f"val_size {val_size} is larger than the pool ({len(pool)})")
    shared = set(pool) & set(test_ids)
    if shared:
        raise ValueError(f"{len(shared)} image IDs are in the pool and the test set")

    val = set(random.Random(seed).sample(pool, val_size))
    return Split(
        train=tuple(image_id for image_id in pool if image_id not in val),
        val=tuple(sorted(val)),
        test=tuple(sorted(set(test_ids))),
    )


def _ids_text(ids: tuple[int, ...]) -> str:
    return json.dumps(list(ids)) + "\n"


def build_report(
    split: Split, caption_counts: dict[int, int], seed: int
) -> dict[str, Any]:
    report: dict[str, Any] = {"seed": seed}
    for name, ids in split.sets().items():
        report[name] = {
            "images": len(ids),
            "captions": sum(caption_counts[image_id] for image_id in ids),
            "sha256": hashlib.sha256(_ids_text(ids).encode()).hexdigest(),
        }
    return report


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def git_state(repo: Path) -> dict[str, Any]:
    commit = os.environ.get(GIT_COMMIT_ENV)
    if commit:
        return {
            "git_commit": commit,
            "git_dirty": os.environ.get(GIT_DIRTY_ENV, "1") != "0",
        }
    return {
        "git_commit": _git_output(repo, "rev-parse", "HEAD"),
        "git_dirty": _git_output(repo, "status", "--porcelain", "--", "src") != "",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Split COCO 2017 into train, validation and test image IDs."
    )
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--report", type=Path, default=Path("results/split_report.json")
    )
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train_counts = load_caption_counts(args.annotations_dir / TRAIN_CAPTIONS)
    test_counts = load_caption_counts(args.annotations_dir / TEST_CAPTIONS)
    split = make_split(list(train_counts), list(test_counts), args.val_size, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in split.sets().items():
        (args.out_dir / f"{name}_ids.json").write_text(_ids_text(ids))

    report = build_report(split, {**train_counts, **test_counts}, args.seed)
    report.update(git_state(REPO_ROOT))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    for name, ids in split.sets().items():
        captions = report[name]["captions"]
        logger.info("%s: %d images, %d captions", name, len(ids), captions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
