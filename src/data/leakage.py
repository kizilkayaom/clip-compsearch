import argparse
import hashlib
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any

from src.data.split import REPO_ROOT, git_state

logger = logging.getLogger(__name__)

SET_NAMES = ("train", "val", "test")


def coco_id_from_filename(filename: str) -> int:
    stem = Path(filename).stem
    if not stem.isdigit():
        raise ValueError(f"Unexpected COCO filename: {filename}")
    return int(stem)


def load_id_sets(splits_dir: Path) -> dict[str, set[int]]:
    return {
        name: set(json.loads((splits_dir / f"{name}_ids.json").read_text()))
        for name in SET_NAMES
    }


def load_sugarcrepe_ids(sugarcrepe_dir: Path) -> set[int]:
    paths = sorted(sugarcrepe_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No SugarCrepe JSON files in {sugarcrepe_dir}")

    ids: set[int] = set()
    for path in paths:
        examples = json.loads(path.read_text()).values()
        ids.update(coco_id_from_filename(example["filename"]) for example in examples)
    return ids


def build_leakage_report(
    id_sets: dict[str, set[int]], sugarcrepe_ids: set[int]
) -> dict[str, Any]:
    return {
        "sizes": {name: len(id_sets[name]) for name in SET_NAMES},
        "shared": {
            f"{first}_{second}": len(id_sets[first] & id_sets[second])
            for first, second in combinations(SET_NAMES, 2)
        },
        "sugarcrepe": {
            "images": len(sugarcrepe_ids),
            **{f"in_{name}": len(sugarcrepe_ids & id_sets[name]) for name in SET_NAMES},
        },
    }


def find_leaks(report: dict[str, Any]) -> list[str]:
    leaks = [
        f"{pair.replace('_', ' and ')} share {count} images"
        for pair, count in report["shared"].items()
        if count
    ]
    sugarcrepe = report["sugarcrepe"]
    for name in ("train", "val"):
        if sugarcrepe[f"in_{name}"]:
            leaks.append(f"{sugarcrepe[f'in_{name}']} SugarCrepe images are in {name}")
    missing = sugarcrepe["images"] - sugarcrepe["in_test"]
    if missing:
        leaks.append(f"{missing} SugarCrepe images are not in test")
    return leaks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check that the split sets and SugarCrepe share no images."
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--sugarcrepe-dir", type=Path, default=Path("data/sugarcrepe"))
    parser.add_argument(
        "--report", type=Path, default=Path("results/leakage_report.json")
    )
    args = parser.parse_args(argv)

    id_sets = load_id_sets(args.splits_dir)
    report = build_leakage_report(id_sets, load_sugarcrepe_ids(args.sugarcrepe_dir))
    report["sha256"] = {
        name: hashlib.sha256(
            (args.splits_dir / f"{name}_ids.json").read_bytes()
        ).hexdigest()
        for name in SET_NAMES
    }
    report.update(git_state(REPO_ROOT))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    leaks = find_leaks(report)
    if leaks:
        raise SystemExit("Leakage found:\n" + "\n".join(leaks))
    logger.info("No leakage: %s", report["shared"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
