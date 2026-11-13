import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LABELS = frozenset({"grammatical", "broken"})
BREAK_TYPES = ("article or number", "word order", "other")


def _normalize(value: str) -> str:
    return " ".join(value.split()).lower()


def _row_error(label: str, break_type: str) -> str | None:
    if label not in LABELS:
        return f"label {label!r} is not one of {sorted(LABELS)}"
    if label == "broken" and break_type not in BREAK_TYPES:
        return f"broken row needs a break_type from {list(BREAK_TYPES)}"
    if label == "grammatical" and break_type:
        return "grammatical row must have an empty break_type"
    return None


def broken_rate(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [
            (_normalize(row["label"]), _normalize(row["break_type"]))
            for row in csv.DictReader(f)
        ]
    if not rows:
        raise ValueError(f"No rows in {path}")
    errors = [
        f"row {index}: {error}"
        for index, (label, break_type) in enumerate(rows, start=1)
        if (error := _row_error(label, break_type))
    ]
    if errors:
        raise ValueError(f"Invalid labels in {path}: " + "; ".join(errors))

    broken = [break_type for label, break_type in rows if label == "broken"]
    return {
        "labelled": len(rows),
        "broken": len(broken),
        "broken_rate": len(broken) / len(rows),
        "broken_by_type": {name: broken.count(name) for name in BREAK_TYPES},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Add the broken-grammar rate of the labelled sample to the report."
    )
    parser.add_argument(
        "--sample", type=Path, default=Path("results/grammar_sample.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/negatives_report.json")
    )
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text())
    report["grammar_sample"].update(broken_rate(args.sample))
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Grammar sample: %s", report["grammar_sample"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
