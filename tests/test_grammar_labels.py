import csv
import json
from pathlib import Path

import pytest

from src.data.grammar_labels import broken_rate, main

FIELDS = (
    "caption_id",
    "image_id",
    "swap_type",
    "caption",
    "negative",
    "label",
    "break_type",
)


def _write_sample(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for index, (label, break_type) in enumerate(rows):
            writer.writerow(
                {
                    "caption_id": index,
                    "image_id": index,
                    "swap_type": "noun",
                    "caption": "A cat chased a dog.",
                    "negative": "A dog chased a cat.",
                    "label": label,
                    "break_type": break_type,
                }
            )


def test_broken_rate_counts_labels_and_break_types(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    rows = [
        ("grammatical", ""),
        ("broken", "article or number"),
        (" Broken ", " Word Order "),
        ("grammatical", ""),
    ]
    _write_sample(path, rows)

    assert broken_rate(path) == {
        "labelled": 4,
        "broken": 2,
        "broken_rate": 0.5,
        "broken_by_type": {"article or number": 1, "word order": 1, "other": 0},
    }


def test_broken_rate_rejects_missing_or_unknown_label(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    _write_sample(path, [("grammatical", ""), ("", ""), ("maybe", "")])

    with pytest.raises(ValueError):
        broken_rate(path)


def test_broken_rate_rejects_broken_row_without_valid_break_type(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.csv"
    _write_sample(path, [("broken", ""), ("broken", "spelling")])

    with pytest.raises(ValueError):
        broken_rate(path)


def test_broken_rate_rejects_break_type_on_grammatical_row(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    _write_sample(path, [("grammatical", "word order")])

    with pytest.raises(ValueError):
        broken_rate(path)


def test_broken_rate_rejects_empty_sample(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    _write_sample(path, [])

    with pytest.raises(ValueError):
        broken_rate(path)


def test_main_adds_rate_to_report(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    _write_sample(sample_path, [("broken", "other"), ("grammatical", "")])
    report_path = tmp_path / "report.json"
    report = {"grammar_sample": {"source": "train", "size": 2}}
    report_path.write_text(json.dumps(report))

    main(["--sample", str(sample_path), "--report", str(report_path)])

    assert json.loads(report_path.read_text())["grammar_sample"] == {
        "source": "train",
        "size": 2,
        "labelled": 2,
        "broken": 1,
        "broken_rate": 0.5,
        "broken_by_type": {"article or number": 0, "word order": 0, "other": 1},
    }
