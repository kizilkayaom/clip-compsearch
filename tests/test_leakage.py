import json
from pathlib import Path

import pytest

from src.data.leakage import (
    build_leakage_report,
    coco_id_from_filename,
    find_leaks,
    load_id_sets,
    load_sugarcrepe_ids,
    main,
)
from src.data.split import REPO_ROOT

SPLITS_DIR = REPO_ROOT / "data" / "splits"
SUGARCREPE_DIR = REPO_ROOT / "data" / "sugarcrepe"

CLEAN_SETS = {"train": {1, 2}, "val": {3}, "test": {10, 11}}


def _write_sugarcrepe(path: Path, filenames: list[str]) -> None:
    examples = {
        str(index): {"filename": name, "caption": "c", "negative_caption": "n"}
        for index, name in enumerate(filenames)
    }
    path.write_text(json.dumps(examples))


def _write_splits(splits_dir: Path, id_sets: dict[str, set[int]]) -> None:
    splits_dir.mkdir()
    for name, ids in id_sets.items():
        (splits_dir / f"{name}_ids.json").write_text(json.dumps(sorted(ids)))


def test_coco_id_from_filename() -> None:
    assert coco_id_from_filename("000000085329.jpg") == 85329


def test_coco_id_from_filename_rejects_other_formats() -> None:
    with pytest.raises(ValueError):
        coco_id_from_filename("COCO_val2014_000000085329.jpg")


def test_load_sugarcrepe_ids_joins_all_files(tmp_path: Path) -> None:
    filenames = ["000000000010.jpg", "000000000011.jpg"]
    _write_sugarcrepe(tmp_path / "swap_obj.json", filenames)
    _write_sugarcrepe(tmp_path / "add_att.json", ["000000000011.jpg"])

    assert load_sugarcrepe_ids(tmp_path) == {10, 11}


def test_load_sugarcrepe_ids_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sugarcrepe_ids(tmp_path)


def test_build_leakage_report_counts_shared_images() -> None:
    id_sets = {"train": {1, 2, 3}, "val": {3, 4}, "test": {5}}

    report = build_leakage_report(id_sets, sugarcrepe_ids={2, 5})

    assert report["sizes"] == {"train": 3, "val": 2, "test": 1}
    assert report["shared"] == {"train_val": 1, "train_test": 0, "val_test": 0}
    assert report["sugarcrepe"] == {
        "images": 2,
        "in_train": 1,
        "in_val": 0,
        "in_test": 1,
    }


def test_find_leaks_accepts_clean_split() -> None:
    assert find_leaks(build_leakage_report(CLEAN_SETS, sugarcrepe_ids={10})) == []


def test_find_leaks_reports_shared_images() -> None:
    id_sets = {**CLEAN_SETS, "val": {2, 3}}

    assert len(find_leaks(build_leakage_report(id_sets, sugarcrepe_ids={10}))) == 1


def test_find_leaks_reports_sugarcrepe_outside_test() -> None:
    report = build_leakage_report(CLEAN_SETS, sugarcrepe_ids={1, 10, 99})

    assert len(find_leaks(report)) == 2


def test_main_writes_report(tmp_path: Path) -> None:
    _write_splits(tmp_path / "splits", CLEAN_SETS)
    (tmp_path / "sugarcrepe").mkdir()
    _write_sugarcrepe(tmp_path / "sugarcrepe" / "swap_obj.json", ["000000000010.jpg"])
    report_path = tmp_path / "results" / "leakage_report.json"

    main(
        [
            "--splits-dir", str(tmp_path / "splits"),
            "--sugarcrepe-dir", str(tmp_path / "sugarcrepe"),
            "--report", str(report_path),
        ]
    )

    report = json.loads(report_path.read_text())
    assert report["shared"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert set(report["sha256"]) == {"train", "val", "test"}
    assert len(report["git_commit"]) == 40


def test_main_fails_on_leak(tmp_path: Path) -> None:
    _write_splits(tmp_path / "splits", {**CLEAN_SETS, "test": {2, 10}})
    (tmp_path / "sugarcrepe").mkdir()
    _write_sugarcrepe(tmp_path / "sugarcrepe" / "swap_obj.json", ["000000000010.jpg"])

    with pytest.raises(SystemExit):
        main(
            [
                "--splits-dir", str(tmp_path / "splits"),
                "--sugarcrepe-dir", str(tmp_path / "sugarcrepe"),
                "--report", str(tmp_path / "leakage_report.json"),
            ]
        )


def test_real_split_sets_share_no_image() -> None:
    id_sets = load_id_sets(SPLITS_DIR)

    assert not id_sets["train"] & id_sets["val"]
    assert not id_sets["train"] & id_sets["test"]
    assert not id_sets["val"] & id_sets["test"]


@pytest.mark.skipif(
    not any(SUGARCREPE_DIR.glob("*.json")),
    reason="SugarCrepe JSON files are not in data/sugarcrepe",
)
def test_real_sugarcrepe_images_are_only_in_test_set() -> None:
    id_sets = load_id_sets(SPLITS_DIR)
    sugarcrepe_ids = load_sugarcrepe_ids(SUGARCREPE_DIR)

    assert sugarcrepe_ids <= id_sets["test"]
    assert not sugarcrepe_ids & id_sets["train"]
    assert not sugarcrepe_ids & id_sets["val"]
