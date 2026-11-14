import hashlib
import json
import random
import subprocess
from pathlib import Path

import pytest

from src.data.split import (
    Split,
    build_report,
    git_state,
    load_caption_counts,
    main,
    make_split,
)

POOL = list(range(1, 101))
TEST_IDS = [1001, 1002, 1003]


def _write_captions(path: Path, captions_per_image: dict[int, int]) -> None:
    images = [{"id": image_id} for image_id in captions_per_image]
    annotations = [
        {"image_id": image_id, "id": index, "caption": f"caption {index}"}
        for image_id, count in captions_per_image.items()
        for index in range(count)
    ]
    path.write_text(json.dumps({"images": images, "annotations": annotations}))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_make_split_partitions_the_pool() -> None:
    split = make_split(POOL, TEST_IDS, val_size=10, seed=0)

    assert len(split.val) == 10
    assert set(split.train) | set(split.val) == set(POOL)
    assert not set(split.train) & set(split.val)
    assert split.test == tuple(TEST_IDS)


def test_make_split_returns_sorted_ids() -> None:
    split = make_split(POOL[::-1], TEST_IDS[::-1], val_size=10, seed=0)

    for ids in (split.train, split.val, split.test):
        assert list(ids) == sorted(ids)


def test_make_split_ignores_input_order() -> None:
    shuffled = POOL.copy()
    random.Random(1).shuffle(shuffled)

    expected = make_split(POOL, TEST_IDS, val_size=10, seed=0)

    assert make_split(shuffled, TEST_IDS, val_size=10, seed=0) == expected


def test_make_split_changes_with_the_seed() -> None:
    first = make_split(POOL, TEST_IDS, val_size=10, seed=0)

    assert make_split(POOL, TEST_IDS, val_size=10, seed=1) != first


def test_make_split_rejects_val_size_larger_than_pool() -> None:
    with pytest.raises(ValueError):
        make_split(POOL, TEST_IDS, val_size=len(POOL) + 1, seed=0)


def test_make_split_rejects_test_ids_in_the_pool() -> None:
    with pytest.raises(ValueError):
        make_split(POOL, [5, 1001], val_size=10, seed=0)


def test_load_caption_counts_includes_images_without_captions(tmp_path: Path) -> None:
    path = tmp_path / "captions.json"
    _write_captions(path, {1: 2, 2: 0, 3: 1})

    assert load_caption_counts(path) == {1: 2, 2: 0, 3: 1}


def test_build_report_counts_images_and_captions() -> None:
    split = Split(train=(1, 2), val=(3,), test=(10,))
    report = build_report(split, {1: 5, 2: 6, 3: 5, 10: 7}, seed=0)

    assert report["seed"] == 0
    assert report["train"]["images"] == 2
    assert report["train"]["captions"] == 11
    assert report["val"]["captions"] == 5
    assert report["test"]["captions"] == 7


def test_git_state_flags_untracked_source_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "committed.py").write_text("")
    _git(tmp_path, "init")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "init")
    assert git_state(tmp_path)["git_dirty"] is False

    (tmp_path / "src" / "new.py").write_text("")

    assert git_state(tmp_path)["git_dirty"] is True


def test_main_writes_id_files_and_report(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    train_counts = {image_id: 5 for image_id in range(1, 21)}
    _write_captions(annotations / "captions_train2017.json", train_counts)
    _write_captions(annotations / "captions_val2017.json", {101: 5, 102: 6, 103: 5})
    out_dir = tmp_path / "splits"
    report_path = tmp_path / "results" / "split_report.json"

    main(
        [
            "--annotations-dir", str(annotations),
            "--out-dir", str(out_dir),
            "--report", str(report_path),
            "--val-size", "4",
            "--seed", "0",
        ]
    )

    expected = make_split(list(range(1, 21)), [101, 102, 103], val_size=4, seed=0)
    report = json.loads(report_path.read_text())
    for name in ("train", "val", "test"):
        path = out_dir / f"{name}_ids.json"
        assert tuple(json.loads(path.read_text())) == getattr(expected, name)
        assert report[name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["test"]["captions"] == 16
    assert len(report["git_commit"]) == 40


def test_git_state_uses_launcher_values_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLIP_COMPSEARCH_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("CLIP_COMPSEARCH_GIT_DIRTY", "0")

    assert git_state(tmp_path) == {"git_commit": "a" * 40, "git_dirty": False}
