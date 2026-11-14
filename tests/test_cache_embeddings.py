import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.data.cache_embeddings import (
    check_cache,
    encode_images,
    image_path,
    load_set,
    main,
    missing_images,
    save_set,
)
from src.features import EMBED_DIM, load_clip

COLORS = {9: (255, 0, 0), 25: (0, 255, 0), 30: (0, 0, 255), 42: (255, 255, 0)}


@pytest.fixture(scope="module")
def clip() -> tuple[CLIPModel, CLIPProcessor]:
    return load_clip("cpu")


def _write_images(image_dir: Path, image_ids: list[int]) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for image_id in image_ids:
        path = image_path(image_dir, image_id)
        Image.new("RGB", (64, 48), COLORS[image_id]).save(path)
        paths.append(path)
    return paths


def test_image_path_uses_coco_file_names(tmp_path: Path) -> None:
    assert image_path(tmp_path, 9) == tmp_path / "000000000009.jpg"


def test_missing_images_lists_ids_without_file(tmp_path: Path) -> None:
    _write_images(tmp_path, [9, 25])

    assert missing_images([9, 25, 30], tmp_path) == [30]


def test_encode_images_keeps_input_order_and_batch_size_has_no_effect(
    tmp_path: Path, clip: tuple[CLIPModel, CLIPProcessor]
) -> None:
    model, processor = clip
    paths = _write_images(tmp_path, [9, 25, 30])

    batched = encode_images(model, processor, paths, batch_size=2, num_threads=1)
    single = encode_images(model, processor, paths, batch_size=1, num_threads=1)
    reversed_order = encode_images(model, processor, paths[::-1], 2, num_threads=1)

    assert batched.shape == (3, EMBED_DIM)
    assert batched.dtype == torch.float32
    assert torch.allclose(batched, single, atol=1e-5)
    assert torch.allclose(batched, reversed_order.flip(0), atol=1e-5)


def test_encode_images_gives_same_rows_with_several_threads(
    tmp_path: Path, clip: tuple[CLIPModel, CLIPProcessor]
) -> None:
    model, processor = clip
    paths = _write_images(tmp_path, [9, 25, 30, 42])

    threaded = encode_images(model, processor, paths, batch_size=3, num_threads=4)
    serial = encode_images(model, processor, paths, batch_size=3, num_threads=1)

    assert torch.allclose(threaded, serial, atol=1e-6)


def test_encode_images_accepts_empty_list(
    clip: tuple[CLIPModel, CLIPProcessor],
) -> None:
    model, processor = clip

    features = encode_images(model, processor, [], batch_size=2, num_threads=2)

    assert features.shape == (0, EMBED_DIM)


def test_encode_images_converts_grayscale_to_rgb(
    tmp_path: Path, clip: tuple[CLIPModel, CLIPProcessor]
) -> None:
    model, processor = clip
    gray_path = tmp_path / "gray.jpg"
    rgb_path = tmp_path / "rgb.jpg"
    Image.new("L", (64, 48), 128).save(gray_path)
    Image.new("RGB", (64, 48), (128, 128, 128)).save(rgb_path)

    paths = [gray_path, rgb_path]
    features = encode_images(model, processor, paths, 2, num_threads=1)

    assert features.shape == (2, EMBED_DIM)
    assert torch.allclose(features[0], features[1], atol=1e-3)


def test_save_and_load_set_round_trip(tmp_path: Path) -> None:
    features = torch.randn(3, EMBED_DIM)
    path = tmp_path / "val.pt"

    save_set(path, [9, 25, 30], features)
    image_ids, loaded = load_set(path)

    assert image_ids == [9, 25, 30]
    assert torch.equal(loaded, features)


def test_save_set_rejects_row_count_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_set(tmp_path / "val.pt", [9, 25], torch.randn(3, EMBED_DIM))


def test_check_cache_passes_for_aligned_rows(
    tmp_path: Path, clip: tuple[CLIPModel, CLIPProcessor]
) -> None:
    model, processor = clip
    image_ids = [9, 25, 30]
    paths = _write_images(tmp_path, image_ids)
    features = encode_images(model, processor, paths, batch_size=3, num_threads=1)

    result = check_cache(model, processor, paths, features, sample_size=3, seed=0)

    assert result["images"] == 3
    assert result["max_abs_diff"] < 1e-4
    assert result["min_cosine"] > 0.9999


def test_check_cache_detects_swapped_rows(
    tmp_path: Path, clip: tuple[CLIPModel, CLIPProcessor]
) -> None:
    model, processor = clip
    paths = _write_images(tmp_path, [9, 25, 30])
    features = encode_images(model, processor, paths, batch_size=3, num_threads=1)
    swapped = features[[1, 0, 2]]

    result = check_cache(model, processor, paths, swapped, sample_size=3, seed=0)

    assert result["max_abs_diff"] > 1e-2


def _run_main(tmp_path: Path, extra: list[str] | None = None) -> tuple[Path, Path]:
    splits = tmp_path / "splits"
    splits.mkdir()
    for name, ids in {"train": [9, 25], "val": [30], "test": [42]}.items():
        (splits / f"{name}_ids.json").write_text(json.dumps(ids) + "\n")
    _write_images(tmp_path / "train2017", [9, 25, 30])
    _write_images(tmp_path / "val2017", [42])
    out_dir = tmp_path / "embeddings"
    report_path = tmp_path / "results" / "embeddings_report.json"
    main(
        [
            "--splits-dir", str(splits),
            "--train2017-dir", str(tmp_path / "train2017"),
            "--val2017-dir", str(tmp_path / "val2017"),
            "--out-dir", str(out_dir),
            "--report", str(report_path),
            "--device", "cpu",
            "--batch-size", "2",
            "--num-threads", "1",
            "--check-size", "2",
            *(extra or []),
        ]
    )
    return out_dir, report_path


def test_main_writes_one_file_per_set_and_report(tmp_path: Path) -> None:
    out_dir, report_path = _run_main(tmp_path)

    for name, expected_ids in {"train": [9, 25], "val": [30], "test": [42]}.items():
        image_ids, features = load_set(out_dir / f"{name}.pt")
        assert image_ids == expected_ids
        assert features.shape == (len(expected_ids), EMBED_DIM)

    report = json.loads(report_path.read_text())
    split_bytes = (tmp_path / "splits" / "train_ids.json").read_bytes()
    assert report["train"]["images"] == 2
    assert report["train"]["split_sha256"] == hashlib.sha256(split_bytes).hexdigest()
    assert report["test"]["check"]["passed"] is True
    assert report["embed_dim"] == EMBED_DIM
    assert len(report["git_commit"]) == 40


def test_main_fails_before_encoding_when_images_are_missing(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    splits.mkdir()
    for name, ids in {"train": [9], "val": [25], "test": [42]}.items():
        (splits / f"{name}_ids.json").write_text(json.dumps(ids) + "\n")
    _write_images(tmp_path / "train2017", [9])
    _write_images(tmp_path / "val2017", [42])

    with pytest.raises(FileNotFoundError):
        main(
            [
                "--splits-dir", str(splits),
                "--train2017-dir", str(tmp_path / "train2017"),
                "--val2017-dir", str(tmp_path / "val2017"),
                "--out-dir", str(tmp_path / "embeddings"),
                "--report", str(tmp_path / "report.json"),
                "--device", "cpu",
                "--num-threads", "1",
            ]
        )
    assert not (tmp_path / "embeddings").exists()
