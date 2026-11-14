import argparse
import hashlib
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.data.split import REPO_ROOT, git_state
from src.features import EMBED_DIM, MODEL_NAME, image_features, load_clip

logger = logging.getLogger(__name__)

SET_ORDER = ("val", "test", "train")
CHECK_TOLERANCE = 1e-4
PROGRESS_EVERY = 50


def _pixel_values(processor: CLIPProcessor, path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    return processor(images=rgb, return_tensors="pt")["pixel_values"][0]


def _load_batch(
    pool: ThreadPoolExecutor, processor: CLIPProcessor, paths: list[Path]
) -> torch.Tensor:
    return torch.stack(list(pool.map(partial(_pixel_values, processor), paths)))


def image_path(image_dir: Path, image_id: int) -> Path:
    return image_dir / f"{image_id:012d}.jpg"


def missing_images(image_ids: list[int], image_dir: Path) -> list[int]:
    return [i for i in image_ids if not image_path(image_dir, i).is_file()]


def _device(model: CLIPModel) -> torch.device:
    return next(model.parameters()).device


def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    paths: list[Path],
    batch_size: int,
    num_threads: int,
) -> torch.Tensor:
    batches = [
        paths[start : start + batch_size] for start in range(0, len(paths), batch_size)
    ]
    device = _device(model)
    chunks = [torch.empty(0, EMBED_DIM)]
    with (
        ThreadPoolExecutor(max(num_threads, 1)) as pool,
        ThreadPoolExecutor(1) as prefetch,
    ):
        submit = partial(prefetch.submit, _load_batch, pool, processor)
        pending = [submit(batch) for batch in batches[:1]]
        for index, _ in enumerate(batches, start=1):
            pixel_values = pending.pop().result()
            if index < len(batches):
                pending.append(submit(batches[index]))
            chunks.append(image_features(model, pixel_values.to(device)).float().cpu())
            if index % PROGRESS_EVERY == 0:
                done = min(index * batch_size, len(paths))
                logger.info("encoded %d/%d images", done, len(paths))
    return torch.cat(chunks)


def save_set(path: Path, image_ids: list[int], features: torch.Tensor) -> None:
    if len(image_ids) != len(features):
        raise ValueError(f"{len(image_ids)} image IDs but {len(features)} rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL_NAME,
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "features": features.contiguous(),
    }
    torch.save(payload, path)


def load_set(path: Path) -> tuple[list[int], torch.Tensor]:
    payload = torch.load(path)
    return payload["image_ids"].tolist(), payload["features"]


def check_cache(
    model: CLIPModel,
    processor: CLIPProcessor,
    paths: list[Path],
    features: torch.Tensor,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    rows = random.Random(seed).sample(range(len(paths)), min(sample_size, len(paths)))
    sampled = [paths[row] for row in rows]
    reference = encode_images(model, processor, sampled, batch_size=1, num_threads=1)
    cached = features[rows]
    return {
        "images": len(rows),
        "max_abs_diff": float((cached - reference).abs().max()),
        "min_cosine": float(F.cosine_similarity(cached, reference).min()),
    }


def _features_sha256(features: torch.Tensor) -> str:
    return hashlib.sha256(features.numpy().tobytes()).hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache CLIP image embeddings for the train, val and test sets."
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--train2017-dir", type=Path, required=True)
    parser.add_argument("--val2017-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("embeddings/clip-vit-base-patch32")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/embeddings_report.json")
    )
    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--check-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def _load_sets(args: argparse.Namespace) -> dict[str, tuple[list[int], Path, str]]:
    image_dirs = {"train": args.train2017_dir, "val": args.train2017_dir}
    image_dirs["test"] = args.val2017_dir
    sets = {}
    for name in SET_ORDER:
        split_path = args.splits_dir / f"{name}_ids.json"
        split_bytes = split_path.read_bytes()
        image_ids = json.loads(split_bytes)[: args.limit]
        missing = missing_images(image_ids, image_dirs[name])
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} {name} images missing in {image_dirs[name]}, "
                f"first: {missing[:5]}"
            )
        sha = hashlib.sha256(split_bytes).hexdigest()
        sets[name] = (image_ids, image_dirs[name], sha)
    return sets


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    sets = _load_sets(args)
    model, processor = load_clip(args.device)
    reference_model, _ = load_clip("cpu")

    report: dict[str, Any] = {
        "model": MODEL_NAME,
        "embed_dim": EMBED_DIM,
        "feature": "get_image_features(...).pooler_output, not normalized",
        "preprocessing": "PIL convert RGB, then CLIPProcessor",
        "device": args.device,
        "batch_size": args.batch_size,
        "num_threads": args.num_threads,
        "limit": args.limit,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "check_tolerance": CHECK_TOLERANCE,
        "check_reference": "cpu, batch size 1",
    }
    failed = []
    for name in SET_ORDER:
        image_ids, image_dir, split_sha = sets[name]
        paths = [image_path(image_dir, image_id) for image_id in image_ids]
        logger.info("%s: %d images", name, len(paths))
        started = time.perf_counter()
        features = encode_images(
            model, processor, paths, args.batch_size, args.num_threads
        )
        elapsed = time.perf_counter() - started
        save_set(args.out_dir / f"{name}.pt", image_ids, features)

        check = check_cache(
            reference_model, processor, paths, features, args.check_size, args.seed
        )
        check["passed"] = check["max_abs_diff"] <= CHECK_TOLERANCE
        if not check["passed"]:
            failed.append(name)
        report[name] = {
            "images": len(image_ids),
            "shape": list(features.shape),
            "split_sha256": split_sha,
            "features_sha256": _features_sha256(features),
            "elapsed_seconds": round(elapsed, 1),
            "images_per_second": round(len(image_ids) / max(elapsed, 1e-9), 1),
            "check": check,
        }
        logger.info("%s done in %.0f s, check: %s", name, elapsed, check)

    report.update(git_state(REPO_ROOT))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise SystemExit(f"Cache check failed for: {failed}")
    logger.info("Report: %s", args.report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
