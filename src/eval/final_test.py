import argparse
import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import CLIPModel, CLIPProcessor

from src.data.cache_embeddings import load_set
from src.data.captions import load_captions
from src.data.leakage import coco_id_from_filename
from src.data.split import REPO_ROOT, TEST_CAPTIONS, git_state
from src.eval.bootstrap import bootstrap_groups, cluster_totals
from src.eval.recall_at_k import caption_ranks, image_ranks
from src.eval.validation import compositional_correct, encode_texts
from src.features import load_clip

logger = logging.getLogger(__name__)

GROUPS = ("base", "inbatch", "hard")
COMPARISONS = (("inbatch", "base"), ("hard", "base"), ("hard", "inbatch"))
RECALL_KS = (1, 5, 10)


@dataclass(frozen=True)
class SugarExample:
    category: str
    key: int
    image_id: int
    caption: str
    negative: str


def load_sugarcrepe(folder: Path) -> list[SugarExample]:
    examples = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text())
        for key in sorted(data, key=int):
            item = data[key]
            examples.append(
                SugarExample(
                    path.stem,
                    int(key),
                    coco_id_from_filename(item["filename"]),
                    item["caption"],
                    item["negative_caption"],
                )
            )
    return examples


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def check_ready(repo: Path, selection: Path, out: Path) -> None:
    if out.exists():
        raise SystemExit(f"{out} exists: the test sets are evaluated only once")
    if _git(repo, "ls-files", "--error-unmatch", str(selection)).returncode != 0:
        raise SystemExit(f"{selection} is not committed")
    if _git(repo, "status", "--porcelain", "--", str(selection)).stdout.strip():
        raise SystemExit(f"{selection} has uncommitted changes")
    if git_state(repo)["git_dirty"]:
        raise SystemExit("src/ has uncommitted changes")


def evaluate_model(
    model: CLIPModel,
    processor: CLIPProcessor,
    examples: list[SugarExample],
    image_ids: list[int],
    image_features: torch.Tensor,
    captions: list[str],
    caption_image_ids: list[int],
    batch_size: int,
) -> dict[str, np.ndarray]:
    row_of = {image_id: row for row, image_id in enumerate(image_ids)}
    images = image_features[[row_of[example.image_id] for example in examples]]
    positives = [example.caption for example in examples]
    negatives = [example.negative for example in examples]
    correct = compositional_correct(
        images,
        encode_texts(model, processor, positives, batch_size),
        encode_texts(model, processor, negatives, batch_size),
    )
    texts = encode_texts(model, processor, captions, batch_size)
    return {
        "sugarcrepe_correct": correct.numpy(),
        "t2i_ranks": caption_ranks(
            texts, image_features, caption_image_ids, image_ids
        ).numpy(),
        "i2t_ranks": image_ranks(
            texts, image_features, caption_image_ids, image_ids
        ).numpy(),
    }


def _model_metrics(output: dict[str, np.ndarray], categories: np.ndarray) -> dict:
    correct = output["sugarcrepe_correct"].astype(float)
    sugarcrepe = {
        category: float(correct[categories == category].mean())
        for category in sorted(set(categories))
    }
    sugarcrepe["macro"] = float(np.mean(list(sugarcrepe.values())))
    return {
        "sugarcrepe": sugarcrepe,
        "t2i": {f"r{k}": float((output["t2i_ranks"] < k).mean()) for k in RECALL_KS},
        "i2t": {f"r{k}": float((output["i2t_ranks"] < k).mean()) for k in RECALL_KS},
    }


def _aggregate(per_seed: list[dict]) -> dict:
    table: dict[str, dict] = {}
    for family, names in per_seed[0].items():
        table[family] = {}
        for name in names:
            values = [metrics[family][name] for metrics in per_seed]
            table[family][name] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else None,
                "per_seed": values,
            }
    return table


def _seed_average(outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        "sugarcrepe": np.mean([o["sugarcrepe_correct"] for o in outputs], axis=0),
        **{
            f"{direction}_r{k}": np.mean(
                [o[f"{direction}_ranks"] < k for o in outputs], axis=0
            )
            for direction in ("t2i", "i2t")
            for k in RECALL_KS
        },
    }


def _paired_deltas(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    categories: np.ndarray,
    example_images: np.ndarray,
    caption_images: np.ndarray,
    images: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict:
    diff = a["sugarcrepe"] - b["sugarcrepe"]
    groups = {
        category: (diff[categories == category], example_images[categories == category])
        for category in sorted(set(categories))
    }
    sums, counts, _ = cluster_totals(groups)
    result = {
        "sugarcrepe": bootstrap_groups(
            sums, counts, list(groups), n_resamples, seed, macro=True
        )
    }
    for direction, clusters in (("t2i", caption_images), ("i2t", images)):
        names = [f"r{k}" for k in RECALL_KS]
        groups = {
            name: (a[f"{direction}_{name}"] - b[f"{direction}_{name}"], clusters)
            for name in names
        }
        sums, counts, _ = cluster_totals(groups)
        result[direction] = bootstrap_groups(
            sums, counts, names, n_resamples, seed, macro=False
        )
    return result


def summarize(
    outputs: dict[str, list[dict[str, np.ndarray]]],
    examples: list[SugarExample],
    caption_image_ids: list[int],
    image_ids: list[int],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    categories = np.array([example.category for example in examples])
    table = {
        group: _aggregate([_model_metrics(o, categories) for o in outputs[group]])
        for group in GROUPS
    }
    averaged = {group: _seed_average(outputs[group]) for group in GROUPS}
    example_images = np.array([example.image_id for example in examples])
    deltas = {
        f"{a}_vs_{b}": _paired_deltas(
            averaged[a],
            averaged[b],
            categories,
            example_images,
            np.array(caption_image_ids),
            np.array(image_ids),
            n_resamples,
            seed,
        )
        for a, b in COMPARISONS
    }
    return {"table": table, "deltas": deltas}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_adapter(path: Path, device: str) -> CLIPModel:
    base, _ = load_clip("cpu")
    return PeftModel.from_pretrained(base, str(path)).to(device).eval()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected checkpoints on the test sets, once."
    )
    parser.add_argument(
        "--selection", type=Path, default=Path("results/selection.json")
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("embeddings/clip-vit-base-patch32/test.pt"),
    )
    parser.add_argument("--sugarcrepe-dir", type=Path, default=Path("data/sugarcrepe"))
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/test_results.json"))
    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, require_ready: bool = True) -> None:
    args = _parse_args(argv)
    if require_ready:
        check_ready(REPO_ROOT, args.selection, args.out)
    started = time.perf_counter()
    selection = json.loads(args.selection.read_text())
    checkpoints = {
        arm: [args.checkpoint_root / path for path in choice["checkpoints"]]
        for arm, choice in selection["arms"].items()
    }
    all_paths = [path for paths in checkpoints.values() for path in paths]
    missing = [str(path) for path in all_paths if not path.is_dir()]
    if missing:
        raise SystemExit(f"Missing checkpoints: {missing}")

    image_ids, image_features = load_set(args.embeddings)
    in_test = set(image_ids)
    examples = load_sugarcrepe(args.sugarcrepe_dir)
    if any(example.image_id not in in_test for example in examples):
        raise SystemExit("A SugarCrepe image is not in the test embeddings")
    captions = load_captions(args.annotations_dir / TEST_CAPTIONS)
    captions = [caption for caption in captions if caption.image_id in in_test]
    texts = [caption.text for caption in captions]
    caption_image_ids = [caption.image_id for caption in captions]

    def run(model: CLIPModel, processor: CLIPProcessor) -> dict[str, np.ndarray]:
        return evaluate_model(
            model, processor, examples, image_ids, image_features,
            texts, caption_image_ids, args.batch_size,
        )

    base, processor = load_clip(args.device)
    outputs = {"base": [run(base, processor)]}
    names = ["base"]
    for arm in ("inbatch", "hard"):
        outputs[arm] = []
        for seed, path in zip(selection["arms"][arm]["seeds"], checkpoints[arm]):
            logger.info("evaluating %s seed %s: %s", arm, seed, path)
            outputs[arm].append(run(_load_adapter(path, args.device), processor))
            names.append(f"{arm}_{seed}")

    summary = summarize(
        outputs, examples, caption_image_ids, image_ids, args.n_resamples, args.seed
    )
    stacked = [outputs["base"][0], *outputs["inbatch"], *outputs["hard"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        models=np.array(names),
        sugarcrepe_correct=np.stack([o["sugarcrepe_correct"] for o in stacked]),
        t2i_ranks=np.stack([o["t2i_ranks"] for o in stacked]),
        i2t_ranks=np.stack([o["i2t_ranks"] for o in stacked]),
        sugarcrepe_category=np.array([e.category for e in examples]),
        sugarcrepe_key=np.array([e.key for e in examples]),
        sugarcrepe_image_id=np.array([e.image_id for e in examples]),
        caption_image_ids=np.array(caption_image_ids),
        image_ids=np.array(image_ids),
    )
    results = {
        "counts": {
            "sugarcrepe": len(examples),
            "images": len(image_ids),
            "captions": len(texts),
            "models": len(names),
        },
        "models": names,
        "bootstrap": {
            "unit": "image ID",
            "resamples": args.n_resamples,
            "interval": "95% percentile",
            "seed": args.seed,
            "seed_handling": "per-example mean over the 3 seeds",
        },
        "rank_ties": "counted against the true match",
        **summary,
        "inputs": {
            "selection_sha256": _sha256(args.selection),
            "embeddings_sha256": _sha256(args.embeddings),
            "adapters_sha256": {
                str(path): _sha256(path / "adapter_model.safetensors")
                for paths in checkpoints.values()
                for path in paths
            },
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        **git_state(REPO_ROOT),
    }
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("Results: %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
