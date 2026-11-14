import argparse
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from src.data.captions import load_captions
from src.data.cache_embeddings import load_set
from src.data.split import REPO_ROOT, TRAIN_CAPTIONS, git_state
from src.eval.recall_at_k import caption_ranks, recall_at_k
from src.features import MAX_TEXT_TOKENS, MODEL_NAME, load_clip, text_features
from src.features import tokenize

logger = logging.getLogger(__name__)

RECALL_KS = (1, 5, 10)


@dataclass(frozen=True)
class CompExample:
    image_id: int
    caption_id: int
    caption: str
    negative: str


@dataclass(frozen=True)
class ValidationSet:
    image_ids: tuple[int, ...]
    image_features: torch.Tensor
    captions: tuple[str, ...]
    caption_image_ids: tuple[int, ...]
    comp_examples: tuple[CompExample, ...]


def select_comp_examples(records: list[dict[str, Any]], seed: int) -> list[CompExample]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=lambda r: (r["image_id"], r["caption_id"])):
        by_image[record["image_id"]].append(record)
    rng = random.Random(seed)
    examples = []
    for image_id in sorted(by_image):
        pick = rng.choice(by_image[image_id])
        examples.append(
            CompExample(image_id, pick["caption_id"], pick["caption"], pick["negative"])
        )
    return examples


@torch.no_grad()
def encode_texts(
    model: CLIPModel, processor: CLIPProcessor, texts: list[str], batch_size: int
) -> torch.Tensor:
    device = next(model.parameters()).device
    chunks = [torch.empty(0, model.config.projection_dim)]
    for start in range(0, len(texts), batch_size):
        tokens = tokenize(processor, texts[start : start + batch_size]).to(device)
        features = text_features(model, tokens["input_ids"], tokens["attention_mask"])
        chunks.append(features.float().cpu())
    return torch.cat(chunks)


def compositional_correct(
    image_features: torch.Tensor,
    caption_features: torch.Tensor,
    negative_features: torch.Tensor,
) -> torch.Tensor:
    caption_scores = F.cosine_similarity(image_features, caption_features)
    negative_scores = F.cosine_similarity(image_features, negative_features)
    return caption_scores > negative_scores


def evaluate(
    model: CLIPModel, processor: CLIPProcessor, data: ValidationSet, batch_size: int
) -> dict[str, float]:
    captions = encode_texts(model, processor, list(data.captions), batch_size)
    ranks = caption_ranks(
        captions, data.image_features, data.caption_image_ids, data.image_ids
    )
    recall = recall_at_k(ranks, RECALL_KS)

    row_of = {image_id: row for row, image_id in enumerate(data.image_ids)}
    rows = [row_of[example.image_id] for example in data.comp_examples]
    positives = [example.caption for example in data.comp_examples]
    negatives = [example.negative for example in data.comp_examples]
    correct = compositional_correct(
        data.image_features[rows],
        encode_texts(model, processor, positives, batch_size),
        encode_texts(model, processor, negatives, batch_size),
    )
    return {
        "comp_accuracy": correct.float().mean().item(),
        "comp_examples": len(rows),
        **{f"t2i_r{k}": value for k, value in recall.items()},
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_validation_set(
    embeddings_path: Path, captions_path: Path, negatives_path: Path, seed: int
) -> ValidationSet:
    image_ids, image_feats = load_set(embeddings_path)
    in_slice = set(image_ids)
    captions = [c for c in load_captions(captions_path) if c.image_id in in_slice]
    records = _read_jsonl(negatives_path)
    outside = {r["image_id"] for r in records} - in_slice
    if outside:
        raise ValueError(f"{len(outside)} negative image IDs are not in the slice")
    return ValidationSet(
        image_ids=tuple(image_ids),
        image_features=image_feats,
        captions=tuple(caption.text for caption in captions),
        caption_image_ids=tuple(caption.image_id for caption in captions),
        comp_examples=tuple(select_comp_examples(records, seed)),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute the base CLIP validation metrics."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("embeddings/clip-vit-base-patch32/val.pt"),
    )
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument(
        "--negatives", type=Path, default=Path("data/negatives/val.jsonl")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/val_baseline.json")
    )
    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    captions_path = args.annotations_dir / TRAIN_CAPTIONS
    data = load_validation_set(
        args.embeddings, captions_path, args.negatives, args.seed
    )
    model, processor = load_clip(args.device)
    metrics = evaluate(model, processor, data, args.batch_size)
    comp_ids = json.dumps([example.caption_id for example in data.comp_examples])

    report: dict[str, Any] = {
        "model": MODEL_NAME,
        "seed": args.seed,
        "device": args.device,
        "max_text_tokens": MAX_TEXT_TOKENS,
        "rank_ties": "counted against the true image",
        "comp_rule": "correct when cos(image, caption) > cos(image, negative)",
        "counts": {
            "images": len(data.image_ids),
            "captions": len(data.captions),
            "comp_examples": len(data.comp_examples),
        },
        "metrics": metrics,
        "inputs": {
            "embeddings_sha256": _sha256(args.embeddings),
            "negatives_sha256": _sha256(args.negatives),
            "comp_caption_ids_sha256": hashlib.sha256(comp_ids.encode()).hexdigest(),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    report.update(git_state(REPO_ROOT))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Base CLIP validation: %s", metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
