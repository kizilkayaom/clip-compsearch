import argparse
import csv
import hashlib
import json
import logging
import random
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from spacy.language import Language

from src.data.leakage import load_id_sets
from src.data.negatives import ENTAILMENT, NLI_MODEL_NAME, SYMMETRIC_RELATIONS
from src.data.negatives import Negative, get_nli_model, negative_from_doc, nli_scores
from src.data.negatives import nlp as default_nlp
from src.data.split import REPO_ROOT, TRAIN_CAPTIONS, git_state

logger = logging.getLogger(__name__)

ScoreFn = Callable[[list[tuple[str, str]]], list[list[float]]]
SETS = ("train", "val")
SAMPLE_FIELDS = (
    "caption_id",
    "image_id",
    "swap_type",
    "caption",
    "negative",
    "label",
    "break_type",
)
PAIR_FIELDS = ("image_id", "caption_id", "caption", "negative", "swap_type")
CAPTION_PREPROCESSING = "collapse whitespace runs to one space and strip the ends"
NLI_LABEL_ORDER = ("contradiction", "entailment", "neutral")
PROGRESS_EVERY = 50_000


@dataclass(frozen=True)
class Caption:
    caption_id: int
    image_id: int
    text: str


def load_captions(path: Path) -> list[Caption]:
    with path.open() as f:
        annotations = json.load(f)["annotations"]
    captions = [
        Caption(entry["id"], entry["image_id"], " ".join(entry["caption"].split()))
        for entry in annotations
    ]
    return sorted(captions, key=lambda caption: (caption.image_id, caption.caption_id))


def assign_sets(
    captions: list[Caption], id_sets: dict[str, set[int]]
) -> dict[str, list[Caption]]:
    set_of_image = {image_id: name for name, ids in id_sets.items() for image_id in ids}
    grouped: dict[str, list[Caption]] = {name: [] for name in id_sets}
    for caption in captions:
        name = set_of_image.get(caption.image_id)
        if name is None:
            raise ValueError(f"Image {caption.image_id} is in no split set")
        grouped[name].append(caption)
    return grouped


def _find_candidates(
    captions: list[Caption], nlp: Language, batch_size: int, n_process: int
) -> list[tuple[Caption, Negative]]:
    candidates = []
    texts = (caption.text for caption in captions)
    docs = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
    for index, (caption, doc) in enumerate(zip(captions, docs), start=1):
        negative = negative_from_doc(doc)
        if negative is not None:
            candidates.append((caption, negative))
        if index % PROGRESS_EVERY == 0:
            logger.info("spaCy: %d/%d captions", index, len(captions))
    return candidates


def _score(
    pairs: list[tuple[Caption, Negative]], score_fn: ScoreFn, chunk_size: int
) -> list[list[float]]:
    scores: list[list[float]] = []
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start : start + chunk_size]
        texts = [(caption.text, negative.text) for caption, negative in chunk]
        scores.extend(score_fn(texts))
        logger.info("NLI: %d/%d pairs", len(scores), len(pairs))
    return scores


def _record(
    caption: Caption, negative: Negative, scores: list[float]
) -> dict[str, Any]:
    return {
        "image_id": caption.image_id,
        "caption_id": caption.caption_id,
        "caption": caption.text,
        "negative": negative.text,
        "swap_type": negative.swap_type,
        "nli_scores": [round(score, 4) for score in scores],
    }


def process_set(
    captions: list[Caption],
    nlp: Language,
    score_fn: ScoreFn,
    batch_size: int = 256,
    n_process: int = 1,
    chunk_size: int = 20_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = _find_candidates(captions, nlp, batch_size, n_process)
    kept = [(caption, neg) for caption, neg in candidates if neg.relation is None]
    scores = _score(kept, score_fn, chunk_size)
    swap_counts = Counter(negative.swap_type for _, negative in candidates)
    relations = Counter(neg.relation for _, neg in candidates if neg.relation)
    entailed = sum(max(range(3), key=row.__getitem__) == ENTAILMENT for row in scores)
    summary = {
        "total_captions": len(captions),
        "noun_swaps": swap_counts["noun"],
        "adjective_swaps": swap_counts["adjective"],
        "no_swap": len(captions) - len(candidates),
        "rule_rejected": len(candidates) - len(kept),
        "rule_rejected_by_relation": dict(sorted(relations.items())),
        "kept_pairs": len(kept),
        "nli_entailment_in_kept": entailed,
        "images": len({caption.image_id for caption in captions}),
        "images_with_kept_pair": len({caption.image_id for caption, _ in kept}),
    }
    records = [
        _record(caption, negative, row)
        for (caption, negative), row in zip(kept, scores, strict=True)
    ]
    return records, summary


def draw_grammar_sample(
    records: list[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    return random.Random(seed).sample(records, min(size, len(records)))


def pairs_sha256(records: list[dict[str, Any]]) -> str:
    lines = (
        json.dumps({field: record[field] for field in PAIR_FIELDS}) + "\n"
        for record in records
    )
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(record) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_grammar_sample(path: Path, sample: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in sample:
            writer.writerow({**record, "label": "", "break_type": ""})


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hard negatives for the train and validation captions."
    )
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/negatives"))
    parser.add_argument(
        "--report", type=Path, default=Path("results/negatives_report.json")
    )
    parser.add_argument(
        "--grammar-sample", type=Path, default=Path("results/grammar_sample.csv")
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--spacy-batch-size", type=int, default=256)
    parser.add_argument("--n-process", type=int, default=1)
    parser.add_argument("--nli-batch-size", type=int, default=256)
    parser.add_argument("--nli-chunk-size", type=int, default=20_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, score_fn: ScoreFn | None = None) -> None:
    args = _parse_args(argv)
    started = time.perf_counter()
    nli_device = "injected"
    if score_fn is None:
        score_fn = partial(nli_scores, batch_size=args.nli_batch_size)
        nli_device = str(get_nli_model().device)

    id_sets = load_id_sets(args.splits_dir)
    captions = assign_sets(
        load_captions(args.annotations_dir / TRAIN_CAPTIONS),
        {name: id_sets[name] for name in SETS},
    )

    meta = default_nlp.meta
    report: dict[str, Any] = {
        "seed": args.seed,
        "limit": args.limit,
        "caption_preprocessing": CAPTION_PREPROCESSING,
        "spacy_model": f"{meta['lang']}_{meta['name']}-{meta['version']}",
        "filter": {
            "rule": "reject swaps joined by a coordinator or a symmetric relation",
            "coordinators": ["and", "or"],
            "symmetric_relations": sorted(SYMMETRIC_RELATIONS),
        },
        "nli_model": NLI_MODEL_NAME,
        "nli_device": nli_device,
        "nli_use": "scores stored per pair, not used for filtering",
        "nli_label_order": list(NLI_LABEL_ORDER),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_records: list[dict[str, Any]] = []
    for name in SETS:
        selected = captions[name][: args.limit]
        logger.info("%s: %d captions", name, len(selected))
        records, summary = process_set(
            selected,
            default_nlp,
            score_fn,
            batch_size=args.spacy_batch_size,
            n_process=args.n_process,
            chunk_size=args.nli_chunk_size,
        )
        jsonl_path = args.out_dir / f"{name}.jsonl"
        report[name] = {
            **summary,
            "sha256": _write_jsonl(jsonl_path, records),
            "pairs_sha256": pairs_sha256(records),
        }
        if name == "train":
            train_records = records

    sample = draw_grammar_sample(train_records, args.sample_size, args.seed)
    _write_grammar_sample(args.grammar_sample, sample)
    report["grammar_sample"] = {"source": "train", "size": len(sample)}
    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    report.update(git_state(REPO_ROOT))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Done in %.0f s. Report: %s", report["elapsed_seconds"], args.report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
