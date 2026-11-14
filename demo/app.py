import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.cache_embeddings import image_path, load_set
from src.eval.final_test import SugarExample, load_sugarcrepe
from src.eval.validation import encode_texts
from src.features import load_adapter, load_clip

DEFAULT_CHECKPOINT = Path("checkpoints/hard_lr5e-04_seed0_epoch2.5")
MODES = ("Fine-tune wins", "Both fail", "Both succeed", "Random")


def rank_gallery(
    query: torch.Tensor, gallery: torch.Tensor, image_ids: Sequence[int], k: int
) -> tuple[list[int], list[float]]:
    scores = F.normalize(query, dim=-1) @ F.normalize(gallery, dim=-1).T
    top = torch.topk(scores[0], k)
    return [image_ids[i] for i in top.indices.tolist()], top.values.tolist()


def verdict(caption_score: float, negative_score: float) -> str:
    return "Pass" if caption_score > negative_score else "Fail"


def _pair_scores(
    model, processor, image: torch.Tensor, texts: list[str]
) -> list[float]:
    features = encode_texts(model, processor, texts, batch_size=2)
    return F.cosine_similarity(image[None, :], features).tolist()


def build_demo(args: argparse.Namespace):
    import gradio as gr

    base, processor = load_clip(args.device)
    tuned = load_adapter(args.checkpoint, args.device)
    image_ids, gallery = load_set(args.gallery)
    row_of = {image_id: row for row, image_id in enumerate(image_ids)}
    examples: list[SugarExample] = load_sugarcrepe(args.sugarcrepe_dir)
    rng = random.Random(0)

    def gallery_items(model, query: str, k: int) -> list[tuple[Image.Image, str]]:
        features = encode_texts(model, processor, [query], batch_size=1)
        ids, scores = rank_gallery(features, gallery, image_ids, int(k))
        return [
            (Image.open(image_path(args.images_dir, i)), f"#{rank} ({score:.3f})")
            for rank, (i, score) in enumerate(zip(ids, scores), start=1)
        ]

    def search(query: str, k: int):
        return gallery_items(base, query, k), gallery_items(tuned, query, k)

    def match_example(mode: str):
        for _ in range(200):
            example = rng.choice(examples)
            image = gallery[row_of[example.image_id]]
            texts = [example.caption, example.negative]
            base_scores = _pair_scores(base, processor, image, texts)
            tuned_scores = _pair_scores(tuned, processor, image, texts)
            base_ok = verdict(*base_scores) == "Pass"
            tuned_ok = verdict(*tuned_scores) == "Pass"
            wanted = {
                "Fine-tune wins": not base_ok and tuned_ok,
                "Both fail": not base_ok and not tuned_ok,
                "Both succeed": base_ok and tuned_ok,
                "Random": True,
            }
            if wanted[mode]:
                break
        table = (
            f"**Category:** {example.category}\n\n"
            f"**Caption:** {example.caption}\n\n"
            f"**Negative:** {example.negative}\n\n"
            "| Model | Caption | Negative | Verdict |\n|---|---|---|---|\n"
            f"| Base CLIP | {base_scores[0]:.4f} | {base_scores[1]:.4f} "
            f"| {verdict(*base_scores)} |\n"
            f"| Fine-tuned | {tuned_scores[0]:.4f} | {tuned_scores[1]:.4f} "
            f"| {verdict(*tuned_scores)} |\n"
        )
        return Image.open(image_path(args.images_dir, example.image_id)), table

    with gr.Blocks(title="CLIP-CompSearch") as demo:
        with gr.Tab("Search"):
            gr.Markdown("Search 5,000 COCO val2017 images; none were used in training.")
            with gr.Row():
                query = gr.Textbox(label="Query", scale=4)
                k = gr.Slider(3, 10, value=5, step=1, label="Results", scale=1)
            button = gr.Button("Search", variant="primary")
            with gr.Row():
                base_gallery = gr.Gallery(label="Base CLIP", columns=5, height=300)
                tuned_gallery = gr.Gallery(label="Fine-tuned", columns=5, height=300)
            button.click(search, [query, k], [base_gallery, tuned_gallery])
        with gr.Tab("Matching"):
            gr.Markdown("SugarCrepe: is the caption scored above its negative?")
            with gr.Row():
                mode = gr.Dropdown(list(MODES), value=MODES[0], label="Show")
                next_button = gr.Button("Next example", variant="primary")
            with gr.Row():
                shown = gr.Image(label="Image", height=400)
                result = gr.Markdown()
            next_button.click(match_example, mode, [shown, result])
    return demo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned CLIP.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--gallery",
        type=Path,
        default=Path("embeddings/clip-vit-base-patch32/test.pt"),
    )
    parser.add_argument("--images-dir", type=Path, default=Path("data/coco/val2017"))
    parser.add_argument("--sugarcrepe-dir", type=Path, default=Path("data/sugarcrepe"))
    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    return parser.parse_args()


if __name__ == "__main__":
    build_demo(_parse_args()).launch()
