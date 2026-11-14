import argparse
import hashlib
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import peft
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPProcessor

from src.data.split import REPO_ROOT, TRAIN_CAPTIONS, git_state
from src.eval.validation import ValidationSet, evaluate, load_validation_set
from src.features import load_clip, text_features, tokenize
from src.finetune.data import TrainData, TrainPair, epoch_batches, load_train_data
from src.finetune.loss import contrastive_loss

logger = logging.getLogger(__name__)

ARMS = ("inbatch", "hard")
LORA_TARGETS = r".*text_model.*\.(q_proj|v_proj)"


@dataclass(frozen=True)
class TrainConfig:
    arm: str
    lr: float
    seed: int
    epochs: int = 3
    batch_size: int = 256
    warmup_steps: int = 50
    weight_decay: float = 0.01
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    eval_batch_size: int = 256
    gradient_checkpointing: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, not {self.arm!r}")


def run_name(config: TrainConfig) -> str:
    return f"{config.arm}_lr{config.lr:.0e}_seed{config.seed}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(
    device: str, r: int = 16, alpha: int = 32, dropout: float = 0.1
) -> tuple[PeftModel, CLIPProcessor]:
    model, processor = load_clip("cpu")
    model.requires_grad_(False)
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS
    )
    return get_peft_model(model, config).to(device), processor


def train_step(
    model: PeftModel,
    processor: CLIPProcessor,
    batch: list[TrainPair],
    data: TrainData,
    arm: str,
    logit_scale: float,
    optimizer: torch.optim.Optimizer,
    device: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    texts = [pair.caption for pair in batch]
    if arm == "hard":
        texts += [pair.negative for pair in batch]
    tokens = tokenize(processor, texts).to(device)
    features = text_features(model, tokens["input_ids"], tokens["attention_mask"])
    size = len(batch)
    negatives = features[size:] if arm == "hard" else None
    images = data.image_features_for([pair.image_id for pair in batch]).to(device)
    loss = contrastive_loss(images, features[:size], negatives, logit_scale)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.item()


def _batches_sha256(data: TrainData, config: TrainConfig) -> str:
    order = [
        [[pair.image_id, pair.caption_id] for pair in batch]
        for epoch in range(1, config.epochs + 1)
        for batch in epoch_batches(
            data.pairs_by_image, config.seed, epoch, config.batch_size
        )
    ]
    return hashlib.sha256(json.dumps(order).encode()).hexdigest()


def _rng_state(device: str) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device == "mps":
        state["mps"] = torch.mps.get_rng_state()
    if device.startswith("cuda"):
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _write_log(path: Path, log: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2) + "\n")


def _checkpoint_steps(steps: int) -> dict[int, float]:
    return {steps // 2: 0.5, steps: 1.0} if steps > 1 else {steps: 1.0}


def run_training(
    config: TrainConfig,
    train_data: TrainData,
    val_data: ValidationSet,
    run_dir: Path,
    log_path: Path,
    resume: bool = False,
    on_checkpoint: Callable[[], None] | None = None,
) -> dict[str, Any]:
    set_seed(config.seed)
    model, processor = build_model(
        config.device, config.lora_r, config.lora_alpha, config.lora_dropout
    )
    if config.gradient_checkpointing:
        model.get_base_model().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    hook = on_checkpoint or (lambda: None)
    logit_scale = model.get_base_model().logit_scale.exp().item()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=config.lr, weight_decay=config.weight_decay
    )
    warmup = max(config.warmup_steps, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    state_path = run_dir / "state.pt"
    start_epoch, global_step = 0, 0
    log: dict[str, Any] = {
        "checkpoints": [],
        "segments": [],
        "completed_epochs": 0,
    }
    if resume and state_path.is_file():
        state = torch.load(state_path, weights_only=False)
        set_peft_model_state_dict(model, state["lora"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, global_step = state["epoch"], state["global_step"]
        _restore_rng(state["rng"])
        log = json.loads(log_path.read_text())
        log["checkpoints"] = [
            c for c in log["checkpoints"] if c["epoch"] <= start_epoch
        ]
        log["completed_epochs"] = start_epoch

    log.update(
        {
            "config": asdict(config),
            "logit_scale": logit_scale,
            "trainable_params": sum(p.numel() for p in params),
            "lora_targets": LORA_TARGETS,
            "images": len(train_data.pairs_by_image),
            "batches_sha256": _batches_sha256(train_data, config),
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
            },
        }
    )
    segment = {"start_epoch": start_epoch, **git_state(REPO_ROOT)}
    log["segments"].append(segment)
    started = time.perf_counter()

    for epoch in range(start_epoch + 1, config.epochs + 1):
        batches = epoch_batches(
            train_data.pairs_by_image, config.seed, epoch, config.batch_size
        )
        log["steps_per_epoch"] = len(batches)
        points = _checkpoint_steps(len(batches))
        model.train()
        losses: list[float] = []
        window_start = time.perf_counter()
        for index, batch in enumerate(batches, start=1):
            losses.append(
                train_step(
                    model, processor, batch, train_data, config.arm,
                    logit_scale, optimizer, config.device, scheduler,
                )
            )
            global_step += 1
            if index not in points:
                continue
            label = epoch - 1 + points[index]
            val_start = time.perf_counter()
            model.eval()
            metrics = evaluate(model, processor, val_data, config.eval_batch_size)
            model.train()
            val_end = time.perf_counter()
            model.save_pretrained(run_dir / "checkpoints" / f"epoch_{label}")
            log["checkpoints"].append(
                {
                    "epoch": label,
                    "step": global_step,
                    "train_loss": sum(losses) / len(losses),
                    "val": metrics,
                    "train_seconds": round(val_start - window_start, 2),
                    "val_seconds": round(val_end - val_start, 2),
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }
            )
            losses = []
            window_start = time.perf_counter()
            _write_log(log_path, log)
            hook()
            logger.info("epoch %.1f: %s", label, log["checkpoints"][-1])

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "lora": get_peft_model_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": _rng_state(config.device),
            },
            state_path,
        )
        log["completed_epochs"] = epoch
        segment["end_epoch"] = epoch
        segment["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        _write_log(log_path, log)
        hook()
    return log


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the text-side LoRA of CLIP.")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=50)
    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument(
        "--train-pairs", type=Path, default=Path("data/negatives/train.jsonl")
    )
    parser.add_argument(
        "--val-negatives", type=Path, default=Path("data/negatives/val.jsonl")
    )
    parser.add_argument(
        "--embeddings-dir", type=Path, default=Path("embeddings/clip-vit-base-patch32")
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--logs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    on_checkpoint: Callable[[], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    args = _parse_args(argv)
    config = TrainConfig(
        arm=args.arm,
        lr=args.lr,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        device=args.device,
    )
    name = run_name(config)
    if args.limit_images is not None:
        name += f"_limit{args.limit_images}"
    train_data = load_train_data(
        args.train_pairs, args.embeddings_dir / "train.pt", args.limit_images
    )
    val_data = load_validation_set(
        args.embeddings_dir / "val.pt",
        args.annotations_dir / TRAIN_CAPTIONS,
        args.val_negatives,
        seed=0,
    )
    log = run_training(
        config,
        train_data,
        val_data,
        run_dir=args.runs_dir / name,
        log_path=args.logs_dir / f"{name}.json",
        resume=args.resume,
        on_checkpoint=on_checkpoint,
    )
    return name, log


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
