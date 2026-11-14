import json
import os
from pathlib import Path
from typing import Any

import modal

from src.data.split import GIT_COMMIT_ENV, GIT_DIRTY_ENV, REPO_ROOT, git_state

APP_NAME = "clip-compsearch-train"
VOLUME_NAME = "clip-compsearch-train"
VOLUME_ROOT = "/vol"
GPU = "A10G"
TIMEOUT_SECONDS = 6 * 60 * 60
ARMS = ("inbatch", "hard")
GRID = (2e-5, 1e-4, 5e-4)
MODEL_NAME = "openai/clip-vit-base-patch32"
PACKAGES = (
    "torch==2.12.1",
    "transformers==5.12.1",
    "peft==0.19.1",
    "numpy==2.5.0",
    "pillow==12.3.0",
    "safetensors==0.8.0",
    "tokenizers==0.22.2",
    "huggingface-hub==1.21.0",
)


DOWNLOAD_MODEL = (
    "python -c \"from transformers import CLIPModel, CLIPProcessor; "
    f"CLIPModel.from_pretrained('{MODEL_NAME}'); "
    f"CLIPProcessor.from_pretrained('{MODEL_NAME}')\""
)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(*PACKAGES)
    .env({"HF_HOME": "/root/hf-cache"})
    .run_commands(DOWNLOAD_MODEL)
    .add_local_python_source("src")
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def remote_argv(
    arm: str,
    lr: float,
    seed: int,
    epochs: int,
    limit_images: int | None = None,
    resume: bool = False,
) -> list[str]:
    logs = "results/smoke" if limit_images else "results/runs"
    argv = [
        "--arm", arm,
        "--lr", str(lr),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--device", "cuda",
        "--annotations-dir", f"{VOLUME_ROOT}/annotations",
        "--train-pairs", f"{VOLUME_ROOT}/data/negatives/train.jsonl",
        "--val-negatives", f"{VOLUME_ROOT}/data/negatives/val.jsonl",
        "--embeddings-dir", f"{VOLUME_ROOT}/embeddings/clip-vit-base-patch32",
        "--runs-dir", f"{VOLUME_ROOT}/runs",
        "--logs-dir", f"{VOLUME_ROOT}/{logs}",
    ]
    if limit_images:
        argv += ["--limit-images", str(limit_images)]
    if resume:
        argv.append("--resume")
    return argv


def sweep_jobs(
    arms: tuple[str, ...], lrs: tuple[float, ...], seed: int, epochs: int
) -> list[dict[str, Any]]:
    return [
        {"arm": arm, "lr": lr, "seed": seed, "epochs": epochs}
        for arm in arms
        for lr in lrs
    ]


def check_clean(state: dict[str, Any], allow_dirty: bool) -> None:
    if state["git_dirty"] and not allow_dirty:
        raise SystemExit(
            "src/ has uncommitted changes: commit first or pass --allow-dirty"
        )


@app.function(gpu=GPU, timeout=TIMEOUT_SECONDS, volumes={VOLUME_ROOT: volume})
def train_remote(argv: list[str], git: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    import logging

    from src.finetune.train import main

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.environ[GIT_COMMIT_ENV] = git["git_commit"]
    os.environ[GIT_DIRTY_ENV] = "1" if git["git_dirty"] else "0"
    volume.reload()
    name, log = main(argv, on_checkpoint=volume.commit)
    volume.commit()
    return name, log


def _save_log(name: str, log: dict[str, Any], smoke: bool) -> Path:
    folder = REPO_ROOT / "results" / ("smoke" if smoke else "runs")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(log, indent=2) + "\n")
    return path


@app.local_entrypoint()
def run(
    arm: str,
    lr: float,
    seed: int = 0,
    epochs: int = 3,
    limit_images: int = 0,
    resume: bool = False,
    allow_dirty: bool = False,
) -> None:
    git = git_state(REPO_ROOT)
    check_clean(git, allow_dirty)
    argv = remote_argv(arm, lr, seed, epochs, limit_images or None, resume)
    name, log = train_remote.remote(argv, git)
    print(f"saved {_save_log(name, log, smoke=bool(limit_images))}")


@app.local_entrypoint()
def sweep(
    seed: int = 0, epochs: int = 3, resume: bool = False, allow_dirty: bool = False
) -> None:
    git = git_state(REPO_ROOT)
    check_clean(git, allow_dirty)
    jobs = sweep_jobs(ARMS, GRID, seed, epochs)
    calls = [(remote_argv(**job, resume=resume), git) for job in jobs]
    for name, log in train_remote.starmap(calls):
        print(f"saved {_save_log(name, log, smoke=False)}")
