import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import CLIPModel

from src.data.split import REPO_ROOT
from src.eval.validation import CompExample, ValidationSet
from src.features import load_clip, text_features, tokenize
from src.finetune.data import TrainData, TrainPair
from src.finetune.train import (
    TrainConfig,
    build_model,
    run_name,
    run_training,
    train_step,
)

CAPTIONS = [
    "A dog runs on the grass.",
    "A red car parked on a street.",
    "A plate of pasta on a table.",
    "A man rides a horse on a beach.",
]
NEGATIVES = [
    "A grass runs on the dog.",
    "A red street parked on a car.",
    "A table of pasta on a plate.",
    "A horse rides a man on a beach.",
]


def _train_data() -> TrainData:
    pairs = {
        image_id: (TrainPair(image_id, image_id, CAPTIONS[i], NEGATIVES[i]),)
        for i, image_id in enumerate((11, 12, 13, 14))
    }
    features = torch.randn(4, 512, generator=torch.Generator().manual_seed(0))
    return TrainData(pairs, features, {11: 0, 12: 1, 13: 2, 14: 3})


def _val_data() -> ValidationSet:
    return ValidationSet(
        image_ids=(21, 22),
        image_features=torch.randn(2, 512, generator=torch.Generator().manual_seed(1)),
        captions=(CAPTIONS[0], CAPTIONS[1]),
        caption_image_ids=(21, 22),
        comp_examples=(CompExample(21, 1, CAPTIONS[0], NEGATIVES[0]),),
    )


def _config(**overrides: object) -> TrainConfig:
    values = {
        "arm": "hard",
        "lr": 1e-3,
        "seed": 0,
        "epochs": 2,
        "batch_size": 2,
        "warmup_steps": 1,
        "device": "cpu",
    }
    values.update(overrides)
    return TrainConfig(**values)


def _dropout_flags(model: torch.nn.Module) -> set[bool]:
    return {
        module.training
        for name, module in model.named_modules()
        if name.endswith("lora_dropout.default")
    }


def test_build_model_trains_only_text_lora_weights() -> None:
    model, _ = build_model("cpu")

    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}

    assert sum(trainable.values()) == 393_216
    assert all("text_model" in name and "lora_" in name for name in trainable)


def test_build_model_matches_base_clip_before_any_update() -> None:
    model, processor = build_model("cpu")
    base, _ = load_clip("cpu")
    tokens = tokenize(processor, CAPTIONS)

    model.eval()
    with torch.no_grad():
        tuned = text_features(model, tokens["input_ids"], tokens["attention_mask"])
        expected = text_features(base, tokens["input_ids"], tokens["attention_mask"])

    assert torch.allclose(tuned, expected, atol=1e-6)


def test_lora_dropout_follows_train_and_eval_mode() -> None:
    model, _ = build_model("cpu")

    model.train()
    assert _dropout_flags(model) == {True}
    model.eval()
    assert _dropout_flags(model) == {False}


def test_train_step_overfits_one_small_batch() -> None:
    torch.manual_seed(0)
    model, processor = build_model("cpu")
    model.train()
    data = _train_data()
    batch = [pairs[0] for pairs in data.pairs_by_image.values()]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    losses = [
        train_step(model, processor, batch, data, "hard", 100.0, optimizer, "cpu")
        for _ in range(40)
    ]

    assert losses[-1] < 0.2 * losses[0]


def _run(tmp_path: Path, name: str, config: object, resume: bool = False) -> dict:
    return run_training(
        config,
        _train_data(),
        _val_data(),
        run_dir=tmp_path / "runs" / name,
        log_path=tmp_path / "logs" / f"{name}.json",
        resume=resume,
    )


def test_run_training_logs_two_checkpoints_per_epoch(tmp_path: Path) -> None:
    log = _run(tmp_path, "a", _config())

    assert [c["epoch"] for c in log["checkpoints"]] == [0.5, 1.0, 1.5, 2.0]
    assert log["completed_epochs"] == 2
    assert log["logit_scale"] == pytest.approx(100.0, abs=1e-3)
    assert set(log["checkpoints"][0]["val"]) >= {"comp_accuracy", "t2i_r1"}
    assert (tmp_path / "runs" / "a" / "checkpoints" / "epoch_1.5").is_dir()
    saved = json.loads((tmp_path / "logs" / "a.json").read_text())
    assert saved["checkpoints"] == log["checkpoints"]


def test_run_training_is_repeatable_with_the_same_seed(tmp_path: Path) -> None:
    first = _run(tmp_path, "a", _config())
    second = _run(tmp_path, "b", _config())

    first_losses = [c["train_loss"] for c in first["checkpoints"]]
    assert first_losses == [c["train_loss"] for c in second["checkpoints"]]


def test_resume_continues_exactly_where_the_run_stopped(tmp_path: Path) -> None:
    straight = _run(tmp_path, "straight", _config(epochs=3))
    _run(tmp_path, "split", _config(epochs=1))
    resumed = _run(tmp_path, "split", _config(epochs=3), resume=True)

    straight_losses = [c["train_loss"] for c in straight["checkpoints"]]
    resumed_losses = [c["train_loss"] for c in resumed["checkpoints"]]
    assert resumed_losses == pytest.approx(straight_losses, rel=1e-5)
    assert resumed["completed_epochs"] == 3


def test_arms_see_the_same_batches_and_differ_in_loss(tmp_path: Path) -> None:
    hard = _run(tmp_path, "hard", _config(arm="hard", epochs=1))
    inbatch = _run(tmp_path, "inbatch", _config(arm="inbatch", epochs=1))

    hard_loss = hard["checkpoints"][0]["train_loss"]
    assert hard["batches_sha256"] == inbatch["batches_sha256"]
    assert hard_loss != inbatch["checkpoints"][0]["train_loss"]


def test_run_name_is_stable() -> None:
    assert run_name(_config(arm="hard", lr=1e-4, seed=2)) == "hard_lr1e-04_seed2"


def test_train_config_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError):
        _config(arm="base")


def test_build_model_returns_clip_text_features(tmp_path: Path) -> None:
    model, _ = build_model("cpu")

    assert isinstance(model.get_base_model(), CLIPModel)


def test_training_imports_do_not_load_spacy() -> None:
    code = (
        "import sys, src.finetune.train; "
        "print(any(m.split('.')[0] in ('spacy', 'sentence_transformers') "
        "for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"


def test_run_training_calls_hook_after_each_checkpoint(tmp_path: Path) -> None:
    calls: list[float] = []

    log = run_training(
        _config(epochs=1),
        _train_data(),
        _val_data(),
        run_dir=tmp_path / "runs" / "hook",
        log_path=tmp_path / "logs" / "hook.json",
        on_checkpoint=lambda: calls.append(1.0),
    )

    assert len(calls) == len(log["checkpoints"]) + log["completed_epochs"]


def test_gradient_checkpointing_gives_the_same_losses(tmp_path: Path) -> None:
    plain = _run(tmp_path, "plain", _config(epochs=1))
    checkpointed = _run(
        tmp_path, "ckpt", _config(epochs=1, gradient_checkpointing=True)
    )

    plain_losses = [c["train_loss"] for c in plain["checkpoints"]]
    ckpt_losses = [c["train_loss"] for c in checkpointed["checkpoints"]]
    assert ckpt_losses == pytest.approx(plain_losses, rel=1e-5)


def test_checkpoints_record_training_and_validation_time(tmp_path: Path) -> None:
    log = _run(tmp_path, "timed", _config(epochs=1))

    for checkpoint in log["checkpoints"]:
        assert checkpoint["train_seconds"] > 0
        assert checkpoint["val_seconds"] > 0


def test_resume_drops_checkpoints_logged_after_the_saved_state(tmp_path: Path) -> None:
    straight = _run(tmp_path, "straight", _config(epochs=2))
    _run(tmp_path, "split", _config(epochs=1))
    log_path = tmp_path / "logs" / "split.json"
    partial = json.loads(log_path.read_text())
    partial["checkpoints"].append({**partial["checkpoints"][0], "epoch": 1.5})
    log_path.write_text(json.dumps(partial))

    resumed = _run(tmp_path, "split", _config(epochs=2), resume=True)

    assert [c["epoch"] for c in resumed["checkpoints"]] == [0.5, 1.0, 1.5, 2.0]
    straight_losses = [c["train_loss"] for c in straight["checkpoints"]]
    resumed_losses = [c["train_loss"] for c in resumed["checkpoints"]]
    assert resumed_losses == pytest.approx(straight_losses, rel=1e-5)


def test_resume_without_saved_state_starts_from_scratch(tmp_path: Path) -> None:
    straight = _run(tmp_path, "straight", _config(epochs=1))
    fresh = _run(tmp_path, "fresh", _config(epochs=1), resume=True)

    straight_losses = [c["train_loss"] for c in straight["checkpoints"]]
    assert [c["train_loss"] for c in fresh["checkpoints"]] == straight_losses


def test_resume_of_a_finished_run_leaves_the_log_unchanged(tmp_path: Path) -> None:
    _run(tmp_path, "done", _config(epochs=1))
    before = json.loads((tmp_path / "logs" / "done.json").read_text())

    resumed = _run(tmp_path, "done", _config(epochs=1), resume=True)

    assert resumed == before
