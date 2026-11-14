import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.cache_embeddings import save_set
from src.eval.final_test import (
    SugarExample,
    check_ready,
    evaluate_model,
    load_sugarcrepe,
    main,
    summarize,
)
from src.features import load_clip
from src.finetune.train import build_model


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _write_sugarcrepe(folder: Path) -> None:
    folder.mkdir(parents=True)
    files = {
        "swap_obj.json": {
            "1": {"filename": "000000000020.jpg", "caption": "A red car.",
                  "negative_caption": "A car red."},
            "0": {"filename": "000000000010.jpg", "caption": "A dog runs.",
                  "negative_caption": "A runs dog."},
        },
        "add_att.json": {
            "0": {"filename": "000000000010.jpg", "caption": "A dog.",
                  "negative_caption": "A blue dog."},
        },
    }
    for name, examples in files.items():
        (folder / name).write_text(json.dumps(examples))


def test_load_sugarcrepe_sorts_by_category_and_key(tmp_path: Path) -> None:
    _write_sugarcrepe(tmp_path / "sugarcrepe")

    examples = load_sugarcrepe(tmp_path / "sugarcrepe")

    assert examples == [
        SugarExample("add_att", 0, 10, "A dog.", "A blue dog."),
        SugarExample("swap_obj", 0, 10, "A dog runs.", "A runs dog."),
        SugarExample("swap_obj", 1, 20, "A red car.", "A car red."),
    ]


def test_check_ready_requires_committed_selection_and_clean_src(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("")
    (tmp_path / "results").mkdir()
    selection = tmp_path / "results" / "selection.json"
    selection.write_text("{}")
    out = tmp_path / "results" / "test_results.json"
    _git(tmp_path, "init")
    _git(tmp_path, "add", "src")
    _git(tmp_path, "commit", "-m", "code")

    with pytest.raises(SystemExit):
        check_ready(tmp_path, selection, out)

    _git(tmp_path, "add", "results")
    _git(tmp_path, "commit", "-m", "selection")
    check_ready(tmp_path, selection, out)

    (tmp_path / "src" / "a.py").write_text("x = 1")
    with pytest.raises(SystemExit):
        check_ready(tmp_path, selection, out)

    (tmp_path / "src" / "a.py").write_text("")
    out.write_text("{}")
    with pytest.raises(SystemExit):
        check_ready(tmp_path, selection, out)


def test_evaluate_model_scores_sugarcrepe_and_both_retrieval_directions() -> None:
    model, processor = load_clip("cpu")
    captions = ["A dog runs on the grass.", "A red car parked on a street."]
    from src.eval.validation import encode_texts

    image_features = encode_texts(model, processor, captions, batch_size=2)
    examples = [
        SugarExample("swap_obj", 0, 10, captions[0], "A grass runs on the dog."),
        SugarExample("swap_obj", 1, 20, captions[1], captions[1]),
    ]

    out = evaluate_model(
        model, processor, examples, [10, 20], image_features,
        captions, [10, 20], batch_size=2,
    )

    assert out["sugarcrepe_correct"].tolist() == [True, False]
    assert out["t2i_ranks"].tolist() == [0, 0]
    assert out["i2t_ranks"].tolist() == [0, 0]


def test_summarize_reports_means_seed_spread_and_paired_intervals() -> None:
    examples = [SugarExample("swap_obj", i, i % 4, "c", "n") for i in range(8)]
    base = {
        "sugarcrepe_correct": np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
        "t2i_ranks": np.array([0, 3, 12, 0]),
        "i2t_ranks": np.array([0, 1]),
    }
    better = {**base, "sugarcrepe_correct": np.ones(8, dtype=bool)}
    outputs = {"base": [base], "inbatch": [base] * 3, "hard": [better] * 3}

    summary = summarize(
        outputs, examples, caption_image_ids=[0, 0, 1, 1], image_ids=[0, 1],
        n_resamples=500, seed=0,
    )

    table = summary["table"]
    assert table["base"]["sugarcrepe"]["swap_obj"]["mean"] == 0.5
    assert table["hard"]["sugarcrepe"]["macro"]["mean"] == 1.0
    assert table["hard"]["sugarcrepe"]["swap_obj"]["std"] == 0.0
    assert table["base"]["t2i"]["r1"]["mean"] == 0.5
    assert table["base"]["i2t"]["r1"]["mean"] == 0.5
    delta = summary["deltas"]["hard_vs_base"]["sugarcrepe"]["swap_obj"]
    assert delta["delta"] == pytest.approx(0.5)
    assert delta["low"] <= 0.5 <= delta["high"]
    assert summary["deltas"]["hard_vs_inbatch"]["t2i"]["r1"]["delta"] == 0.0


def _write_inputs(tmp_path: Path) -> list[str]:
    _write_sugarcrepe(tmp_path / "sugarcrepe")
    model, processor = load_clip("cpu")
    from src.eval.validation import encode_texts

    feats = encode_texts(model, processor, ["A dog.", "A red car."], batch_size=2)
    save_set(tmp_path / "test.pt", [10, 20], feats)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    entries = [
        {"id": 1, "image_id": 10, "caption": "A dog runs."},
        {"id": 2, "image_id": 20, "caption": "A red car."},
        {"id": 3, "image_id": 20, "caption": "A car on a road."},
    ]
    (annotations / "captions_val2017.json").write_text(
        json.dumps({"images": [], "annotations": entries})
    )
    checkpoints = []
    for arm in ("inbatch", "hard"):
        for seed in (0, 1, 2):
            path = tmp_path / "runs" / f"{arm}_{seed}" / "checkpoints" / "epoch_0.5"
            torch.manual_seed(seed)
            build_model("cpu")[0].save_pretrained(path)
            checkpoints.append(str(path))
    selection = {
        "arms": {
            arm: {"lr": 1e-4, "epoch": 0.5, "seeds": [0, 1, 2],
                  "checkpoints": checkpoints[i * 3 : i * 3 + 3]}
            for i, arm in enumerate(("inbatch", "hard"))
        }
    }
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    return checkpoints


def test_main_writes_results_and_per_example_outputs(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    out = tmp_path / "results" / "test_results.json"

    main(
        [
            "--selection", str(tmp_path / "selection.json"),
            "--embeddings", str(tmp_path / "test.pt"),
            "--sugarcrepe-dir", str(tmp_path / "sugarcrepe"),
            "--annotations-dir", str(tmp_path / "annotations"),
            "--checkpoint-root", "/",
            "--out", str(out),
            "--device", "cpu",
            "--n-resamples", "200",
        ],
        require_ready=False,
    )

    results = json.loads(out.read_text())
    assert results["counts"] == {
        "sugarcrepe": 3, "images": 2, "captions": 3, "models": 7
    }
    assert set(results["table"]) == {"base", "inbatch", "hard"}
    assert set(results["deltas"]) == {
        "inbatch_vs_base", "hard_vs_base", "hard_vs_inbatch"
    }
    per_example = np.load(out.with_suffix(".npz"))
    assert per_example["sugarcrepe_correct"].shape == (7, 3)
    assert per_example["t2i_ranks"].shape == (7, 3)
    assert per_example["i2t_ranks"].shape == (7, 2)
    assert list(per_example["models"]) == [
        "base", "inbatch_0", "inbatch_1", "inbatch_2", "hard_0", "hard_1", "hard_2"
    ]
