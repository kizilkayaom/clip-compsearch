import json
from pathlib import Path

import pytest

from src.finetune.select import (
    average_checkpoints,
    main,
    select_checkpoint,
    select_final,
    sweep_actions,
)

BASE_R1 = 0.30


def _run(lr: float, epochs: int, points: list[tuple[float, float, float]]) -> dict:
    return {
        "config": {"arm": "hard", "lr": lr, "seed": 0, "epochs": epochs},
        "checkpoints": [
            {"epoch": epoch, "val": {"t2i_r1": r1, "comp_accuracy": comp}}
            for epoch, r1, comp in points
        ],
    }


def test_select_checkpoint_takes_best_composition_within_the_r1_guard() -> None:
    runs = [
        _run(1e-4, 3, [(0.5, 0.300, 0.75), (1.0, 0.295, 0.80), (1.5, 0.280, 0.90)]),
        _run(5e-4, 3, [(0.5, 0.291, 0.79)]),
    ]

    choice = select_checkpoint(runs, BASE_R1, guard=0.01)

    assert (choice["lr"], choice["epoch"]) == (1e-4, 1.0)
    assert choice["guard_passed"] is True


def test_select_checkpoint_breaks_ties_by_earliest_checkpoint() -> None:
    runs = [_run(1e-4, 3, [(0.5, 0.30, 0.80), (1.0, 0.30, 0.80)])]

    assert select_checkpoint(runs, BASE_R1, guard=0.01)["epoch"] == 0.5


def test_select_checkpoint_falls_back_to_best_r1_when_guard_fails() -> None:
    runs = [_run(1e-4, 3, [(0.5, 0.25, 0.90), (1.0, 0.27, 0.80)])]

    choice = select_checkpoint(runs, BASE_R1, guard=0.01)

    assert choice["epoch"] == 1.0
    assert choice["guard_passed"] is False


@pytest.mark.parametrize(
    ("best_lr", "expected_lr"),
    [(2e-5, 1e-5), (5e-4, 1e-3), (1e-4, None)],
    ids=["lower_edge", "upper_edge", "inside"],
)
def test_sweep_actions_extend_the_grid_at_an_edge(
    best_lr: float, expected_lr: float | None
) -> None:
    runs = [
        _run(lr, 3, [(1.0, 0.30, 0.90 if lr == best_lr else 0.70)])
        for lr in (2e-5, 1e-4, 5e-4)
    ]

    actions = sweep_actions(runs, BASE_R1)

    extra = [a["lr"] for a in actions if a["action"] == "run_lr"]
    assert extra == ([] if expected_lr is None else [expected_lr])


def test_sweep_actions_do_not_extend_the_grid_twice() -> None:
    runs = [
        _run(lr, 3, [(1.0, 0.30, 0.90 if lr == 1e-5 else 0.70)])
        for lr in (1e-5, 2e-5, 1e-4, 5e-4)
    ]

    assert [a for a in sweep_actions(runs, BASE_R1) if a["action"] == "run_lr"] == []


def test_sweep_actions_extend_epochs_when_last_checkpoint_wins() -> None:
    runs = [
        _run(1e-4, 3, [(2.5, 0.30, 0.80), (3.0, 0.30, 0.85)]),
        _run(2e-5, 3, [(3.0, 0.30, 0.70)]),
        _run(5e-4, 3, [(3.0, 0.30, 0.70)]),
    ]

    actions = sweep_actions(runs, BASE_R1)

    assert {"action": "extend_epochs", "lr": 1e-4, "epochs": 5} in actions


def test_sweep_actions_do_not_extend_a_five_epoch_run() -> None:
    runs = [
        _run(1e-4, 5, [(4.5, 0.30, 0.80), (5.0, 0.30, 0.85)]),
        _run(2e-5, 3, [(3.0, 0.30, 0.70)]),
        _run(5e-4, 3, [(3.0, 0.30, 0.70)]),
    ]

    assert sweep_actions(runs, BASE_R1) == []


def _seed_run(
    arm: str, lr: float, seed: int, points: list[tuple[float, float, float]]
) -> dict:
    run = _run(lr, 3, points)
    run["config"].update({"arm": arm, "seed": seed})
    return run


def test_average_checkpoints_over_seeds() -> None:
    runs = [
        _seed_run("hard", 1e-4, seed, [(0.5, 0.30 + d, 0.80 + d), (1.0, 0.31, 0.90)])
        for seed, d in ((0, 0.00), (1, 0.03), (2, 0.06))
    ]

    averaged = average_checkpoints(runs)

    assert [c["epoch"] for c in averaged] == [0.5, 1.0]
    assert averaged[0]["val"]["comp_accuracy"] == pytest.approx(0.83)
    assert averaged[0]["val"]["t2i_r1"] == pytest.approx(0.33)
    assert averaged[0]["seeds"] == [0, 1, 2]


def test_select_final_uses_the_sweep_rate_and_seed_averages() -> None:
    sweep = [
        _seed_run("hard", 1e-4, 0, [(0.5, 0.30, 0.95), (1.0, 0.30, 0.90)]),
        _seed_run("hard", 5e-4, 0, [(0.5, 0.30, 0.70), (1.0, 0.30, 0.70)]),
    ]
    seeds = [
        _seed_run("hard", 1e-4, seed, [(0.5, 0.30, 0.80), (1.0, 0.30, 0.92)])
        for seed in (1, 2)
    ]

    final = select_final(sweep + seeds, BASE_R1)

    assert final["lr"] == 1e-4
    assert final["epoch"] == 1.0
    assert final["val"]["comp_accuracy"] == pytest.approx((0.90 + 0.92 + 0.92) / 3)
    assert final["seeds"] == [0, 1, 2]


def test_select_final_requires_three_seeds() -> None:
    runs = [
        _seed_run("hard", 1e-4, seed, [(0.5, 0.30, 0.80)]) for seed in (0, 1)
    ]

    with pytest.raises(ValueError):
        select_final(runs, BASE_R1)


def test_main_final_writes_selection_for_both_arms(tmp_path: Path) -> None:
    logs = tmp_path / "runs"
    logs.mkdir()
    for arm in ("hard", "inbatch"):
        for seed in (0, 1, 2):
            run = _seed_run(arm, 1e-4, seed, [(0.5, 0.30, 0.80), (1.0, 0.30, 0.85)])
            name = f"{arm}_lr1e-04_seed{seed}"
            (logs / f"{name}.json").write_text(json.dumps(run))
    baseline = tmp_path / "val_baseline.json"
    baseline.write_text(json.dumps({"metrics": {"t2i_r1": BASE_R1}}))
    out = tmp_path / "selection.json"

    main(
        [
            "--final",
            "--logs-dir", str(logs),
            "--baseline", str(baseline),
            "--out", str(out),
        ]
    )

    selection = json.loads(out.read_text())
    assert set(selection["arms"]) == {"hard", "inbatch"}
    hard = selection["arms"]["hard"]
    assert (hard["lr"], hard["epoch"], hard["seeds"]) == (1e-4, 1.0, [0, 1, 2])
    assert hard["checkpoints"] == [
        f"runs/hard_lr1e-04_seed{seed}/checkpoints/epoch_1.0" for seed in (0, 1, 2)
    ]
    assert len(selection["git_commit"]) == 40
