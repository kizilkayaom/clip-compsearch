import pytest

from src.finetune.select import select_checkpoint, sweep_actions

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
