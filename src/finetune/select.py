import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.data.split import REPO_ROOT, git_state

logger = logging.getLogger(__name__)

ARMS = ("inbatch", "hard")
R1_GUARD = 0.01
LOWER_EDGE_EXTRA = 1e-5
UPPER_EDGE_EXTRA = 1e-3
EXTENDED_EPOCHS = 5
REQUIRED_SEEDS = (0, 1, 2)
METRICS = ("comp_accuracy", "t2i_r1", "t2i_r5", "t2i_r10")
RULE = (
    "Keep checkpoints whose validation R@1 is at most 1 point below base CLIP; "
    "choose the highest compositional accuracy; else the highest R@1."
)


def _candidates(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "lr": run["config"]["lr"],
            "epochs": run["config"]["epochs"],
            "epoch": checkpoint["epoch"],
            "val": checkpoint["val"],
        }
        for run in runs
        for checkpoint in run["checkpoints"]
    ]


def select_checkpoint(
    runs: list[dict[str, Any]], base_r1: float, guard: float = R1_GUARD
) -> dict[str, Any]:
    candidates = _candidates(runs)
    passing = [c for c in candidates if c["val"]["t2i_r1"] >= base_r1 - guard]
    if passing:
        best = max(
            passing, key=lambda c: (c["val"]["comp_accuracy"], -c["epoch"], -c["lr"])
        )
        return {**best, "guard_passed": True}
    best = max(candidates, key=lambda c: (c["val"]["t2i_r1"], -c["epoch"], -c["lr"]))
    return {**best, "guard_passed": False}


def sweep_actions(
    runs: list[dict[str, Any]], base_r1: float, guard: float = R1_GUARD
) -> list[dict[str, Any]]:
    choice = select_checkpoint(runs, base_r1, guard)
    tried = {run["config"]["lr"] for run in runs}
    actions = []
    if choice["lr"] == min(tried) and LOWER_EDGE_EXTRA not in tried:
        actions.append({"action": "run_lr", "lr": LOWER_EDGE_EXTRA})
    if choice["lr"] == max(tried) and UPPER_EDGE_EXTRA not in tried:
        actions.append({"action": "run_lr", "lr": UPPER_EDGE_EXTRA})
    is_last = choice["epoch"] == choice["epochs"]
    if is_last and choice["epochs"] < EXTENDED_EPOCHS:
        actions.append(
            {"action": "extend_epochs", "lr": choice["lr"], "epochs": EXTENDED_EPOCHS}
        )
    return actions


def average_checkpoints(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_epoch: dict[float, list[dict[str, float]]] = defaultdict(list)
    for run in runs:
        for checkpoint in run["checkpoints"]:
            by_epoch[checkpoint["epoch"]].append(checkpoint["val"])
    seeds = sorted(run["config"]["seed"] for run in runs)
    return [
        {
            "epoch": epoch,
            "seeds": seeds,
            "val": {
                metric: mean(val[metric] for val in vals)
                for metric in METRICS
                if all(metric in val for val in vals)
            },
        }
        for epoch, vals in sorted(by_epoch.items())
        if len(vals) == len(runs)
    ]


def select_final(
    runs: list[dict[str, Any]], base_r1: float, guard: float = R1_GUARD
) -> dict[str, Any]:
    sweep = [run for run in runs if run["config"]["seed"] == REQUIRED_SEEDS[0]]
    lr = select_checkpoint(sweep, base_r1, guard)["lr"]
    chosen = [run for run in runs if run["config"]["lr"] == lr]
    seeds = sorted(run["config"]["seed"] for run in chosen)
    if seeds != list(REQUIRED_SEEDS):
        raise ValueError(f"lr {lr} has seeds {seeds}, need {list(REQUIRED_SEEDS)}")
    epochs = {run["config"]["epochs"] for run in chosen}
    if len(epochs) != 1:
        raise ValueError(f"lr {lr} runs have different epoch counts: {epochs}")

    averaged = average_checkpoints(chosen)
    config = {"lr": lr, "epochs": epochs.pop()}
    choice = select_checkpoint(
        [{"config": config, "checkpoints": averaged}], base_r1, guard
    )
    return {**choice, "seeds": seeds}


def _run_name(arm: str, lr: float, seed: int) -> str:
    return f"{arm}_lr{lr:.0e}_seed{seed}"


def _load_runs(logs_dir: Path, arm: str) -> list[dict[str, Any]]:
    runs = [json.loads(path.read_text()) for path in sorted(logs_dir.glob("*.json"))]
    return [run for run in runs if run["config"]["arm"] == arm]


def _final_selection(logs_dir: Path, base_r1: float) -> dict[str, Any]:
    arms = {}
    for arm in ARMS:
        choice = select_final(_load_runs(logs_dir, arm), base_r1)
        choice["checkpoints"] = [
            f"runs/{_run_name(arm, choice['lr'], seed)}/checkpoints/"
            f"epoch_{choice['epoch']}"
            for seed in choice["seeds"]
        ]
        arms[arm] = choice
    return {
        "rule": RULE,
        "base_r1": base_r1,
        "guard": R1_GUARD,
        "arms": arms,
        **git_state(REPO_ROOT),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Apply the selection rule and the sweep rules."
    )
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--logs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("results/val_baseline.json")
    )
    parser.add_argument("--out", type=Path, default=Path("results/selection.json"))
    args = parser.parse_args(argv)

    base_r1 = json.loads(args.baseline.read_text())["metrics"]["t2i_r1"]
    if args.final:
        selection = _final_selection(args.logs_dir, base_r1)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection["arms"], indent=2))
        return
    if args.arm is None:
        parser.error("--arm is required without --final")

    runs = [
        run
        for run in _load_runs(args.logs_dir, args.arm)
        if run["config"]["seed"] == args.seed
    ]
    if not runs:
        raise SystemExit(f"No runs for arm {args.arm} with seed {args.seed}")
    result = {
        "arm": args.arm,
        "base_r1": base_r1,
        "choice": select_checkpoint(runs, base_r1),
        "actions": sweep_actions(runs, base_r1),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
