import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

R1_GUARD = 0.01
LOWER_EDGE_EXTRA = 1e-5
UPPER_EDGE_EXTRA = 1e-3
EXTENDED_EPOCHS = 5


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Apply the selection rule and the sweep rules to one arm."
    )
    parser.add_argument("--arm", choices=("inbatch", "hard"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("results/val_baseline.json")
    )
    args = parser.parse_args(argv)

    base_r1 = json.loads(args.baseline.read_text())["metrics"]["t2i_r1"]
    paths = sorted(args.logs_dir.glob("*.json"))
    runs = [json.loads(path.read_text()) for path in paths]
    runs = [
        run
        for run in runs
        if run["config"]["arm"] == args.arm and run["config"]["seed"] == args.seed
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
