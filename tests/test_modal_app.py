import pytest

from src.finetune.modal_app import VOLUME_ROOT, check_clean, remote_argv, sweep_jobs
from src.finetune.train import _parse_args


def test_remote_argv_points_every_path_at_the_volume() -> None:
    argv = remote_argv(arm="hard", lr=1e-4, seed=0, epochs=3)
    args = _parse_args(argv)

    assert args.device == "cuda"
    assert (args.arm, args.lr, args.seed, args.epochs) == ("hard", 1e-4, 0, 3)
    for path in (
        args.annotations_dir,
        args.train_pairs,
        args.val_negatives,
        args.embeddings_dir,
        args.runs_dir,
        args.logs_dir,
    ):
        assert str(path).startswith(VOLUME_ROOT)
    assert args.limit_images is None
    assert args.resume is False


def test_remote_argv_passes_resume_and_limit() -> None:
    argv = remote_argv(
        arm="inbatch", lr=2e-5, seed=1, epochs=5, limit_images=512, resume=True
    )
    args = _parse_args(argv)

    assert args.resume is True
    assert args.limit_images == 512
    assert args.epochs == 5


def test_sweep_jobs_cover_both_arms_and_the_grid() -> None:
    grid = (2e-5, 1e-4, 5e-4)
    jobs = sweep_jobs(arms=("inbatch", "hard"), lrs=grid, seed=0, epochs=3)

    assert len(jobs) == 6
    assert {(job["arm"], job["lr"]) for job in jobs} == {
        (arm, lr) for arm in ("inbatch", "hard") for lr in (2e-5, 1e-4, 5e-4)
    }


def test_check_clean_refuses_a_dirty_tree() -> None:
    with pytest.raises(SystemExit):
        check_clean({"git_commit": "a" * 40, "git_dirty": True}, allow_dirty=False)

    check_clean({"git_commit": "a" * 40, "git_dirty": True}, allow_dirty=True)
    check_clean({"git_commit": "a" * 40, "git_dirty": False}, allow_dirty=False)
