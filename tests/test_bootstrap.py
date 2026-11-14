import numpy as np
import pytest

from src.eval.bootstrap import bootstrap_groups, cluster_totals

N_RESAMPLES = 2000


def _interval(values: np.ndarray, clusters: np.ndarray, seed: int = 0) -> tuple:
    sums, counts, _ = cluster_totals({"all": (values, clusters)})
    result = bootstrap_groups(sums, counts, ["all"], N_RESAMPLES, seed)
    return result["all"]["delta"], result["all"]["low"], result["all"]["high"]


def test_identical_models_give_an_interval_at_zero() -> None:
    diffs = np.zeros(300)
    clusters = np.repeat(np.arange(100), 3)

    delta, low, high = _interval(diffs, clusters)

    assert delta == low == high == 0.0


def test_interval_contains_a_known_difference() -> None:
    diffs = np.zeros(1000)
    diffs[::10] = 1.0
    clusters = np.arange(1000)

    delta, low, high = _interval(diffs, clusters)

    assert delta == pytest.approx(0.1)
    assert low < 0.1 < high


def test_clustering_widens_the_interval_for_copied_examples() -> None:
    rng = np.random.default_rng(0)
    base = rng.choice([0.0, 1.0], size=200)
    copied = np.repeat(base, 5)

    _, plain_low, plain_high = _interval(copied, np.arange(len(copied)))
    _, low, high = _interval(copied, np.repeat(np.arange(200), 5))

    assert (high - low) > 1.5 * (plain_high - plain_low)


def test_bootstrap_is_repeatable_with_the_seed() -> None:
    rng = np.random.default_rng(1)
    diffs = rng.normal(size=400)
    clusters = rng.integers(0, 80, size=400)

    assert _interval(diffs, clusters, seed=3) == _interval(diffs, clusters, seed=3)
    assert _interval(diffs, clusters, seed=3) != _interval(diffs, clusters, seed=4)


def test_groups_share_one_resample_of_the_clusters() -> None:
    clusters = np.arange(500)
    values = np.random.default_rng(2).normal(size=500)
    sums, counts, universe = cluster_totals(
        {"a": (values, clusters), "b": (values, clusters)}
    )

    result = bootstrap_groups(sums, counts, ["a", "b"], N_RESAMPLES, seed=0)

    assert len(universe) == 500
    assert result["a"] == result["b"]
    assert result["macro"]["delta"] == pytest.approx(result["a"]["delta"])
    assert result["macro"]["low"] == pytest.approx(result["a"]["low"])


def test_cluster_totals_align_groups_on_the_union_of_clusters() -> None:
    sums, counts, universe = cluster_totals(
        {
            "a": (np.array([1.0, 2.0, 3.0]), np.array([10, 10, 30])),
            "b": (np.array([5.0]), np.array([20])),
        }
    )

    assert universe.tolist() == [10, 20, 30]
    assert sums.tolist() == [[3.0, 0.0, 3.0], [0.0, 5.0, 0.0]]
    assert counts.tolist() == [[2, 0, 1], [0, 1, 0]]
