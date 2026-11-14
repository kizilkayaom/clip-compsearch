from typing import Any

import numpy as np

CHUNK = 500


def cluster_totals(
    groups: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    universe = np.unique(np.concatenate([clusters for _, clusters in groups.values()]))
    sums = np.zeros((len(groups), len(universe)))
    counts = np.zeros((len(groups), len(universe)), dtype=np.int64)
    for row, (values, clusters) in enumerate(groups.values()):
        index = np.searchsorted(universe, clusters)
        np.add.at(sums[row], index, values)
        np.add.at(counts[row], index, 1)
    return sums, counts, universe


def _interval(samples: np.ndarray, ci: float) -> tuple[float, float]:
    tail = (1 - ci) / 2 * 100
    low, high = np.nanpercentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_groups(
    sums: np.ndarray,
    counts: np.ndarray,
    names: list[str],
    n_resamples: int,
    seed: int,
    ci: float = 0.95,
    macro: bool | None = None,
) -> dict[str, dict[str, Any]]:
    observed = sums.sum(axis=1) / counts.sum(axis=1)
    rng = np.random.default_rng(seed)
    n_clusters = sums.shape[1]
    uniform = np.full(n_clusters, 1.0 / n_clusters)
    chunks = []
    for start in range(0, n_resamples, CHUNK):
        size = min(CHUNK, n_resamples - start)
        weights = rng.multinomial(n_clusters, uniform, size=size)
        with np.errstate(invalid="ignore", divide="ignore"):
            chunks.append((weights @ sums.T) / (weights @ counts.T))
    samples = np.concatenate(chunks)

    result = {}
    for column, name in enumerate(names):
        low, high = _interval(samples[:, column], ci)
        result[name] = {"delta": float(observed[column]), "low": low, "high": high}
    add_macro = len(names) > 1 if macro is None else macro
    if add_macro:
        low, high = _interval(samples.mean(axis=1), ci)
        result["macro"] = {"delta": float(observed.mean()), "low": low, "high": high}
    return result
