"""Unified user-level distribution metrics for both evaluation datasets."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

METRICS = (
    "history_nearest",
    "target_best",
    "coverage_rad",
    "future_mode_coverage",
    "energy",
    "mmd2_u",
)
METRIC_DIRECTIONS = {
    "history_nearest": "diagnostic",
    "target_best": "higher",
    "coverage_rad": "lower",
    "future_mode_coverage": "higher",
    "energy": "lower",
    "mmd2_u": "lower",
}


def normalized(values: np.ndarray, name: str = "embeddings") -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"invalid {name}")
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if (norm < 1e-12).any():
        raise ValueError(f"zero vector in {name}")
    return values / norm


def calibrate_chordal_bandwidth(
    train_item_embeddings: np.ndarray,
    *,
    pair_count: int = 500_000,
    seed: int = 2026,
) -> float:
    """Median nonzero chord distance from sampled distinct Train item pairs."""
    values = normalized(train_item_embeddings, "train item embeddings")
    if len(values) < 2 or pair_count < 1:
        raise ValueError("bandwidth calibration needs at least two items and one pair")
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(values), size=pair_count)
    right = rng.integers(0, len(values) - 1, size=pair_count)
    right += right >= left
    distance = np.linalg.norm(values[left] - values[right], axis=1)
    distance = distance[distance > 0]
    if not len(distance):
        raise ValueError("all sampled Train pairs have zero chord distance")
    return float(np.median(distance))


def target_mode_labels(targets: np.ndarray, *, tau_clust: float) -> np.ndarray:
    """Average-linkage target-only clustering under cosine distance."""
    targets = normalized(targets, "targets")
    if not -1 <= tau_clust <= 1:
        raise ValueError("tau_clust must be a cosine similarity")
    if len(targets) == 1:
        return np.ones(1, dtype=np.int32)
    distance = np.clip(1 - np.clip(targets @ targets.T, -1, 1), 0, 2)
    np.fill_diagonal(distance, 0)
    return fcluster(
        linkage(squareform(distance, checks=False), method="average"),
        t=1 - tau_clust,
        criterion="distance",
    ).astype(np.int32)


def future_mode_coverage(
    generated: np.ndarray,
    targets: np.ndarray,
    *,
    tau_clust: float,
    tau_hit: float,
) -> float:
    """Assign each supported generation to only its globally nearest target mode."""
    generated = normalized(generated, "generated")
    targets = normalized(targets, "targets")
    labels = target_mode_labels(targets, tau_clust=tau_clust)
    similarity = np.clip(generated @ targets.T, -1, 1)
    nearest = similarity.argmax(axis=1)
    supported = similarity.max(axis=1) >= tau_hit
    hit_modes = set(labels[nearest[supported]].tolist())
    return float(len(hit_modes) / len(set(labels.tolist())))


def energy_score(generated: np.ndarray, targets: np.ndarray) -> float:
    generated = normalized(generated, "generated")
    targets = normalized(targets, "targets")
    if len(generated) < 2:
        raise ValueError("Energy needs at least two generated samples")
    cross = np.linalg.norm(generated[:, None] - targets[None, :], axis=-1)
    within = np.linalg.norm(generated[:, None] - generated[None, :], axis=-1)
    samples = len(generated)
    return float(cross.mean() - within.sum() / (2 * samples * (samples - 1)))


def mmd2_unbiased(generated: np.ndarray, targets: np.ndarray, *, bandwidth: float) -> float | None:
    """Unbiased chordal Gaussian-RBF MMD; negative estimates are retained."""
    generated = normalized(generated, "generated")
    targets = normalized(targets, "targets")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    samples, target_count = len(generated), len(targets)
    if samples < 2:
        raise ValueError("MMD needs at least two generated samples")
    if target_count < 2:
        return None
    generated_distance = np.linalg.norm(generated[:, None] - generated[None, :], axis=-1)
    target_distance = np.linalg.norm(targets[:, None] - targets[None, :], axis=-1)
    cross_distance = np.linalg.norm(generated[:, None] - targets[None, :], axis=-1)
    kgg = np.exp(-(generated_distance**2) / (2 * bandwidth**2))
    ktt = np.exp(-(target_distance**2) / (2 * bandwidth**2))
    kgt = np.exp(-(cross_distance**2) / (2 * bandwidth**2))
    return float(
        (kgg.sum() - np.trace(kgg)) / (samples * (samples - 1))
        + (ktt.sum() - np.trace(ktt)) / (target_count * (target_count - 1))
        - 2 * kgt.mean()
    )


def score_user(
    generated: np.ndarray,
    targets: np.ndarray,
    history: np.ndarray,
    *,
    bandwidth: float,
    tau_clust: float = 0.6900425553321838,
    tau_hit: float = 0.6900425553321838,
) -> dict[str, float | None]:
    generated = normalized(generated, "generated")
    targets = normalized(targets, "targets")
    history = normalized(history, "history")
    if generated.shape[1] != targets.shape[1] or targets.shape[1] != history.shape[1]:
        raise ValueError("embedding dimensions differ")
    similarity = np.clip(generated @ targets.T, -1, 1)
    return {
        "history_nearest": float(np.max(np.clip(generated @ history.T, -1, 1), axis=1).mean()),
        "target_best": float(similarity.max(axis=0).mean()),
        "coverage_rad": float(np.arccos(similarity).min(axis=0).mean()),
        "future_mode_coverage": future_mode_coverage(
            generated, targets, tau_clust=tau_clust, tau_hit=tau_hit
        ),
        "energy": energy_score(generated, targets),
        "mmd2_u": mmd2_unbiased(generated, targets, bandwidth=bandwidth),
    }


def macro_average(records: Sequence[Mapping[str, float | None]]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for metric in METRICS:
        values = [float(record[metric]) for record in records if record.get(metric) is not None]
        result[metric] = float(np.mean(values)) if values else None
        result[f"{metric}_n_users"] = len(values)
    return result


def paired_user_bootstrap(
    left: Sequence[Mapping[str, float | None]],
    right: Sequence[Mapping[str, float | None]],
    *,
    resamples: int = 2000,
    seed: int = 2026,
) -> dict[str, dict[str, float | int | None]]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires aligned nonempty users")
    rng = np.random.default_rng(seed)
    output = {}
    for metric in METRICS:
        differences = np.asarray(
            [
                float(lhs[metric]) - float(rhs[metric])
                for lhs, rhs in zip(left, right, strict=True)
                if lhs.get(metric) is not None and rhs.get(metric) is not None
            ],
            dtype=np.float64,
        )
        if not len(differences):
            output[metric] = {"mean": None, "ci_low": None, "ci_high": None, "n_users": 0}
            continue
        draws = differences[
            rng.integers(0, len(differences), size=(resamples, len(differences)))
        ].mean(axis=1)
        output[metric] = {
            "mean": float(differences.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "n_users": int(len(differences)),
        }
    return output

