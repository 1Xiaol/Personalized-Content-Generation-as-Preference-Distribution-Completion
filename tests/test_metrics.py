import numpy as np

from recdiffusion.metrics import (
    calibrate_chordal_bandwidth,
    future_mode_coverage,
    mmd2_unbiased,
    score_user,
)


def test_future_mode_coverage_assigns_one_mode_per_generation() -> None:
    targets = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    generated = np.asarray([[0.8, 0.6]])
    value = future_mode_coverage(generated, targets, tau_clust=0.9, tau_hit=0.5)
    assert value == 0.5


def test_single_target_only_excludes_mmd() -> None:
    generated = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
    target = np.asarray([[1.0, 0.0]])
    result = score_user(generated, target, target, bandwidth=1.0)
    assert result["mmd2_u"] is None
    assert result["target_best"] is not None


def test_unbiased_mmd_retains_finite_estimate() -> None:
    values = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    estimate = mmd2_unbiased(values, values, bandwidth=1.0)
    assert estimate is not None and np.isfinite(estimate)


def test_bandwidth_uses_nonzero_distances() -> None:
    items = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    value = calibrate_chordal_bandwidth(items, pair_count=100, seed=3)
    assert value > 0

