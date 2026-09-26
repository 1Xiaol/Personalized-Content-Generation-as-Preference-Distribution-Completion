"""Explicit, pickle-free array contracts for training and evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite values")


def _require_unit_rows(name: str, values: np.ndarray, mask: np.ndarray | None = None) -> None:
    _require_finite(name, values)
    norms = np.linalg.norm(values, axis=-1)
    selected = norms if mask is None else norms[~mask]
    if mask is not None and not len(selected):
        return
    if not len(selected) or not np.allclose(selected, 1.0, atol=2e-3, rtol=0):
        raise ValueError(f"{name} must contain L2-normalized non-padding rows")


def _validate_conditioning(result: dict[str, np.ndarray], users: int, dimension: int) -> None:
    for values_name, mask_name in (("history", "history_mask"), ("evidence", "evidence_mask")):
        values = result[values_name]
        mask = result[mask_name]
        if values.ndim != 3 or values.shape[0] != users or values.shape[2] != dimension:
            raise ValueError(f"invalid {values_name} shape")
        if mask.dtype != np.bool_ or mask.shape != values.shape[:2]:
            raise ValueError(f"{mask_name} must be boolean and match {values_name}")
        if mask.all(axis=1).any() and values_name == "history":
            raise ValueError("every user needs at least one observed-history row")
        _require_unit_rows(values_name, values, mask)


def load_training_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load dense padded arrays used by ``scripts/train_transport.py``."""
    required = {
        "source",
        "target",
        "history",
        "history_mask",
        "evidence",
        "evidence_mask",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"missing training arrays: {sorted(missing)}")
        result = {name: np.asarray(archive[name]) for name in required}
    source, target = result["source"], result["target"]
    if source.ndim != 3 or source.shape != target.shape or len(source) == 0:
        raise ValueError("source and target must share nonempty [users, samples, dimension] axes")
    _require_unit_rows("source", source)
    _require_unit_rows("target", target)
    _validate_conditioning(result, len(source), source.shape[-1])
    return result


def load_source_calibration_npz(path: str | Path) -> dict[str, np.ndarray]:
    required = {
        "source",
        "target",
        "target_valid",
        "history",
        "history_mask",
        "evidence",
        "evidence_mask",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"missing source-calibration arrays: {sorted(missing)}")
        result = {name: np.asarray(archive[name]) for name in required}
    source, target, target_valid = result["source"], result["target"], result["target_valid"]
    if source.ndim != 3 or target.ndim != 3 or len(source) == 0:
        raise ValueError("source and target must use [users, samples, dimension] axes")
    if source.shape[0] != target.shape[0] or source.shape[2] != target.shape[2]:
        raise ValueError("source and target user or embedding axes differ")
    if target_valid.dtype != np.bool_ or target_valid.shape != target.shape[:2]:
        raise ValueError("target_valid must be boolean and match target axes")
    if (~target_valid).all(axis=1).any():
        raise ValueError("every user needs at least one valid target")
    _require_unit_rows("source", source)
    _require_unit_rows("target", target, ~target_valid)
    _validate_conditioning(result, len(source), source.shape[-1])
    return result


def load_generation_npz(path: str | Path) -> dict[str, np.ndarray]:
    required = {
        "user_id",
        "source",
        "history",
        "history_mask",
        "evidence",
        "evidence_mask",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"missing generation arrays: {sorted(missing)}")
        result = {name: np.asarray(archive[name]) for name in required}
    source = result["source"]
    if source.ndim != 3 or len(source) == 0 or result["user_id"].shape != (len(source),):
        raise ValueError("generation source and user axes are invalid")
    if len(set(map(int, result["user_id"]))) != len(source):
        raise ValueError("generation user IDs must be unique")
    _require_unit_rows("source", source)
    _validate_conditioning(result, len(source), source.shape[-1])
    return result


def load_evaluation_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load generated arrays and flattened ragged history/target axes."""
    required = {
        "user_id",
        "generated",
        "history_values",
        "history_offsets",
        "target_values",
        "target_offsets",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"missing evaluation arrays: {sorted(missing)}")
        result = {name: np.asarray(archive[name]) for name in required}
    users = len(result["user_id"])
    if result["user_id"].shape != (users,) or len(set(map(int, result["user_id"]))) != users:
        raise ValueError("evaluation user IDs must form a unique one-dimensional axis")
    for name in ("history_offsets", "target_offsets"):
        offsets = result[name]
        if offsets.shape != (users + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError(f"invalid {name}")
    if result["history_offsets"][-1] != len(result["history_values"]):
        raise ValueError("history_offsets do not cover history_values")
    if result["target_offsets"][-1] != len(result["target_values"]):
        raise ValueError("target_offsets do not cover target_values")
    if result["generated"].ndim != 3 or len(result["generated"]) != users:
        raise ValueError("generated must have shape [users, samples, dimension]")
    dimension = result["generated"].shape[-1]
    if result["history_values"].ndim != 2 or result["history_values"].shape[1] != dimension:
        raise ValueError("history_values embedding axis differs")
    if result["target_values"].ndim != 2 or result["target_values"].shape[1] != dimension:
        raise ValueError("target_values embedding axis differs")
    if (np.diff(result["history_offsets"]) < 1).any() or (np.diff(result["target_offsets"]) < 1).any():
        raise ValueError("every evaluation user needs history and target rows")
    _require_unit_rows("generated", result["generated"])
    _require_unit_rows("history_values", result["history_values"])
    _require_unit_rows("target_values", result["target_values"])
    return result


def ragged_row(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    return values[int(offsets[index]) : int(offsets[index + 1])]
