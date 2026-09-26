import numpy as np
import torch
from torch.nn import functional as F

from recdiffusion.checkpoints import (
    load_source_calibration,
    load_transport,
    save_source_calibration,
    save_transport,
)
from recdiffusion.data import (
    load_evaluation_npz,
    load_source_calibration_npz,
    load_training_npz,
)
from recdiffusion.source_calibration import ConditionalSourceCalibration, SourceCalibrationConfig
from recdiffusion.transport import TransportConfig, build_transport


def unit(rng: np.random.Generator, *shape: int) -> np.ndarray:
    value = rng.normal(size=shape).astype(np.float32)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def test_npz_contracts_accept_padded_evidence(tmp_path) -> None:
    rng = np.random.default_rng(5)
    source = unit(rng, 3, 2, 8)
    history = unit(rng, 3, 2, 8)
    target = unit(rng, 3, 3, 8)
    evidence = unit(rng, 3, 2, 8)
    history_mask = np.zeros((3, 2), dtype=bool)
    evidence_mask = np.ones((3, 2), dtype=bool)
    np.savez(
        tmp_path / "train.npz",
        source=source,
        target=source.copy(),
        history=history,
        history_mask=history_mask,
        evidence=evidence,
        evidence_mask=evidence_mask,
    )
    load_training_npz(tmp_path / "train.npz")
    np.savez(
        tmp_path / "calibration.npz",
        source=source,
        target=target,
        target_valid=np.asarray([[True, True, False], [True, True, True], [True, False, False]]),
        history=history,
        history_mask=history_mask,
        evidence=evidence,
        evidence_mask=evidence_mask,
    )
    load_source_calibration_npz(tmp_path / "calibration.npz")


def test_checkpoint_round_trip(tmp_path) -> None:
    config = TransportConfig.toy(8)
    model = build_transport("dual_view_conditioned_rfm", config, seed=4)
    transport_path = tmp_path / "transport.pt"
    save_transport(transport_path, model)
    restored = load_transport(transport_path)
    for left, right in zip(model.state_dict().values(), restored.state_dict().values(), strict=True):
        assert torch.equal(left, right)

    calibration = ConditionalSourceCalibration(
        SourceCalibrationConfig(
            cond_dim=config.condition_dim * (config.global_tokens + 1),
            hidden=16,
            semantic_dim=8,
            history_tokens=config.global_tokens,
        )
    )
    calibration_path = tmp_path / "calibration.pt"
    save_source_calibration(
        calibration_path,
        calibration,
        transport_route=model.route,
        transport_config=config,
    )
    restored_calibration = load_source_calibration(calibration_path)
    for left, right in zip(
        calibration.state_dict().values(), restored_calibration.state_dict().values(), strict=True
    ):
        assert torch.equal(left, right)


def test_evaluation_contract_rejects_missing_rows(tmp_path) -> None:
    values = unit(np.random.default_rng(7), 2, 8)
    np.savez(
        tmp_path / "bad.npz",
        user_id=np.asarray([0, 1]),
        generated=unit(np.random.default_rng(8), 2, 2, 8),
        history_values=values[:1],
        history_offsets=np.asarray([0, 1, 1]),
        target_values=values,
        target_offsets=np.asarray([0, 1, 2]),
    )
    try:
        load_evaluation_npz(tmp_path / "bad.npz")
    except ValueError as error:
        assert "every evaluation user needs history" in str(error)
    else:
        raise AssertionError("empty history rows must be rejected")
