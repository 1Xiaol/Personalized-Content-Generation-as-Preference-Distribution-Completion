import torch
from torch.nn import functional as F

from recdiffusion.source_calibration import ConditionalSourceCalibration, SourceCalibrationConfig
from recdiffusion.training import rectified_flow_loss
from recdiffusion.transport import TransportConfig, build_transport


def synthetic_batch():
    torch.manual_seed(2)
    history = F.normalize(torch.randn(3, 4, 8), dim=-1)
    evidence = F.normalize(torch.randn(3, 2, 8), dim=-1)
    history_mask = torch.zeros(3, 4, dtype=torch.bool)
    evidence_mask = torch.tensor([[False, False], [False, True], [True, True]])
    source = F.normalize(torch.randn(3, 2, 8), dim=-1)
    target = F.normalize(torch.randn(3, 2, 8), dim=-1)
    return history, evidence, history_mask, evidence_mask, source, target


def test_history_and_dual_view_transport_shapes() -> None:
    config = TransportConfig.toy(8)
    history, evidence, history_mask, evidence_mask, source, _ = synthetic_batch()
    for route in ("history_conditioned_rfm", "dual_view_conditioned_rfm"):
        model = build_transport(route, config)
        memory = model.encode(history, history_mask, evidence, evidence_mask)
        velocity = model.velocity(source, torch.full((3, 2, 1), 0.5), memory)
        assert velocity.shape == source.shape
        assert torch.isfinite(velocity).all()


def test_source_calibration_is_exact_identity_at_initialization() -> None:
    config = TransportConfig.toy(8)
    history, evidence, history_mask, evidence_mask, source, _ = synthetic_batch()
    model = build_transport("dual_view_conditioned_rfm", config)
    memory = model.encode(history, history_mask, evidence, evidence_mask)
    source_calibration = ConditionalSourceCalibration(SourceCalibrationConfig(
        cond_dim=config.condition_dim * (config.global_tokens + 1),
        hidden=16,
        semantic_dim=8,
        history_tokens=config.global_tokens,
    ))
    assert torch.allclose(source_calibration(source, memory), source, atol=2e-6, rtol=0)


def test_source_calibration_accepts_history_conditioned_memory() -> None:
    config = TransportConfig.toy(8)
    history, evidence, history_mask, evidence_mask, source, _ = synthetic_batch()
    model = build_transport("history_conditioned_rfm", config)
    memory = model.encode(history, history_mask, evidence, evidence_mask)
    source_calibration = ConditionalSourceCalibration(SourceCalibrationConfig(
        cond_dim=config.condition_dim * (config.global_tokens + 1),
        hidden=16,
        semantic_dim=8,
        history_tokens=config.global_tokens,
    ))
    assert torch.allclose(source_calibration(source, memory), source, atol=2e-6, rtol=0)


def test_rectified_flow_loss_is_differentiable() -> None:
    config = TransportConfig.toy(8)
    history, evidence, history_mask, evidence_mask, source, target = synthetic_batch()
    model = build_transport("dual_view_conditioned_rfm", config)
    losses = rectified_flow_loss(
        model, source, target, history, history_mask, evidence, evidence_mask
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert any(parameter.grad is not None for parameter in model.parameters())
