import torch
from torch.nn import functional as F

from recdiffusion.geometry import parallel_transport, sphere_exp, sphere_log, spherical_heun


def test_exp_log_round_trip() -> None:
    torch.manual_seed(1)
    base = F.normalize(torch.randn(7, 8), dim=-1)
    target = F.normalize(base + 0.2 * torch.randn(7, 8), dim=-1)
    reconstructed = sphere_exp(base, sphere_log(base, target))
    assert torch.allclose(reconstructed, target, atol=2e-5, rtol=0)


def test_parallel_transport_is_tangent_at_destination() -> None:
    start = F.normalize(torch.randn(5, 8), dim=-1)
    end = F.normalize(start + 0.1 * torch.randn(5, 8), dim=-1)
    tangent = torch.randn(5, 8)
    tangent -= (tangent * start).sum(-1, keepdim=True) * start
    transported = parallel_transport(start, end, tangent)
    assert torch.allclose((transported * end).sum(-1), torch.zeros(5), atol=1e-5, rtol=0)


def test_zero_velocity_heun_preserves_source() -> None:
    source = F.normalize(torch.randn(2, 4, 8), dim=-1)
    result = spherical_heun(lambda value, time: torch.zeros_like(value), source, steps=3)
    assert torch.allclose(result, source, atol=1e-6, rtol=0)

