"""Numerically stable operations on the unit hypersphere."""
from __future__ import annotations

import math
from collections.abc import Callable

import torch


def normalize(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)


def project_tangent(base: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Project ``value`` onto the tangent space at unit vector ``base``."""
    return value - (value * base).sum(dim=-1, keepdim=True) * base


def sphere_exp(base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    """Exponential map on the unit sphere."""
    radius = tangent.norm(dim=-1, keepdim=True)
    result = torch.cos(radius) * base + torch.sinc(radius / math.pi) * tangent
    return normalize(result)


def sphere_log(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Shortest-path logarithmic map from ``base`` to ``target``."""
    cosine = (base * target).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cosine)
    direction = target - cosine * base
    return direction * (angle / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def parallel_transport(start: torch.Tensor, end: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    """Transport a tangent vector from ``start`` to ``end`` on the short geodesic."""
    denominator = (1 + (start * end).sum(dim=-1, keepdim=True)).clamp_min(1e-7)
    transported = tangent - (tangent * end).sum(dim=-1, keepdim=True) * (start + end) / denominator
    return project_tangent(end, transported)


def spherical_heun(
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    source: torch.Tensor,
    *,
    steps: int = 32,
) -> torch.Tensor:
    """Second-order Heun integration while keeping every state on the sphere."""
    if steps < 1:
        raise ValueError("steps must be positive")
    value = normalize(source.float())
    step_size = 1.0 / steps
    for index in range(steps):
        time = torch.full((*value.shape[:-1], 1), index * step_size, device=value.device)
        first = project_tangent(value, velocity(value, time).float())
        predictor = sphere_exp(value, first * step_size)
        second = project_tangent(predictor, velocity(predictor, time + step_size).float())
        second_at_value = parallel_transport(predictor, value, second)
        value = sphere_exp(value, (first + second_at_value) * (0.5 * step_size))
    return value

