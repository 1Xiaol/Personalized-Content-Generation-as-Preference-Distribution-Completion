"""Identity-initialized conditional source calibration on the unit sphere."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .geometry import project_tangent, sphere_exp


@dataclass(frozen=True)
class SourceCalibrationConfig:
    cond_dim: int = 2304
    hidden: int = 768
    semantic_dim: int = 512
    history_tokens: int = 8
    max_angle: float = 0.20


def condition_from_memory(memory: tuple, history_tokens: int = 8) -> torch.Tensor:
    if len(memory) == 5:
        history, history_mask, evidence, evidence_mask, present = memory
        if evidence_mask.dtype != torch.bool or present.dtype != torch.bool:
            raise ValueError("evidence masks must be boolean")
        if evidence.shape[1] < history_tokens:
            raise ValueError("evidence memory does not contain the configured global tokens")
        valid = (~evidence_mask[:, history_tokens:]) & present[:, None]
        values = evidence[:, history_tokens:].float().masked_fill(~valid[..., None], 0)
        supplementary = values.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
    elif len(memory) == 2:
        history, history_mask = memory
        valid = ~history_mask[:, history_tokens:]
        values = history[:, history_tokens:].float().masked_fill(~valid[..., None], 0)
        supplementary = values.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
    else:
        raise ValueError("unsupported transport memory structure")
    if history_mask.dtype != torch.bool:
        raise ValueError("history mask must be boolean")
    if history.shape[1] < history_tokens:
        raise ValueError("history memory does not contain the configured global tokens")
    if history_mask[:, :history_tokens].any():
        raise ValueError("history global tokens cannot be padding")
    history_condition = history[:, :history_tokens].float().flatten(1)
    return torch.cat((history_condition, supplementary), dim=-1)


class ConditionalSourceCalibration(nn.Module):
    """Bounded spherical source correction conditioned on transport memories."""

    def __init__(
        self,
        config: SourceCalibrationConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or SourceCalibrationConfig()
        cond_dim = config.cond_dim
        hidden = config.hidden
        semantic_dim = config.semantic_dim
        history_tokens = config.history_tokens
        max_angle = config.max_angle
        if min(cond_dim, hidden, semantic_dim, history_tokens) < 1 or not 0 < max_angle < math.pi:
            raise ValueError("invalid conditional source calibration configuration")
        self.config = config
        self.history_tokens = history_tokens
        self.semantic_dim = semantic_dim
        self.max_angle = max_angle
        self.cond_norm = nn.LayerNorm(cond_dim)
        self.cond = nn.Linear(cond_dim, hidden)
        self.source_norm = nn.LayerNorm(semantic_dim)
        self.source = nn.Linear(semantic_dim, hidden)
        self.mix = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.out_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, semantic_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, source: torch.Tensor, memory: tuple, *, return_movement: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        condition = condition_from_memory(memory, self.history_tokens)
        if source.ndim != 3 or source.shape[0] != len(condition) or source.shape[-1] != self.semantic_dim:
            raise ValueError("source and memory axes differ")
        hidden = self.source(self.source_norm(source.float())) + self.cond(self.cond_norm(condition))[:, None]
        hidden = hidden + self.mix(hidden)
        raw = self.output(self.out_norm(hidden)).float()
        tangent = project_tangent(source, raw)
        radius = tangent.norm(dim=-1, keepdim=True)
        scale = self.max_angle * torch.where(
            radius < 1e-4,
            1 - radius.square() / 3,
            torch.tanh(radius) / radius.clamp_min(1e-12),
        )
        movement = tangent * scale
        adjusted = sphere_exp(source, movement)
        adjusted = adjusted + torch.where(
            radius == 0, source.float() - adjusted, torch.zeros_like(adjusted)
        ).detach()
        return (adjusted, movement) if return_movement else adjusted


def source_calibration_loss(
    generated: torch.Tensor,
    targets: torch.Tensor,
    target_valid: torch.Tensor,
    movement: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """User-level Energy + 0.10 coverage + 0.01 source displacement."""
    if generated.ndim != 3 or targets.ndim != 3 or movement.shape != generated.shape:
        raise ValueError("expected [user, sample, dimension] tensors")
    if target_valid.dtype != torch.bool or target_valid.shape != targets.shape[:2]:
        raise ValueError("target_valid must match target axes")
    counts = target_valid.sum(1)
    if generated.shape[1] < 2 or (counts < 1).any():
        raise ValueError("source calibration needs at least two samples and one target")
    generated = generated.float()
    targets = targets.float().masked_fill(~target_valid[..., None], 0)
    cross_distance = torch.cdist(generated, targets).masked_fill(~target_valid[:, None], 0)
    pair_distance = torch.cdist(generated, generated)
    samples = generated.shape[1]
    energy = cross_distance.sum((1, 2)) / (samples * counts) - pair_distance.sum((1, 2)) / (
        2 * samples * (samples - 1)
    )
    cosine = torch.einsum("bkd,bmd->bkm", generated, targets)
    cosine = cosine.masked_fill(~target_valid[:, None], -1.0).max(1).values
    angles = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7)).masked_fill(~target_valid, 0)
    coverage = angles.sum(1) / counts
    displacement = movement.float().square().sum(-1).mean(1)
    total = energy + 0.10 * coverage + 0.01 * displacement
    return {"loss": total, "energy": energy, "coverage_rad": coverage, "movement": displacement}
