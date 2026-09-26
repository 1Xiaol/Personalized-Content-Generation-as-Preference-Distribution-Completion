"""History-conditioned and dual-view conditioned Riemannian flow transports."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import project_tangent


@dataclass(frozen=True)
class TransportConfig:
    semantic_dim: int = 512
    condition_dim: int = 256
    flow_dim: int = 512
    condition_blocks: int = 4
    flow_blocks: int = 10
    heads: int = 8
    condition_ffn: int = 1024
    flow_ffn: int = 2048
    global_tokens: int = 8
    time_features: int = 128
    dropout: float = 0.1

    @classmethod
    def toy(cls, semantic_dim: int = 8) -> "TransportConfig":
        return cls(semantic_dim, 16, 16, 1, 2, 1, 32, 32, 2, 16, 0.0)


class RMSNorm(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        return (value.float() * scale).to(value.dtype) * self.weight.to(value.dtype)


class SetBlock(nn.Module):
    def __init__(self, config: TransportConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.condition_dim)
        self.attention = nn.MultiheadAttention(
            config.condition_dim, config.heads, dropout=config.dropout, batch_first=True
        )
        self.norm2 = RMSNorm(config.condition_dim)
        self.ffn_in = nn.Linear(config.condition_dim, 2 * config.condition_ffn)
        self.ffn_out = nn.Linear(config.condition_ffn, config.condition_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        attended = self.attention(
            normalized, normalized, normalized, key_padding_mask=mask, need_weights=False
        )[0]
        value = value + self.dropout(attended)
        gate, content = self.ffn_in(self.norm2(value)).chunk(2, -1)
        return value + self.dropout(self.ffn_out(F.silu(gate) * content))


class ContextReadout(nn.Module):
    def __init__(self, config: TransportConfig) -> None:
        super().__init__()
        dimension = config.condition_dim
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.heads = config.heads
        self.head_dim = dimension // config.heads

    def forward(self, query: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = memory.shape
        key = self.key(memory).view(batch, length, self.heads, self.head_dim)
        value = self.value(memory).view(batch, length, self.heads, self.head_dim)
        projected = self.query(query).view(batch, -1, self.heads, self.head_dim)
        score = torch.einsum("bqhd,blhd->bqhl", projected, key) / math.sqrt(self.head_dim)
        score = score.masked_fill(mask[:, None, None], -torch.inf)
        probability = torch.softmax(score.float(), -1).to(score.dtype)
        readout = torch.einsum("bqhl,blhd->bqhd", probability, value).reshape(batch, -1, dimension)
        return self.output(readout)


class SetEncoder(nn.Module):
    def __init__(self, config: TransportConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(config.semantic_dim, config.condition_dim)
        self.input_norm = RMSNorm(config.condition_dim)
        self.global_tokens = nn.Parameter(torch.empty(config.global_tokens, config.condition_dim))
        nn.init.normal_(self.global_tokens, std=0.02)
        self.blocks = nn.ModuleList([SetBlock(config) for _ in range(config.condition_blocks)])
        self.query_projection = nn.Linear(config.semantic_dim, config.condition_dim)
        self.readout = ContextReadout(config)

    def encode(self, value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_mask = torch.zeros(
            (len(value), len(self.global_tokens)), dtype=torch.bool, device=value.device
        )
        memory = torch.cat(
            (self.global_tokens[None].expand(len(value), -1, -1), self.input_norm(self.input_projection(value))),
            dim=1,
        )
        mask = torch.cat((global_mask, mask.to(torch.bool)), dim=1)
        for block in self.blocks:
            memory = block(memory, mask)
        return memory.masked_fill(mask[..., None], 0), mask


def _time_embedding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    frequency = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), dimension // 2, device=time.device)
    )
    angle = 2 * math.pi * time.float() * frequency
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)


class HistoryFlowBlock(nn.Module):
    def __init__(self, config: TransportConfig) -> None:
        super().__init__()
        self.norm_cross = RMSNorm(config.flow_dim)
        self.semantic_query = nn.Linear(config.semantic_dim, config.flow_dim)
        self.attention = nn.MultiheadAttention(
            config.flow_dim,
            config.heads,
            batch_first=True,
            kdim=config.condition_dim,
            vdim=config.condition_dim,
        )
        self.film = nn.Linear(config.flow_dim, 2 * config.flow_dim)
        self.norm_ffn = RMSNorm(config.flow_dim)
        self.ffn_in = nn.Linear(config.flow_dim, 2 * config.flow_ffn)
        self.ffn_out = nn.Linear(config.flow_ffn, config.flow_dim)

    def forward(
        self,
        state: torch.Tensor,
        semantic: torch.Tensor,
        time: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = self.film(time).chunk(2, -1)
        query = self.norm_cross(state) * (1 + scale) + shift + self.semantic_query(semantic)
        state = state + self.attention(query, memory, memory, key_padding_mask=mask, need_weights=False)[0]
        gate, content = self.ffn_in(self.norm_ffn(state)).chunk(2, -1)
        return state + self.ffn_out(F.silu(gate) * content)


class HistoryConditionedRFM(nn.Module):
    """Riemannian flow matching conditioned only on observed history."""

    route = "history_conditioned_rfm"

    def __init__(self, config: TransportConfig) -> None:
        super().__init__()
        self.config = config
        self.config_record = asdict(config)
        self.history = SetEncoder(config)
        self.state_input = nn.Linear(config.semantic_dim, config.flow_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_features, config.flow_dim), nn.SiLU(), nn.Linear(config.flow_dim, config.flow_dim)
        )
        self.flow_blocks = nn.ModuleList([HistoryFlowBlock(config) for _ in range(config.flow_blocks)])
        self.output_norm = RMSNorm(config.flow_dim)
        self.output = nn.Linear(config.flow_dim, config.semantic_dim)

    def encode(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        evidence: torch.Tensor | None = None,
        evidence_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del evidence, evidence_mask
        return self.history.encode(history, history_mask)

    def velocity(self, value: torch.Tensor, time: torch.Tensor, memory: tuple[torch.Tensor, ...]) -> torch.Tensor:
        encoded, mask = memory
        state = self.state_input(value)
        embedded_time = self.time_mlp(_time_embedding(time, self.config.time_features))
        for block in self.flow_blocks:
            state = block(state, value, embedded_time, encoded, mask)
        return project_tangent(value, self.output(self.output_norm(state)).float())

    def context_logits(self, memory: tuple[torch.Tensor, ...], target: torch.Tensor) -> torch.Tensor:
        encoded, mask = memory
        query = self.history.query_projection(target)[None].expand(len(encoded), -1, -1)
        readout = self.history.readout(query, encoded, mask)
        query = F.normalize(query.float(), dim=-1, eps=1e-12)
        readout = F.normalize(readout.float(), dim=-1, eps=1e-12)
        return (query * readout).sum(-1)


def _masked_fusion(logits: torch.Tensor, evidence_present: torch.Tensor) -> torch.Tensor:
    available = torch.stack((torch.ones_like(evidence_present), evidence_present), dim=-1)[:, None]
    return torch.softmax(logits.float().masked_fill(~available, -torch.inf), dim=-1)


class DualViewFlowBlock(nn.Module):
    def __init__(self, config: TransportConfig) -> None:
        super().__init__()
        self.norm_cross = RMSNorm(config.flow_dim)
        self.history_query = nn.Linear(config.semantic_dim, config.flow_dim)
        self.evidence_query = nn.Linear(config.semantic_dim, config.flow_dim)
        attention = dict(
            embed_dim=config.flow_dim,
            num_heads=config.heads,
            batch_first=True,
            kdim=config.condition_dim,
            vdim=config.condition_dim,
        )
        self.history_attention = nn.MultiheadAttention(**attention)
        self.evidence_attention = nn.MultiheadAttention(**attention)
        self.fusion = nn.Sequential(
            nn.Linear(3 * config.flow_dim, config.flow_dim), nn.SiLU(), nn.Linear(config.flow_dim, 2)
        )
        nn.init.normal_(self.fusion[-1].weight, std=1e-3)
        nn.init.zeros_(self.fusion[-1].bias)
        self.film = nn.Linear(config.flow_dim, 2 * config.flow_dim)
        self.norm_ffn = RMSNorm(config.flow_dim)
        self.ffn_in = nn.Linear(config.flow_dim, 2 * config.flow_ffn)
        self.ffn_out = nn.Linear(config.flow_ffn, config.flow_dim)

    def forward(self, state: torch.Tensor, semantic: torch.Tensor, time: torch.Tensor, memory: tuple) -> torch.Tensor:
        history, history_mask, evidence, evidence_mask, present = memory
        scale, shift = self.film(time).chunk(2, -1)
        modulated = self.norm_cross(state) * (1 + scale) + shift
        history_read = self.history_attention(
            modulated + self.history_query(semantic),
            history,
            history,
            key_padding_mask=history_mask,
            need_weights=False,
        )[0]
        evidence_read = self.evidence_attention(
            modulated + self.evidence_query(semantic),
            evidence,
            evidence,
            key_padding_mask=evidence_mask,
            need_weights=False,
        )[0]
        evidence_read = evidence_read.masked_fill(~present[:, None, None], 0)
        weights = _masked_fusion(self.fusion(torch.cat((modulated, history_read, evidence_read), -1)), present)
        state = state + weights[..., :1] * history_read + weights[..., 1:] * evidence_read
        gate, content = self.ffn_in(self.norm_ffn(state)).chunk(2, -1)
        return state + self.ffn_out(F.silu(gate) * content)


class DualViewConditionedRFM(nn.Module):
    """Riemannian flow matching with separate history and collaborative-evidence memories."""

    route = "dual_view_conditioned_rfm"

    def __init__(self, config: TransportConfig) -> None:
        super().__init__()
        self.config = config
        self.config_record = asdict(config)
        self.history = SetEncoder(config)
        self.evidence = SetEncoder(config)
        self.context_fusion = nn.Sequential(
            nn.Linear(4 * config.condition_dim, config.condition_dim), nn.GELU(), nn.Linear(config.condition_dim, 2)
        )
        nn.init.normal_(self.context_fusion[-1].weight, std=1e-3)
        nn.init.zeros_(self.context_fusion[-1].bias)
        self.state_input = nn.Linear(config.semantic_dim, config.flow_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_features, config.flow_dim), nn.SiLU(), nn.Linear(config.flow_dim, config.flow_dim)
        )
        self.flow_blocks = nn.ModuleList([DualViewFlowBlock(config) for _ in range(config.flow_blocks)])
        self.output_norm = RMSNorm(config.flow_dim)
        self.output = nn.Linear(config.flow_dim, config.semantic_dim)

    def encode(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        evidence: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> tuple:
        history_memory, history_full_mask = self.history.encode(history, history_mask)
        evidence_memory, evidence_full_mask = self.evidence.encode(evidence, evidence_mask)
        return (
            history_memory,
            history_full_mask,
            evidence_memory,
            evidence_full_mask,
            (~evidence_mask).any(1),
        )

    def velocity(self, value: torch.Tensor, time: torch.Tensor, memory: tuple) -> torch.Tensor:
        state = self.state_input(value)
        embedded_time = self.time_mlp(_time_embedding(time, self.config.time_features))
        for block in self.flow_blocks:
            state = block(state, value, embedded_time, memory)
        return project_tangent(value, self.output(self.output_norm(state)).float())

    def context_logits(self, memory: tuple, target: torch.Tensor) -> torch.Tensor:
        history, history_mask, evidence, evidence_mask, present = memory
        history_query = self.history.query_projection(target)[None].expand(len(history), -1, -1)
        evidence_query = self.evidence.query_projection(target)[None].expand(len(history), -1, -1)
        history_read = self.history.readout(history_query, history, history_mask)
        evidence_read = self.evidence.readout(evidence_query, evidence, evidence_mask)
        history_query, evidence_query, history_read, evidence_read = [
            F.normalize(item.float(), dim=-1, eps=1e-12)
            for item in (history_query, evidence_query, history_read, evidence_read)
        ]
        evidence_query = evidence_query.masked_fill(~present[:, None, None], 0)
        evidence_read = evidence_read.masked_fill(~present[:, None, None], 0)
        weights = _masked_fusion(
            self.context_fusion(torch.cat((history_query, history_read, evidence_query, evidence_read), -1)),
            present,
        )
        return weights[..., 0] * (history_query * history_read).sum(-1) + weights[..., 1] * (
            evidence_query * evidence_read
        ).sum(-1)


def build_transport(
    route: str, config: TransportConfig | None = None, seed: int = 2026
) -> nn.Module:
    config = config or TransportConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if route == "history_conditioned_rfm":
            return HistoryConditionedRFM(config)
        if route == "dual_view_conditioned_rfm":
            return DualViewConditionedRFM(config)
    raise ValueError(f"unknown route: {route}")


def parameter_groups(model: nn.Module) -> list[dict]:
    """Optimizer groups for the conditioning encoders and transport network."""
    if getattr(model, "route", None) == "dual_view_conditioned_rfm":
        encoder_parameters = list(model.history.parameters()) + list(model.evidence.parameters())
    else:
        encoder_parameters = list(model.history.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    remaining = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    return [
        {"params": encoder_parameters, "name": "encoders", "lr": 5e-5},
        {"params": remaining, "name": "transport_fusion", "lr": 1e-4},
    ]
