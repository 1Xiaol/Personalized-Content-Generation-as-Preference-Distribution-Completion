"""Training objectives for history-conditioned and dual-view transports."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .geometry import parallel_transport, sphere_exp, sphere_log


def rectified_flow_loss(
    model: torch.nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
    history_mask: torch.Tensor,
    evidence: torch.Tensor | None = None,
    evidence_mask: torch.Tensor | None = None,
    *,
    context_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Spherical conditional flow matching plus in-batch context InfoNCE."""
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("source and target must share [batch, sample, dimension]")
    batch, samples, _ = source.shape
    if getattr(model, "route", None) == "dual_view_conditioned_rfm" and (
        evidence is None or evidence_mask is None
    ):
        raise ValueError("dual-view conditioning requires collaborative evidence")
    memory = model.encode(history, history_mask, evidence, evidence_mask)
    time = torch.rand((batch, samples, 1), device=source.device)
    initial_velocity = sphere_log(source.float(), target.float())
    state = sphere_exp(source.float(), time * initial_velocity)
    desired = parallel_transport(source.float(), state, initial_velocity)
    predicted = model.velocity(state, time, memory)
    flow = F.mse_loss(predicted.float(), desired.float())

    target_pool = target[:, 0]
    logits = model.context_logits(memory, target_pool)
    labels = torch.arange(batch, device=source.device)
    context = F.cross_entropy(logits, labels)
    total = flow + context_weight * context
    return {"loss": total, "flow": flow, "context": context}
