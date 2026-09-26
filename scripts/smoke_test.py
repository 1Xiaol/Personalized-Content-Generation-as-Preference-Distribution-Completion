#!/usr/bin/env python3
"""Run transport, source calibration, metric, and judge paths on synthetic data."""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from recdiffusion.geometry import spherical_heun
from recdiffusion.judge import classify_aspect_states, incremental_target_ids
from recdiffusion.metrics import score_user
from recdiffusion.source_calibration import ConditionalSourceCalibration, SourceCalibrationConfig
from recdiffusion.transport import TransportConfig, build_transport


def main() -> None:
    torch.manual_seed(7)
    config = TransportConfig.toy(8)
    model = build_transport("dual_view_conditioned_rfm", config, seed=7).eval()
    history = F.normalize(torch.randn(2, 3, 8), dim=-1)
    evidence = F.normalize(torch.randn(2, 2, 8), dim=-1)
    history_mask = torch.zeros(2, 3, dtype=torch.bool)
    evidence_mask = torch.tensor([[False, False], [True, True]])
    source = F.normalize(torch.randn(2, 4, 8), dim=-1)
    memory = model.encode(history, history_mask, evidence, evidence_mask)
    source_calibration = ConditionalSourceCalibration(SourceCalibrationConfig(
        cond_dim=config.condition_dim * (config.global_tokens + 1),
        hidden=16,
        semantic_dim=8,
        history_tokens=config.global_tokens,
    ))
    adjusted = source_calibration(source, memory)
    if not torch.allclose(adjusted, source, atol=2e-6, rtol=0):
        raise AssertionError("conditional source calibration is not identity initialized")
    with torch.no_grad():
        generated = spherical_heun(lambda value, time: model.velocity(value, time, memory), adjusted, steps=2)
    if not torch.allclose(generated.norm(dim=-1), torch.ones(2, 4), atol=2e-5, rtol=0):
        raise AssertionError("spherical rollout left the unit sphere")

    metric = score_user(
        generated[0].numpy(),
        F.normalize(torch.randn(3, 8), dim=-1).numpy(),
        history[0].numpy(),
        bandwidth=1.0,
    )
    states = classify_aspect_states(
        ["M1", "M2"],
        [
            {
                "direct_aspect_ids": ["M1"],
                "related_aspect_ids": ["M2"],
                "uncertain_aspect_ids": [],
            }
        ],
    )
    incremental = incremental_target_ids(
        [
            {"aspect_id": "M1", "target_ids": ["T1"]},
            {"aspect_id": "M2", "target_ids": ["T2"]},
        ],
        states,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "transport": "dual_view_conditioned_rfm",
                "metric_keys": sorted(metric),
                "incremental_targets": sorted(incremental),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
