#!/usr/bin/env python3
"""Train conditional source calibration against a frozen transport."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch

from recdiffusion.checkpoints import load_transport, save_source_calibration
from recdiffusion.data import load_source_calibration_npz
from recdiffusion.geometry import spherical_heun
from recdiffusion.io import read_json
from recdiffusion.source_calibration import (
    ConditionalSourceCalibration,
    SourceCalibrationConfig,
    source_calibration_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--heun-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    arrays = load_source_calibration_npz(args.input)
    device = torch.device(args.device)
    transport = load_transport(args.transport, device).eval()
    transport.requires_grad_(False)
    default = SourceCalibrationConfig(
        cond_dim=transport.config.condition_dim * (transport.config.global_tokens + 1),
        semantic_dim=transport.config.semantic_dim,
        history_tokens=transport.config.global_tokens,
    )
    config = default
    if args.config is not None:
        raw = read_json(args.config)
        if "max_angle_rad" in raw and "max_angle" not in raw:
            raw["max_angle"] = raw["max_angle_rad"]
        names = {field.name for field in fields(SourceCalibrationConfig)}
        values = {name: raw.get(name, getattr(default, name)) for name in names}
        config = SourceCalibrationConfig(**values)
    expected_cond_dim = transport.config.condition_dim * (transport.config.global_tokens + 1)
    if config.cond_dim != expected_cond_dim or config.semantic_dim != transport.config.semantic_dim:
        raise ValueError("source-calibration dimensions do not match the transport")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    calibration = ConditionalSourceCalibration(config).to(device)
    optimizer = torch.optim.AdamW(calibration.parameters(), lr=args.learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed)
    users = len(arrays["source"])
    for update in range(args.updates):
        index = torch.randint(users, (args.batch_size,), generator=generator)
        batch = {
            name: torch.from_numpy(values[index.numpy()]).to(device)
            for name, values in arrays.items()
        }
        with torch.no_grad():
            memory = transport.encode(
                batch["history"],
                batch["history_mask"],
                batch["evidence"],
                batch["evidence_mask"],
            )
        adjusted, movement = calibration(batch["source"], memory, return_movement=True)
        generated = spherical_heun(
            lambda value, time: transport.velocity(value, time, memory),
            adjusted,
            steps=args.heun_steps,
        )
        losses = source_calibration_loss(
            generated, batch["target"], batch["target_valid"], movement
        )
        loss = losses["loss"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibration.parameters(), 1.0)
        optimizer.step()
        if update % 100 == 0 or update + 1 == args.updates:
            record = {"update": update + 1, **{name: float(value.mean().detach()) for name, value in losses.items()}}
            print(json.dumps(record))
    save_source_calibration(
        args.output,
        calibration,
        transport_route=transport.route,
        transport_config=transport.config,
    )
    print(json.dumps({"checkpoint": str(args.output)}))


if __name__ == "__main__":
    main()

