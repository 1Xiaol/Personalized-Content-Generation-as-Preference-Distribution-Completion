#!/usr/bin/env python3
"""Reference single-process trainer for the dense transport data contract."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch

from recdiffusion.checkpoints import save_transport
from recdiffusion.data import load_training_npz
from recdiffusion.io import read_json
from recdiffusion.transport import TransportConfig, build_transport, parameter_groups
from recdiffusion.training import rectified_flow_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--route",
        choices=("history_conditioned_rfm", "dual_view_conditioned_rfm"),
        required=True,
    )
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    arrays = load_training_npz(args.input)
    device = torch.device(args.device)
    config = TransportConfig()
    if args.config is not None:
        raw = read_json(args.config)
        names = {field.name for field in fields(TransportConfig)}
        config = TransportConfig(**{name: raw[name] for name in names if name in raw})
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = build_transport(args.route, config, seed=args.seed).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model), weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed)
    users = len(arrays["source"])
    for update in range(args.updates):
        index = torch.randint(users, (args.batch_size,), generator=generator)
        batch = {
            name: torch.from_numpy(values[index.numpy()]).to(device)
            for name, values in arrays.items()
        }
        losses = rectified_flow_loss(
            model,
            batch["source"],
            batch["target"],
            batch["history"],
            batch["history_mask"],
            batch["evidence"],
            batch["evidence_mask"],
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 100 == 0 or update + 1 == args.updates:
            print(json.dumps({"update": update + 1, **{k: float(v.detach()) for k, v in losses.items()}}))
    save_transport(args.output, model)
    print(json.dumps({"checkpoint": str(args.output), "route": args.route}))


if __name__ == "__main__":
    main()
