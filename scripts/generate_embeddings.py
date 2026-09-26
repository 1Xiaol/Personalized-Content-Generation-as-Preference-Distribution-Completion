#!/usr/bin/env python3
"""Generate semantic samples from a trained transport and optional source calibration."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from recdiffusion.checkpoints import load_source_calibration, load_transport
from recdiffusion.data import load_generation_npz
from recdiffusion.geometry import spherical_heun


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--source-calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--heun-steps", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    arrays = load_generation_npz(args.input)
    device = torch.device(args.device)
    transport = load_transport(args.transport, device).eval()
    transport.requires_grad_(False)
    calibration = None
    if args.source_calibration is not None:
        calibration = load_source_calibration(args.source_calibration, device).eval()
        calibration.requires_grad_(False)

    output = []
    with torch.inference_mode():
        for begin in range(0, len(arrays["source"]), args.batch_size):
            end = min(begin + args.batch_size, len(arrays["source"]))
            batch = {
                name: torch.from_numpy(values[begin:end]).to(device)
                for name, values in arrays.items()
                if name != "user_id"
            }
            memory = transport.encode(
                batch["history"],
                batch["history_mask"],
                batch["evidence"],
                batch["evidence_mask"],
            )
            source = calibration(batch["source"], memory) if calibration is not None else batch["source"]
            generated = spherical_heun(
                lambda value, time: transport.velocity(value, time, memory),
                source,
                steps=args.heun_steps,
            )
            output.append(generated.cpu().numpy().astype(np.float32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        user_id=arrays["user_id"],
        generated=np.concatenate(output, axis=0),
    )
    print(f"wrote generated semantic samples to {args.output}")


if __name__ == "__main__":
    main()
