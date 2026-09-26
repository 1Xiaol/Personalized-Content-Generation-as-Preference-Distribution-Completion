#!/usr/bin/env python3
"""Pack validated NPY arrays into one of the documented NPZ contracts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from recdiffusion.data import (
    load_evaluation_npz,
    load_generation_npz,
    load_source_calibration_npz,
    load_training_npz,
)


REQUIRED = {
    "transport": {"source", "target", "history", "history_mask", "evidence", "evidence_mask"},
    "source-calibration": {
        "source",
        "target",
        "target_valid",
        "history",
        "history_mask",
        "evidence",
        "evidence_mask",
    },
    "generation": {"user_id", "source", "history", "history_mask", "evidence", "evidence_mask"},
    "evaluation": {
        "user_id",
        "generated",
        "history_values",
        "history_offsets",
        "target_values",
        "target_offsets",
    },
}
VALIDATORS = {
    "transport": load_training_npz,
    "source-calibration": load_source_calibration_npz,
    "generation": load_generation_npz,
    "evaluation": load_evaluation_npz,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", choices=sorted(REQUIRED), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--array",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat once for every field in the selected schema.",
    )
    args = parser.parse_args()

    paths = {}
    for entry in args.array:
        name, separator, raw_path = entry.partition("=")
        if not separator or not name or name in paths:
            raise ValueError(f"invalid or duplicate --array entry: {entry}")
        paths[name] = Path(raw_path)
    if set(paths) != REQUIRED[args.schema]:
        missing = sorted(REQUIRED[args.schema] - set(paths))
        extra = sorted(set(paths) - REQUIRED[args.schema])
        raise ValueError(f"array fields differ; missing={missing}, extra={extra}")

    arrays = {}
    for name, path in paths.items():
        value = np.load(path, allow_pickle=False)
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{path} is not a single NPY array")
        arrays[name] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    VALIDATORS[args.schema](args.output)
    print(f"wrote validated {args.schema} data to {args.output}")


if __name__ == "__main__":
    main()

