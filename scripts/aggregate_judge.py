#!/usr/bin/env python3
"""Aggregate already validated judge outputs; this script never calls a remote model."""
from __future__ import annotations

import argparse
from pathlib import Path

from recdiffusion.io import read_json, write_json
from recdiffusion.judge import aggregate_ivc_users


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    payload = read_json(args.input)
    result = aggregate_ivc_users(
        payload["users"],
        methods=args.methods,
        bootstrap_resamples=args.bootstrap,
        seed=args.seed,
    )
    write_json(args.output, result)


if __name__ == "__main__":
    main()
