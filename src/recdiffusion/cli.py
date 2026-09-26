"""Command-line entry points for bandwidth calibration and metric evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import load_evaluation_npz, ragged_row
from .io import write_json
from .metrics import calibrate_chordal_bandwidth, macro_average, score_user


def calibrate(args: argparse.Namespace) -> None:
    items = np.load(args.items, allow_pickle=False)
    bandwidth = calibrate_chordal_bandwidth(items, pair_count=args.pairs, seed=args.seed)
    write_json(args.output, {"kernel": "chordal_gaussian_rbf", "bandwidth": bandwidth})


def evaluate(args: argparse.Namespace) -> None:
    data = load_evaluation_npz(args.input)
    records = []
    for index, user_id in enumerate(data["user_id"]):
        record = score_user(
            data["generated"][index],
            ragged_row(data["target_values"], data["target_offsets"], index),
            ragged_row(data["history_values"], data["history_offsets"], index),
            bandwidth=args.bandwidth,
            tau_clust=args.tau_clust,
            tau_hit=args.tau_hit,
        )
        records.append({"user_id": int(user_id), **record})
    write_json(args.output, {"per_user": records, "macro": macro_average(records)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recdiffusion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    bandwidth = subparsers.add_parser("calibrate-bandwidth")
    bandwidth.add_argument("--items", type=Path, required=True)
    bandwidth.add_argument("--output", type=Path, required=True)
    bandwidth.add_argument("--pairs", type=int, default=500_000)
    bandwidth.add_argument("--seed", type=int, default=2026)
    bandwidth.set_defaults(function=calibrate)
    metrics = subparsers.add_parser("evaluate")
    metrics.add_argument("--input", type=Path, required=True)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.add_argument("--bandwidth", type=float, required=True)
    metrics.add_argument("--tau-clust", type=float, default=0.6900425553321838)
    metrics.add_argument("--tau-hit", type=float, default=0.6900425553321838)
    metrics.set_defaults(function=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

