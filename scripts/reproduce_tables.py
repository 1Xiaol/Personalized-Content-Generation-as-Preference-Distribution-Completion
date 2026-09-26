#!/usr/bin/env python3
"""Render a paper-ready metric table from evaluator outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from recdiffusion.io import read_json
from recdiffusion.metrics import METRICS


LABELS = {
    "history_nearest": "History-nearest",
    "target_best": "Target-best",
    "coverage_rad": "Coverage",
    "future_mode_coverage": "Future-mode coverage",
    "energy": "Energy",
    "mmd2_u": "MMD2-U",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True, help="LABEL=metrics.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for item in args.method:
        label, separator, path = item.partition("=")
        if not separator:
            raise ValueError("--method must use LABEL=PATH")
        rows.append((label, read_json(path)["macro"]))
    header = ["Method", *[LABELS[name] for name in METRICS]]
    lines = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * len(METRICS)]
    for label, values in rows:
        rendered = ["--" if values[name] is None else f"{values[name]:.6f}" for name in METRICS]
        lines.append("| " + " | ".join([label, *rendered]) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
