"""Portable checkpoint helpers for transport and source-calibration modules."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .source_calibration import ConditionalSourceCalibration, SourceCalibrationConfig
from .transport import TransportConfig, build_transport


def save_transport(path: str | Path, model: torch.nn.Module) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "transport-v1",
        "route": model.route,
        "config": asdict(model.config),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    torch.save(payload, destination)


def load_transport(path: str | Path, device: str | torch.device = "cpu") -> torch.nn.Module:
    payload: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema") != "transport-v1":
        raise ValueError("unsupported transport checkpoint")
    model = build_transport(payload["route"], TransportConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device)


def save_source_calibration(
    path: str | Path,
    calibration: ConditionalSourceCalibration,
    *,
    transport_route: str,
    transport_config: TransportConfig,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "conditional-source-calibration-v1",
        "transport_route": transport_route,
        "transport_config": asdict(transport_config),
        "config": asdict(calibration.config),
        "state_dict": {
            name: value.detach().cpu() for name, value in calibration.state_dict().items()
        },
    }
    torch.save(payload, destination)


def load_source_calibration(
    path: str | Path, device: str | torch.device = "cpu"
) -> ConditionalSourceCalibration:
    payload: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema") != "conditional-source-calibration-v1":
        raise ValueError("unsupported source-calibration checkpoint")
    calibration = ConditionalSourceCalibration(SourceCalibrationConfig(**payload["config"]))
    calibration.load_state_dict(payload["state_dict"], strict=True)
    return calibration.to(device)

