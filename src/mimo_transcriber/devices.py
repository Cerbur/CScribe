from __future__ import annotations

import platform as platform_module
from dataclasses import dataclass
from typing import Literal

from mimo_transcriber.config import Device

SelectedDevice = Literal["cpu", "cuda", "mps"]
FallbackCategory = Literal[
    "not_built",
    "runtime_unavailable",
    "unsupported_operator",
    "out_of_memory",
    "preflight_failed",
    "full_run_failed",
]


@dataclass(frozen=True)
class DeviceCapabilities:
    cuda_available: bool
    mps_built: bool
    mps_available: bool
    platform: str
    machine: str


@dataclass(frozen=True)
class DeviceDecision:
    requested_device: Device
    selected_device: SelectedDevice
    mps_built: bool | None = None
    mps_available: bool | None = None
    preflight_elapsed_seconds: float | None = None
    fallback_category: FallbackCategory | None = None
    fallback_reason: str | None = None


def collect_device_capabilities() -> DeviceCapabilities:
    import torch

    return DeviceCapabilities(
        cuda_available=torch.cuda.is_available(),
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        platform=platform_module.system(),
        machine=platform_module.machine(),
    )
