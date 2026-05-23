# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from torch._dynamo.device_interface import DeviceInterface
from typing import Any
from dataclasses import dataclass

# Recording the device properties in the main process but used in worker process.
caching_worker_device_properties: dict[str, Any] = {}
caching_worker_current_devices: dict[str, int] = {}

# Cached compute capability — detected once on first access.
_cached_compute_capability: str | None = None
_cached_device_properties: SpyreDeviceProperties | None = None


def _detect_compute_capability() -> str:
    """Detect the Sentient generation. Called once, result is cached.

    TODO: Query from C++ runtime (flex knows the hardware generation).
    Falls back to SENARCH env var, then "rcudd1a" default.
    """
    import os

    return os.environ.get("SENARCH", "rcudd1a")


def _detect_device_properties() -> SpyreDeviceProperties:
    """Build device properties from runtime configuration.

    TODO: Query from C++ runtime for hardware-detected values.
    """
    import os

    try:
        num_cores = int(os.environ.get("SENCORES", "32"))
    except ValueError:
        num_cores = 32

    return SpyreDeviceProperties(
        type="spyre",
        index=0,
        multi_processor_count=num_cores,
    )


@dataclass(frozen=True)
class SpyreDeviceProperties:
    type: str
    index: int
    multi_processor_count: int


class SpyreInterface(DeviceInterface):
    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.spyre.is_available()  # type: ignore[attr-defined]

    @staticmethod
    def exchange_device(device: int) -> int:
        return 0  # Spyre has a single device, previous is always 0

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        return 0

    @classmethod
    def get_device_properties(
        cls, device: torch.types.Device = None
    ) -> SpyreDeviceProperties:
        return cls.Worker.get_device_properties(device)

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> Any:
        """Return the Sentient generation identifier (e.g., "rcudd1a", "sen1p5").

        This is an architectural identifier that matches GPUTarget.arch in
        the Triton backend, not a version number like CUDA's "8.0". Inductor
        uses it for backend routing, not for heuristic tile-size tables.
        """
        global _cached_compute_capability
        if _cached_compute_capability is None:
            _cached_compute_capability = _detect_compute_capability()
        return _cached_compute_capability

    class Worker(DeviceInterface.Worker):
        @staticmethod
        def set_device(device: int):
            pass  # Spyre has a single device, no-op

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def exchange_device(device: int) -> int:
            return 0  # Spyre has a single device, previous is always 0

        @staticmethod
        def maybe_exchange_device(device: int) -> int:
            return 0

        @staticmethod
        def get_device_properties(device: torch.types.Device = None):
            global _cached_device_properties
            if _cached_device_properties is None:
                _cached_device_properties = _detect_device_properties()
            return _cached_device_properties
