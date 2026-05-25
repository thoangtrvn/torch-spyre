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


class SpyreEvent:
    """Wall-clock event for benchmarking compatibility.

    launch_kernel_from_bytes() is asynchronous — it submits a CB and
    returns control to the CPU immediately (same as CUDA kernel launches).
    Without device-side timestamp recording wired up yet, record()
    synchronizes the device first, then captures a wall-clock timestamp.
    This ensures elapsed_time() measures actual execution time, not just
    CB submission time.

    Future: GPU-style stream-ordered events
    ----------------------------------------
    CUDA events work differently from wall-clock timing:

    1. event.record(stream) inserts a lightweight marker into the GPU
       command stream. The CPU continues immediately — no blocking.
    2. The GPU hardware captures a device-side timestamp when it
       processes the marker in stream order (after all previously
       submitted work on that stream).
    3. event.elapsed_time(other) reads the GPU-captured timestamps
       from both events and returns the difference. This gives pure
       device execution time with zero CPU overhead.

    Benefits over wall-clock timing:
    - Non-blocking: record() doesn't stall the CPU pipeline
    - Precise: measures device time, not wall-clock time that includes
      CPU scheduling jitter and other host-side overhead
    - Stream-ordered: timestamps are captured in the device's execution
      order, not the host's submission order
    - Overlapping: host can submit more work while reading timing from
      previously completed events

    To implement for AIU:
    - The flex runtime / hardware needs an API to record a device-side
      timestamp at a given point in the CB stream
    - record() would submit a "record timestamp" CB command instead of
      calling synchronize()
    - elapsed_time() would query the recorded device timestamps
    - The C++ binding in torch_spyre._C would expose these operations
    """

    def __init__(self, *, enable_timing: bool = False):
        self._enable_timing = enable_timing
        self._time_ns: int = 0
        self._device = None

    def record(self, stream=None):
        if self._enable_timing:
            import time
            from torch_spyre.streams import synchronize
            synchronize()
            self._time_ns = time.monotonic_ns()

    def elapsed_time(self, end_event: "SpyreEvent") -> float:
        # Return milliseconds between two synchronized timestamps.
        if self._enable_timing and end_event._enable_timing:
            return (end_event._time_ns - self._time_ns) / 1_000_000.0
        return 0.0


class SpyreInterface(DeviceInterface):
    Event = SpyreEvent

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.spyre.is_available()  # type: ignore[attr-defined]

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def set_device(device: torch.types.Device) -> None:
        pass  # Spyre has a single device, no-op

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def exchange_device(device: int) -> int:
        return 0  # Spyre has a single device, previous is always 0

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        return 0

    @staticmethod
    def stream(stream: torch.Stream) -> Any:
        return stream.__enter__()

    @staticmethod
    def current_stream() -> torch.Stream:
        from torch_spyre.streams import current_stream
        return current_stream()

    @staticmethod
    def set_stream(stream: torch.Stream) -> None:
        from torch_spyre import _C
        _C.set_current_stream(stream._cdata if hasattr(stream, "_cdata") else stream)

    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int) -> None:
        pass  # Spyre is synchronous, no-op

    @staticmethod
    def get_raw_stream(device_idx: int) -> int:
        from torch_spyre.streams import current_stream
        s = current_stream(torch.device("spyre", device_idx))
        return s.id

    @staticmethod
    def synchronize(device: torch.types.Device = None) -> None:
        from torch_spyre.streams import synchronize
        synchronize(device)

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
