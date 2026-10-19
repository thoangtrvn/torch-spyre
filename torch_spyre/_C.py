# Copyright 2026 The Torch-Spyre Authors.
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

import enum

from torch_spyre._runtime import codegen_only_available, require_runtime

if not codegen_only_available():
    raise ImportError(
        "torch_spyre._C runtime extension is not built. Set "
        "TORCH_SPYRE_CODEGEN_ONLY=1 only for compiler/import/TTIR capture workflows, "
        "or install the full Spyre runtime build."
    )

try:
    from torch_spyre import _codegen_backend as _backend
except ImportError:
    _backend = None

__all__: list[str] = [
    "DataFormats",
    "SpyreTensorLayout",
    "copy_tensor",
    "current_device",
    "device_count",
    "empty_with_layout",
    "encode_constant",
    "get_device_dtype",
    "get_elem_in_stick",
    "get_spyre_tensor_layout",
    "launch_kernel_from_bytes",
    "set_device",
    "set_spyre_tensor_layout",
    "spyre_empty_with_layout",
]

if _backend is not None:
    DataFormats = _backend.DataFormats
    SpyreTensorLayout = _backend.SpyreTensorLayout
    get_device_dtype = _backend.get_device_dtype
    get_elem_in_stick = _backend.get_elem_in_stick
    empty_with_layout = _backend.empty_with_layout
    spyre_empty_with_layout = _backend.spyre_empty_with_layout
    get_spyre_tensor_layout = _backend.get_spyre_tensor_layout
    set_spyre_tensor_layout = _backend.set_spyre_tensor_layout
    copy_tensor = _backend.copy_tensor
    encode_constant = _backend.encode_constant
    device_count = _backend.device_count
    current_device = _backend.current_device
    set_device = _backend.set_device
else:

    class DataFormats(enum.IntEnum):
        SEN169_FP16 = 0
        IEEE_FP32 = 1
        INVALID = 2
        SEN143_FP8 = 3
        SEN152_FP8 = 4
        SEN153_FP9 = 5
        SENINT2 = 6
        SENINT4 = 7
        SENINT8 = 8
        SENINT16 = 9
        SENINT24 = 10
        IEEE_INT64 = 11
        IEEE_INT32 = 12
        SENUINT32 = 13
        SENUINT2 = 14
        IEEE_FP16 = 15
        BOOL = 16
        BFLOAT16 = 17
        SEN18F_FP24 = 18

        def elems_per_stick(self) -> int:
            if self in {
                DataFormats.SEN169_FP16,
                DataFormats.IEEE_FP16,
                DataFormats.BFLOAT16,
            }:
                return 64
            if self == DataFormats.IEEE_FP32:
                return 32
            if self == DataFormats.IEEE_INT32:
                return 32
            if self == DataFormats.IEEE_INT64:
                return 16
            if self == DataFormats.BOOL:
                return 128
            if self in {DataFormats.SEN143_FP8, DataFormats.SEN152_FP8}:
                return 128
            raise NotImplementedError(
                f"elements-per-stick is undefined for {self.name}"
            )

    import torch

    _TORCH_DTYPE_TO_DEVICE_DTYPE = {
        torch.float16: DataFormats.SEN169_FP16,
        torch.float32: DataFormats.IEEE_FP32,
        torch.bfloat16: DataFormats.BFLOAT16,
        torch.int64: DataFormats.IEEE_INT64,
        torch.int32: DataFormats.IEEE_INT32,
        torch.bool: DataFormats.BOOL,
    }

    if hasattr(torch, "float8_e4m3fn"):
        _TORCH_DTYPE_TO_DEVICE_DTYPE[torch.float8_e4m3fn] = DataFormats.SEN143_FP8
    if hasattr(torch, "float8_e5m2"):
        _TORCH_DTYPE_TO_DEVICE_DTYPE[torch.float8_e5m2] = DataFormats.SEN152_FP8

    def get_device_dtype(dtype: torch.dtype) -> DataFormats:
        try:
            return _TORCH_DTYPE_TO_DEVICE_DTYPE[dtype]
        except KeyError as e:
            raise NotImplementedError(
                f"unsupported Spyre dtype in codegen-only mode: {dtype}"
            ) from e

    def get_elem_in_stick(dtype: torch.dtype) -> int:
        return get_device_dtype(dtype).elems_per_stick()

    class SpyreTensorLayout:
        def __init__(self, *args):
            if len(args) == 2:
                host_size, dtype = args
                self.device_dtype = get_device_dtype(dtype)
                eps = self.device_dtype.elems_per_stick()
                if len(host_size) == 0:
                    self.device_size = []
                    self.stride_map = []
                elif len(host_size) == 1:
                    n = int(host_size[0])
                    self.device_size = [(n + eps - 1) // eps, eps]
                    self.stride_map = [eps, 1]
                else:
                    self.device_size = [int(v) for v in host_size]
                    self.device_size[-1] = (self.device_size[-1] + eps - 1) // eps
                    self.device_size.append(eps)
                    self.stride_map = []
                    stride = 1
                    for size in reversed(self.device_size):
                        self.stride_map.insert(0, stride)
                        stride *= int(size)
            elif len(args) == 4:
                host_size, host_strides, dtype, dim_order = args
                self.device_dtype = get_device_dtype(dtype)
                eps = self.device_dtype.elems_per_stick()
                self.device_size = [int(v) for v in host_size]
                if self.device_size:
                    self.device_size[-1] = (self.device_size[-1] + eps - 1) // eps
                    self.device_size.append(eps)
                self.stride_map = [int(v) for v in host_strides]
                self.dim_order = [int(v) for v in dim_order]
            elif len(args) == 3:
                device_size, stride_map, device_dtype = args
                self.device_size = [int(v) for v in device_size]
                self.stride_map = [int(v) for v in stride_map]
                self.device_dtype = DataFormats(int(device_dtype))
            else:
                raise TypeError("SpyreTensorLayout expects 2, 3, or 4 arguments")

        def __repr__(self) -> str:
            dtype_name = f"DataFormats.{self.device_dtype.name}"
            return (
                f"SpyreTensorLayout(device_size={self.device_size}, "
                f"stride_map ={self.stride_map}, device_dtype={dtype_name})"
            )

        __str__ = __repr__

        def __eq__(self, other) -> bool:
            if not isinstance(other, SpyreTensorLayout):
                return False
            return (
                self.device_size == other.device_size
                and self.stride_map == other.stride_map
                and self.device_dtype == other.device_dtype
            )

        def __hash__(self) -> int:
            return hash(
                (tuple(self.device_size), tuple(self.stride_map), self.device_dtype)
            )

        def elems_per_stick(self) -> int:
            return self.device_dtype.elems_per_stick()

    def copy_tensor(*args, **kwargs) -> None:
        require_runtime("torch_spyre._C.copy_tensor")

    def empty_with_layout(*args, **kwargs):
        require_runtime("torch_spyre._C.empty_with_layout")

    def spyre_empty_with_layout(*args, **kwargs):
        require_runtime("torch_spyre._C.spyre_empty_with_layout")

    def get_spyre_tensor_layout(*args, **kwargs) -> SpyreTensorLayout:
        require_runtime("torch_spyre._C.get_spyre_tensor_layout")

    def set_spyre_tensor_layout(*args, **kwargs) -> None:
        require_runtime("torch_spyre._C.set_spyre_tensor_layout")

    def device_count() -> int:
        return 0

    def current_device() -> int:
        return 0

    def set_device(idx: int) -> None:
        if idx != 0:
            raise RuntimeError("codegen-only Spyre backend exposes only fake device 0")


if _backend is not None:
    from torch_spyre._monkey_patch import _patch_tensor_for_spyre

    _patch_tensor_for_spyre()


def start_runtime() -> None:
    require_runtime("torch_spyre._C.start_runtime")


def free_runtime() -> None:
    require_runtime("torch_spyre._C.free_runtime")


def launch_kernel(*args, **kwargs) -> None:
    require_runtime("torch_spyre._C.launch_kernel")


def launch_kernel_from_bytes(*args, **kwargs) -> None:
    require_runtime("torch_spyre._C.launch_kernel_from_bytes")


def fill_tensor(*args, **kwargs):
    require_runtime("torch_spyre._C.fill_tensor")


def to_with_layout(*args, **kwargs):
    require_runtime("torch_spyre._C.to_with_layout")


if _backend is None:

    def encode_constant(*args, **kwargs) -> int:
        # no _codegen_backend extension: nothing can encode constants
        require_runtime("torch_spyre._C.encode_constant")


def get_downcast_warning() -> bool:
    require_runtime("torch_spyre._C.get_downcast_warning")


def set_downcast_warning(*args, **kwargs) -> None:
    require_runtime("torch_spyre._C.set_downcast_warning")


def get_stream_from_pool(*args, **kwargs):
    require_runtime("torch_spyre._C.get_stream_from_pool")


def current_stream(*args, **kwargs):
    require_runtime("torch_spyre._C.current_stream")


def default_stream(*args, **kwargs):
    require_runtime("torch_spyre._C.default_stream")


def set_current_stream(*args, **kwargs) -> None:
    require_runtime("torch_spyre._C.set_current_stream")


def synchronize(*args, **kwargs) -> None:
    require_runtime("torch_spyre._C.synchronize")


def synchronize_bounded(*args, **kwargs) -> None:
    require_runtime("torch_spyre._C.synchronize_bounded")
