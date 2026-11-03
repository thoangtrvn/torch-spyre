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

"""Bit-preserving (raw-word) host<->device copies that bypass DCI/ConvertData.

The normal tensor copy path (``Tensor.copy_`` / ``torch_spyre._C.copy_tensor``)
always attaches a :class:`DataConversionInfo` and therefore routes through
deeptools' ``ConvertData`` on the device. ``ConvertData`` has no same-format
arm for several dtypes — most notably ``(SENINT16, SENINT16)`` (torch.int16
maps to SENINT16 in both directions) — so both H2D prefill and D2H readback of
raw-int16 escape tensors throw ``DtException`` at
``sen_data_convert.cpp:2505`` on-box. The same applies to the wider
same-format class (SENINT24/SENINT2/SEN152/SEN153).

The functions here issue the DMA with ``dci=nullptr`` — the same raw-word
mechanism the init-packet program copy uses — so ``ConvertData`` is never
involved. They are the sanctioned route around that dispatch gap.

This is an explicitly-named, opt-in escape hatch:

- it is **bit-preserving only** — no format conversion, ever. A dtype or
  layout mismatch raises instead of converting;
- the normal (dci-attaching) copy path is unchanged and remains the default
  for everything else;
- the host tensor must be a dense contiguous **allocation-sized mirror** of
  the stick-padded device allocation (the transfer is the full device
  allocation, not ``numel`` bytes);
- intended for harness/KAT/probe code that needs raw words on the device
  (DL16 bit-exact readback discipline: raw-int16 words in, raw-int16 words
  out — see the spyre-knowledgebase "word-domain soundness" law).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# torch_spyre registers "spyre" as a device alias, so tensors allocated on
# the device report device.type "spyre"; the underlying torch backend key is
# "privateuse1" and some torch versions surface that spelling instead. Both
# direction guards must accept either — box-witnessed 2026-11-03 that a
# single-literal "privateuse1" comparison rejects every real spyre tensor
# (D2H always refuses) and lets spyre input slip into the H2D wrapper.
_SPYRE_DEVICE_TYPES = frozenset({"spyre", "privateuse1"})


def _raw_copy_entry():
    """Resolve ``torch_spyre._C.raw_copy_tensor`` or refuse loud."""
    from torch_spyre._runtime import codegen_only_available, require_runtime

    if codegen_only_available():
        # Typed refusal before any allocation is attempted: codegen-only
        # processes have no DMA surface at all.
        require_runtime("torch_spyre._C.raw_copy_tensor")
    try:
        from torch_spyre import _C
    except ImportError as e:  # runtime extension not built in this install
        raise NotImplementedError(
            "torch_spyre.raw_copy_* requires the full Spyre runtime build "
            "(torch_spyre._C); this process cannot import the runtime "
            "extension and cannot issue device DMAs."
        ) from e
    fn = getattr(_C, "raw_copy_tensor", None)
    if fn is None:
        raise NotImplementedError(
            "torch_spyre._C.raw_copy_tensor is not exposed by this build of "
            "the runtime extension; rebuild torch_spyre._C from a source "
            "tree that includes spyre::spyre_raw_copy (csrc/spyre_mem.cpp, "
            "bound in csrc/module.cpp)."
        )
    return fn


def raw_copy_to_device(
    tensor: torch.Tensor,
    device: torch.device | str | int | None = None,
    *,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Bit-preserving copy of ``tensor``'s raw words onto the spyre device.

    Allocates a spyre tensor with the same shape and dtype and DMAs the host
    storage as raw bytes with no ``DataConversionInfo`` (bypassing
    ``ConvertData`` — the sanctioned route around the missing
    ``(SENINT16, SENINT16)`` arm, DtException ``sen_data_convert.cpp:2505``).

    The host tensor must be contiguous and its storage must be at least as
    large as the stick-padded device allocation of the returned tensor
    (allocations are padded to whole 128B sticks). For a plain contiguous
    tensor created at the final shape this holds automatically when numel is
    a multiple of the dtype's elements-per-stick; otherwise size the host
    storage up to the stick boundary (e.g. allocate from the device layout).

    This never converts values. Use the normal ``tensor.to('spyre')`` /
    ``copy_`` path for anything that needs conversion.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"raw_copy_to_device expects a torch.Tensor, got {type(tensor)!r}"
        )
    if tensor.device.type in _SPYRE_DEVICE_TYPES:
        raise TypeError(
            "raw_copy_to_device expects a CPU tensor (input is already on "
            "the spyre device); use raw_copy_from_device for D2H"
        )
    if device is None:
        device = "spyre"
    raw_copy = _raw_copy_entry()
    dev_tensor = torch.empty(
        tensor.shape, dtype=tensor.dtype, device=torch.device(device)
    )
    return raw_copy(tensor, dev_tensor, non_blocking=non_blocking)


def raw_copy_from_device(
    tensor: torch.Tensor, *, non_blocking: bool = False
) -> torch.Tensor:
    """Bit-preserving copy of a spyre tensor's raw words back to the host.

    Allocates a CPU tensor whose storage mirrors the stick-padded device
    allocation and DMAs it as raw bytes with no ``DataConversionInfo``
    (bypassing ``ConvertData`` — the sanctioned route around the missing
    ``(SENINT16, SENINT16)`` arm for D2H readback).

    The device tensor must be contiguous (allocation-sized mirror semantics:
    the whole physical allocation is transferred, including stick padding).
    This never converts values.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"raw_copy_from_device expects a torch.Tensor, got {type(tensor)!r}"
        )
    if tensor.device.type not in _SPYRE_DEVICE_TYPES:
        raise TypeError(
            "raw_copy_from_device expects a spyre tensor (input is on CPU); "
            "use raw_copy_to_device for H2D"
        )
    raw_copy = _raw_copy_entry()
    # Size the host storage to the physical (stick-padded) allocation so the
    # allocation-sized raw DMA stays in bounds. The logical view onto it keeps
    # the tensor's own shape. _raw_copy_entry() succeeded, so torch_spyre._C
    # is importable and exposes get_spyre_tensor_layout alongside
    # raw_copy_tensor.
    from torch_spyre import _C

    dev_size = _C.get_spyre_tensor_layout(tensor).device_size
    pad_numel = 1
    for s in dev_size:
        pad_numel *= int(s)
    host = torch.empty(pad_numel, dtype=tensor.dtype, device="cpu")
    host = host[: tensor.numel()].view(tensor.shape)
    return raw_copy(tensor, host, non_blocking=non_blocking)


__all__ = ["raw_copy_to_device", "raw_copy_from_device"]
