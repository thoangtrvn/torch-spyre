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

import torch
import torch_spyre.ops.fallbacks  # noqa: F401
from .fallbacks import _get_op_overloads
from ._compile import compile_once, dispatch_to_torch_compile
import torch_spyre._C as _C
import warnings


aten = torch.ops.aten


def maybe_wrap_dim(dim: int, ndims: int) -> int:
    if dim < 0:
        return dim + ndims
    return dim


def register_torch_compile_kernel(ops):
    for op in _get_op_overloads(ops):
        if "Tensor" not in str(op._schema):
            # there are some ops that do not take in Tensors
            # like aten.sum.int
            continue
        if "dtype" in op.name():
            # ops that change dtype are not supported yet
            continue
        compiled_kernel = compile_once(op, dynamic=False)(dispatch_to_torch_compile)
        torch.library.register_kernel(op.name(), ["spyre"])(compiled_kernel)


register_torch_compile_kernel(
    [
        aten.mm,
        aten.silu.out,
        aten.mish.out,
        aten.abs,
        aten.add,
        aten.bitwise_not,
        aten.logical_not,
        aten.bmm,
        aten.cat,
        aten.div,
        aten.exp,
        aten.floor,
        aten.log,
        aten.amax,  # max(dim) reduction — registers .out overload so direct aten.amax works
        aten.mean,
        aten.mul,
        aten.reciprocal,
        aten.neg,
        aten.relu,
        aten.rsqrt,
        aten.sigmoid,
        # aten._softmax is REGISTERED ELSEWHERE (#112): torch_spyre/ops/softmax.py
        # binds it to the spyre::softmax custom op, which allocates its own scratch
        # segment and launches [scratch, x, out]. Leaving it here too would register a
        # SECOND kernel for the same (aten::_softmax, spyre) key. Do not re-add it
        # without removing that binding first.
        aten.stack,
        aten.sum,
        aten.sqrt,
        aten.tanh,
        aten.sub,
        aten.addmm,
        aten.eq,
        aten.ge,
        aten.gt,
        aten.lt,
        aten.maximum,
        aten.pow,
        aten.linalg_vector_norm,
    ]
)


@torch.library.register_kernel("aten::fill_.Scalar", ["spyre"])  # type:ignore
def spyre__fill_scalar(
    self: torch.Tensor, other: int | float | bool | complex
) -> torch.Tensor:
    # Device-side fill via FillDMA (no CPU staging / H2D copy).
    if isinstance(other, complex):
        raise TypeError("spyre fill_ does not support complex fill values")
    _C.fill_tensor(self, float(other))
    return self


@torch.library.register_kernel("aten::full", ["spyre"])  # type:ignore
def spyre_full(
    size: list | tuple,
    fill_value: int | float | bool | complex,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    if isinstance(fill_value, complex):
        raise TypeError("spyre full does not support complex fill values")
    t = torch.empty(size, dtype=dtype, device=device)
    _C.fill_tensor(t, float(fill_value))
    return t


@torch.library.register_kernel("aten::ones", ["spyre"])  # type:ignore
def spyre_ones(
    size: list | tuple,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    t = torch.empty(size, dtype=dtype, device=device)
    _C.fill_tensor(t, 1.0)
    return t


@torch.library.register_kernel("aten::normal_", ["spyre"])  # type:ignore
def spyre__normal_(self, mean=0.0, std=1.0, *, generator=None):
    # Honest draw path: device RNG has no normal-distribution kernel at all (its
    # only witnessed artifact is the T=1 raw-state Threefry KAT). With the
    # device-RNG switch ON we REFUSE typed rather than silently offload; with it
    # OFF (the default) this stays the documented host-CPU offload.
    from torch_spyre.rng import SpyreRngError, device_rng_enabled

    if device_rng_enabled():
        raise SpyreRngError(
            "device RNG has no normal-distribution path: the only witnessed "
            "artifact is the T=1 raw-state Threefry KAT (uniform-derived). "
            "Turn off TORCH_SPYRE_DEVICE_RNG to use the documented host-CPU "
            "offload, or draw normals from a uniform stream on the host."
        )
    # "normal_" generates a random tensor, thus copying
    # "self" back from SPYRE to CPU is not needed.
    # cpu_tmp = self.to("cpu")

    # Create a new tensor on cpu itself to avoid unnecessary data copy.
    cpu_tmp = torch.empty_like(self, device="cpu", memory_format=torch.preserve_format)
    cpu_tmp.normal_(mean, std, generator=generator)
    self.copy_(cpu_tmp)
    return self


@torch.library.register_kernel("aten::zero_", ["spyre"])  # type:ignore
def spyre__zero_(self: torch.Tensor) -> torch.Tensor:
    """Zero out the tensor in-place using device-side FillDMA."""
    _C.fill_tensor(self, 0.0)
    return self


def _reject_per_call_generator(generator) -> None:
    """Device RNG has ONE per-device default generator; a per-call ``generator=``
    is not supported yet (multi-stream unbuilt). Refuse TYPED rather than
    silently draw from the wrong stream. The switch-off host path DOES honor
    ``generator=``, so a silent ignore on the device path would be a
    correctness trap. Module-level so it is unit-testable (the decorated
    ``spyre__uniform_`` name resolves to ``None`` on import)."""
    if generator is not None:
        from torch_spyre.rng import SpyreRngError

        raise SpyreRngError(
            "device RNG does not support a per-call generator= yet (the device "
            "path uses the single per-device default generator). Pass "
            "generator=None, or turn off TORCH_SPYRE_DEVICE_RNG for the "
            "host-CPU offload, which honors generator=."
        )


@torch.library.register_kernel("aten::uniform_", "spyre")  # type:ignore
def spyre__uniform_(self, from_=0.0, to=1.0, generator=None):
    # Honest draw path (see torch_spyre/rng.py). `torch.rand` lands here: it
    # allocates via the factory fallback, then fills with uniform_.
    #
    # Default (switch OFF): the documented host-CPU offload — draw on CPU, copy
    # to device. Reproducible only via the CPU stream; this is NOT device RNG.
    #
    # Switch ON (TORCH_SPYRE_DEVICE_RNG): the device route writes DEVICE-
    # generated uniforms straight into ``self`` — the silicon-witnessed T=1
    # UNIFORM egress (binary d0930bd5…), bound to the caller tensor (or a
    # (2,32) scratch + one device→device copy), with ZERO host round trip. The
    # egress writes the bare [0,1) fp32 grid, so only the identity map is a
    # device route; any other (from_, to) REFUSES TYPED (no on-device affine
    # kernel). It also refuses typed beyond one T=1 tile (numel > 64, T-DV-34),
    # a non-fp32 or non-contiguous caller, or an unregistered pattern. A caller
    # who opted into device RNG never gets a host draw.
    from torch_spyre.rng import device_rng_enabled

    if device_rng_enabled():
        from torch_spyre.rng import route_uniform_fill

        _reject_per_call_generator(generator)
        route_uniform_fill(self, from_, to)  # raises typed
        return self

    # Create a new tensor on cpu
    cpu_tmp = torch.empty_like(self, device="cpu", memory_format=torch.preserve_format)

    # Fill the CPU tensor with uniform random values
    cpu_tmp.uniform_(from_, to, generator=generator)

    # Copy the CPU tensor back to the spyre device
    self.copy_(cpu_tmp)

    return self


@torch.library.register_kernel("aten::_local_scalar_dense", "spyre")
def spyre__local_scalar_dense(self):
    return self.cpu().item()


@torch.library.register_kernel("aten::_copy_from", ["spyre"])
def spyre__copy_from(self, dst, non_blocking=False):
    if self.numel() == 0:
        return dst

    # Check if views of same data
    if (
        self.data_ptr() == dst.data_ptr()
        and self.storage_offset() == dst.storage_offset()
        and self.stride() == dst.stride()
        and self.size() == dst.size()
        and self.dtype == dst.dtype
        and self.is_conj() == dst.is_conj()
        and self.is_neg() == dst.is_neg()
    ):
        return dst

    if (self.device.type == "cpu" and dst.device.type == "spyre") or (
        self.device.type == "spyre" and dst.device.type == "cpu"
    ):
        _C.copy_tensor(self, dst, non_blocking)
        return dst
    elif self.device.type == "spyre" and self.device == dst.device:
        # FAIL LOUD on an on-device dtype conversion. A differently-typed dst on the
        # SAME spyre device means someone asked for an on-device dtype cast (e.g.
        # `.float()` / `.to(dtype)`, which lands here via the composite _to_copy →
        # _copy_from). copy_from_d2d is the identity (clone) kernel: it does a raw
        # element copy and CANNOT convert the per-element type. A spyre tensor is
        # stored in the device format (DL16 for 1p0), so a mismatched-dtype clone
        # reinterprets the bytes and returns GARBAGE, silently — the same
        # silent-wrong class as an unimplemented op returning uninitialized memory.
        # It must ERROR, not misread. A real on-device conversion needs a DEDICATED
        # conversion kernel (SFP ICVT / a codegen convert op). Until then, convert on
        # CPU: move first, then cast — `x.cpu().float()`, NOT `x.float()` on device.
        if self.dtype != dst.dtype:
            raise NotImplementedError(
                f"Spyre: on-device dtype conversion {self.dtype} -> {dst.dtype} "
                f"is NOT supported. The device→device copy is the identity (clone) "
                f"kernel (copy_from_d2d) — a raw element copy that CANNOT convert the "
                f"element type, so it would return garbage (the device stores DL16/"
                f"device-format). A real conversion needs a dedicated kernel (SFP "
                f"ICVT / codegen convert op), not yet implemented. Convert on CPU: "
                f"move first, then cast — `x.cpu().to({dst.dtype})`, not "
                f"`x.to({dst.dtype})` on device."
            )
        torch.ops.spyre.copy_from_d2d(self, dst)
        return dst
    else:
        if non_blocking:
            warnings.warn(
                f"non_blocking is set to {non_blocking}", UserWarning, stacklevel=2
            )

        torch.ops.aten._copy_from.default(self, dst, non_blocking)
        return dst


# INSERT_CODEGEN_HERE
