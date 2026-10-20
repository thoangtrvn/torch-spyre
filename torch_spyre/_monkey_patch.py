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


from torch_spyre.constants import DEVICE_NAME
from torch_spyre._runtime import codegen_only_available

from typing import Optional

import torch

from torch._dynamo.guards import GuardBuilder


def _patch_tensor_for_spyre():
    from torch_spyre._C import (
        SpyreTensorLayout,
        copy_tensor,
        empty_with_layout,
        get_spyre_tensor_layout,
        spyre_empty_with_layout,
    )

    if getattr(torch.Tensor, "_spyre_tensor_patched", False):
        return

    orig_repr = torch.Tensor.__repr__
    orig_to = torch.Tensor.to
    orig_empty = torch.empty
    orig_empty_strided = torch.empty_strided
    try:
        # The fallthroughs in spyre_to / spyre_empty / spyre_empty_strided
        # call the original pybind builtins as plain callables; Dynamo
        # cannot inline them and graph-breaks. Mark them callable-in-graph
        # so the fallthroughs trace as graph nodes (T-SR-59: stock HF
        # decoder factories and .to(dtype) calls must capture).
        from torch import _dynamo as _torch_dynamo

        _torch_dynamo.allow_in_graph(orig_to)
        _torch_dynamo.allow_in_graph(orig_empty)
        _torch_dynamo.allow_in_graph(orig_empty_strided)
    except Exception:
        pass

    def spyre_aware_repr(self):
        dev = getattr(self, "device", None)
        if dev is not None and dev.type == DEVICE_NAME:
            try:
                s = orig_repr(self.to("cpu"))
            except Exception:
                # Fallback if .to("cpu") fails for some weird reason
                return (
                    f"SpyreTensor(shape={tuple(self.shape)}, "
                    f"dtype={self.dtype}, device={self.device})"
                )
            if "device=" in s:
                return s.replace("device='cpu'", f"device='{self.device}'")
            if s.endswith(")"):
                s = s[:-1] + f", device='{self.device}')"
            else:
                # Odd case: just append device info
                s = s + f" (device='{self.device}')"
            return s

        # Non-spyre tensors use normal behavior
        return orig_repr(self)

    def device_tensor_layout(self: torch.Tensor) -> Optional[SpyreTensorLayout]:
        if self.device is not None and self.device.type == DEVICE_NAME:
            if isinstance(self, torch._subclasses.FakeTensor):
                return None  # catch FakeTensor BEFORE calling device_tensor_layout()
            return get_spyre_tensor_layout(self)
        else:
            return None

    def spyre_to(self, *args, device_layout=None, **kwargs):
        if torch.compiler.is_compiling() and device_layout is None:
            # Dynamo-transparent: the spyre-layout copy is an eager
            # codegen-only concern; under tracing, delegate to the original
            # allow_in_graph'ed builtin so the graph gets a native aten::to
            # node instead of graph-breaking on the pybind
            # SpyreTensorLayout constructor (T-SR-59: StaticCache's
            # cumulative_length.to(device) must capture).
            return orig_to(self, *args, **kwargs)
        requested_device = kwargs.get("device")
        if requested_device is None and args:
            requested_device = args[0]
        if codegen_only_available() and requested_device is not None:
            # Only str/int/torch.device are torch.device() constructors; a
            # positional dtype or Tensor argument (e.g. x.to(torch.float32))
            # must never reach torch.device(); in eager the raised TypeError
            # is caught below, but under Dynamo inlining the wrapped
            # InternalTorchDynamoError escapes the except clause and kills
            # graph capture (T-SR-59).
            if isinstance(requested_device, (str, int, torch.device)):
                try:
                    requested_device = torch.device(requested_device)
                except (TypeError, RuntimeError):
                    requested_device = None
            else:
                requested_device = None
        else:
            requested_device = None

        # Check if copy kwarg is explicitly set
        copy = kwargs.get("copy")

        # Determine dtype from various possible sources
        dtype = None
        if len(args) > 0:
            # If args[0] is a dtype instance, use it
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            # If args[0] is a Tensor, use its dtype
            elif isinstance(args[0], torch.Tensor):
                dtype = args[0].dtype

        # Check for dtype in kwargs
        if dtype is None and "dtype" in kwargs:
            dtype = kwargs["dtype"]

        # Check for tensor kwarg
        if dtype is None and "tensor" in kwargs:
            tensor_arg = kwargs["tensor"]
            if isinstance(tensor_arg, torch.Tensor):
                dtype = tensor_arg.dtype

        # Fall back to self.dtype if no dtype was specified
        if dtype is None:
            dtype = self.dtype

        if device_layout is None:
            if requested_device is not None and requested_device.type == DEVICE_NAME:
                device_layout = SpyreTensorLayout(list(self.size()), dtype)
            if device_layout is None:
                return orig_to(self, *args, **kwargs)

        fake = _maybe_fake_spyre_tensor(
            self.size(),
            self.stride(),
            dtype,
            requested_device or torch.device(DEVICE_NAME),
            requires_grad=self.requires_grad,
        )
        if fake is not None:
            return fake

        dst = spyre_empty_with_layout(self.size(), self.stride(), dtype, device_layout)

        if self.device.type == "cpu":
            copy_tensor(self, dst, non_blocking=False)
            return dst
        else:  # device to device copy
            # If device_layout is the same as self and copy is not True, return self
            current_layout = device_tensor_layout(self)
            if not copy and current_layout is not None and current_layout == device_layout:
                return self
            else:
                return torch.ops.spyre.copy_from_d2d(self, dst)

    def _codegen_requested_spyre_device(device):
        if not codegen_only_available() or device is None:
            return None
        # Same Dynamo-inlining hazard as spyre_to: a dtype/Tensor argument
        # must not reach torch.device(); the eager TypeError is caught here
        # but becomes an uncaught InternalTorchDynamoError under tracing.
        if not isinstance(device, (str, int, torch.device)):
            return None
        try:
            requested_device = torch.device(device)
        except (TypeError, RuntimeError):
            return None
        if requested_device.type != DEVICE_NAME:
            return None
        return requested_device

    def _normalize_empty_size(args):
        if not args:
            raise TypeError("torch.empty requires a size argument")
        if len(args) == 1:
            size = args[0]
            if isinstance(size, int):
                return (size,)
            return tuple(size)
        return tuple(args)

    def _current_fake_mode():
        from torch._C import _TorchDispatchModeKey

        return torch._C._get_dispatch_mode(_TorchDispatchModeKey.FAKE)

    def _maybe_fake_spyre_tensor(size, stride, dtype, requested_device, requires_grad=False):
        fake_mode = _current_fake_mode()
        if fake_mode is None:
            return None
        from torch._subclasses.fake_tensor import FakeTensor

        dtype = dtype or torch.get_default_dtype()
        meta = orig_empty_strided(tuple(size), tuple(stride), dtype=dtype, device="meta")
        return FakeTensor(
            fake_mode,
            meta,
            device=requested_device,
            requires_grad=requires_grad,
        )

    def spyre_empty(
        *args,
        device_layout=None,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
        memory_format=torch.contiguous_format,
    ):
        requested_device = _codegen_requested_spyre_device(device)
        if torch.compiler.is_compiling():
            # Dynamo-transparent: the spyre-layout allocation is an eager
            # codegen-only concern; under tracing, delegate to the original
            # builtin (allow_in_graph'ed at patch time) so the graph gets a
            # native aten::empty node on the requested device instead of
            # graph-breaking on the pybind SpyreTensorLayout constructor
            # (T-SR-59: StaticCache's torch.zeros must capture).
            return orig_empty(
                *args,
                out=out,
                dtype=dtype,
                layout=layout,
                device=device,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
                memory_format=memory_format,
            )
        if device_layout is None and requested_device is not None:
            size = _normalize_empty_size(args)
            device_layout = SpyreTensorLayout(list(size), dtype or torch.get_default_dtype())

        if device_layout is None:  # use original implementation if no layout is provided
            return orig_empty(
                *args,
                out=out,
                dtype=dtype,
                layout=layout,
                device=device,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
                memory_format=memory_format,
            )

        size = _normalize_empty_size(args)
        fake = _maybe_fake_spyre_tensor(
            size,
            orig_empty(size, dtype=dtype, device="meta", memory_format=memory_format).stride(),
            dtype,
            requested_device or device,
            requires_grad=requires_grad,
        )
        if fake is not None:
            return fake

        # layout_opt is omitted; c10::Layout has no pybind11 type caster,
        # so py_empty_with_layout drops that parameter and always uses
        # the default (Strided).
        return empty_with_layout(
            size,
            device_layout,
            dtype,
            requested_device or device,
            pin_memory,
            memory_format,
        )

    def spyre_empty_strided(
        size,
        stride,
        *,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
    ):
        requested_device = _codegen_requested_spyre_device(device)
        if requested_device is None or torch.compiler.is_compiling():
            # Dynamo-transparent (see spyre_empty): delegate to the original
            # allow_in_graph'ed builtin under tracing.
            return orig_empty_strided(
                size,
                stride,
                dtype=dtype,
                layout=layout,
                device=device,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
            )
        resolved_dtype = dtype or torch.get_default_dtype()
        fake = _maybe_fake_spyre_tensor(
            size,
            stride,
            resolved_dtype,
            requested_device,
            requires_grad=requires_grad,
        )
        if fake is not None:
            return fake
        device_layout = SpyreTensorLayout(list(size), list(stride), resolved_dtype, list(range(len(size))))
        return spyre_empty_with_layout(size, stride, resolved_dtype, device_layout)

    torch.Tensor.__repr__ = spyre_aware_repr
    torch.Tensor.device_tensor_layout = device_tensor_layout
    torch.Tensor._spyre_tensor_patched = True
    torch.Tensor.to = spyre_to
    torch.empty = spyre_empty
    torch.empty_strided = spyre_empty_strided

    # ── SpyreTensorLayout Guard Extension ────────────
    # Extends TENSOR_MATCH to guard on SpyreTensorLayout
    # preventing wrong compiled graph reuse when layout
    # changes.
    # ─────────────────────────────────────────────────

    _original_TENSOR_MATCH = GuardBuilder.TENSOR_MATCH

    def _spyre_TENSOR_MATCH(self, guard, value=None):
        # run original TENSOR_MATCH
        _original_TENSOR_MATCH(self, guard, value=value)
        # get tensor value
        if value is None:
            value = self.get(guard)
        ## dereference WeakRef if needed
        if isinstance(value, torch.utils.weak.TensorWeakRef):
            value = value()

        if value is None:
            return

        # not a Spyre tensor → skip
        if value.device.type != DEVICE_NAME:
            return

        # get layout safely
        expected_layout = value.device_tensor_layout()
        if expected_layout is None:
            return

        # add lambda guard on tensor's child manager
        # same node as TENSOR_MATCH!
        tensor_guard_manager = self.get_guard_manager(guard)
        tensor_guard_manager.add_lambda_guard(
            lambda x: (
                x.device.type != DEVICE_NAME
                or x.device_tensor_layout() == expected_layout
            ),
            [f"SpyreTensorLayout({guard.name}) == {expected_layout}"],
            guard.user_stack,
        )

    GuardBuilder.TENSOR_MATCH = _spyre_TENSOR_MATCH
