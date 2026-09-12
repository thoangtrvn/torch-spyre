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

from contextlib import contextmanager

import logging
import os
from typing import Callable, Optional

import sympy
import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import InputType
from torch._inductor.virtualized import V
from torch_spyre.constants import DEVICE_NAME


_DESCRIPTOR_POINTWISE_ENV = "TORCH_SPYRE_EXPERIMENTAL_DESCRIPTOR_POINTWISE"
_DESCRIPTOR_POINTWISE_SHAPE = (512, 128)


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _static_int(dim) -> int | None:
    try:
        return int(dim)
    except (TypeError, ValueError):
        pass
    try:
        sizevars = V.graph.sizevars
    except AttributeError:
        return None
    for attr in ("statically_known_int", "guard_int", "size_hint"):
        helper = getattr(sizevars, attr, None)
        if helper is None:
            continue
        try:
            return int(helper(dim))
        except (TypeError, ValueError, AttributeError, RuntimeError):
            pass
    return None


def _spyre_descriptor_shape_allowed(buffer_name: str | None) -> bool:
    if buffer_name is None:
        return False
    try:
        buf = V.graph.try_get_buffer(buffer_name)
    except (AttributeError, KeyError, RuntimeError):
        return False
    if buf is None or not hasattr(buf, "get_size"):
        return False
    try:
        shape = tuple(_static_int(dim) for dim in buf.get_size())
    except (TypeError, ValueError, AttributeError):
        return False
    return shape == _DESCRIPTOR_POINTWISE_SHAPE


@contextmanager
def spyre_data_types():
    saved = torch._prims_common._computation_dtype_map
    torch._prims_common._computation_dtype_map = {
        torch.bfloat16: torch.bfloat16,
        torch.float16: torch.float16,
        torch.complex32: torch.complex32,
    }
    try:
        yield
    finally:
        torch._prims_common._computation_dtype_map = saved


@contextmanager
def enable_spyre_context(
    example_inputs: list[InputType],
    decomps: Optional[dict[torch._ops.OperatorBase, Callable]] = None,
):
    """
    Context manager that sets up the Spyre compilation environment.

    This CM configures PyTorch Inductor to compile graphs for the Spyre device by:
      - Enabling Spyre-specific data type handling without implicit fp32 upcasts
      - Activating Spyre lowerings and decompositions
      - Disabling incompatible optimizations (reduction splitting, permute fusion)

    Args:
        example_inputs: List of example inputs to the graph being compiled. Used to
            set real inputs in the virtualized context for shape inference and
            optimization decisions.
        decomps: Decomposition table to be populated with Spyre-specific
            decompositions. Maps operator overloads to their decomposition implementations.
            This is typically a clone of PyTorch Inductor's global decomposition registry.
    """

    if decomps is None:
        decomps = torch._inductor.decomposition.decompositions

    from torch_spyre._inductor.lowering import enable_spyre_lowerings

    # Ensure decorators run (custom ops/decomp/lowerings modules)
    import torch_spyre._inductor.customops  # noqa: F401
    from torch_spyre._inductor.decompositions import (
        enable_spyre_decompositions,
    )

    import torch_spyre._inductor.lowering  # noqa: F401

    _descriptor_pointwise_enabled = _env_flag_enabled(_DESCRIPTOR_POINTWISE_ENV)

    # Inductor config tweaks (saved/restored)
    new_config = {
        "split_reductions": False,
        "benchmark_harness": False,
        # Adding this configuration in so as to avoid the optimization of turning small matmuls into non-matmuls
        # found here: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ir.py#L1580
        "unroll_reductions_threshold": 1,
        # Disable fusing of mm + permute/transpose for now.
        "permute_fusion": False,
        "allow_buffer_reuse": False,  # For now, as buffer reuse does not consider stride_map.
        # Prefer Triton 3.7+ TensorDescriptor-backed TTIR for regular tensor
        # accesses. This is a frontend evidence knob only: Sentient still must
        # prove stick layout, packet contracts, launch ABI, and HW correctness.
        "triton.use_tensor_descriptor": True,
        # Let Inductor assume aligned inputs so descriptor/TMA-style paths are
        # not blocked by conservative alignment uncertainty. This does not prove
        # 128B Sentient stick alignment; backend checks still own that contract.
        "assume_aligned_inputs": True,
        # Enable native matmul (ops.dot + make_reduction("dot")) for Spyre.
        # Required for TritonKernel to set is_native_matmul=True on the kernel,
        # which gates the tl.dot codegen path (simd.py:440-449).
        "triton.native_matmul": True,
        # AIU 1.0 uses DL16 as its native 16-bit compute representation. Do not
        # silently widen f16/bf16 pointwise loads to fp32 and truncate on store;
        # dtype changes must come from explicit PyTorch conversion operations.
        "triton.codegen_upcast_to_fp32": False,
    }

    # TensorDescriptorOptions is mutually exclusive with the older block-ptr
    # path in TritonKernel.indexing(); disable block pointers only for the
    # opt-in diagnostic capture route so regular Spyre compilation is unchanged.
    if _descriptor_pointwise_enabled:
        new_config["triton.use_block_ptr"] = False

    # PyTorch versions differ on this option. Enable it when available so
    # pointwise kernels retain a tiled grid instead of being flattened into a
    # 1D XBLOCK form that destroys useful descriptor/layout evidence.
    if hasattr(torch._inductor.config.triton, "tile_pointwise"):
        new_config["triton.tile_pointwise"] = True

    from torch._inductor.fx_passes import joint_graph

    origin_pass = list(joint_graph.pass_patterns)
    # disable mul_softmax_pattern and div_softmax_pattern for now
    joint_graph.pass_patterns.pop()

    # Monkey-patch use_native_matmul to accept Spyre devices.
    # The native matmul path (ops.dot + make_reduction("dot")) produces
    # the correct IR for TritonScheduling to emit tt.dot. Without this
    # patch, use_native_matmul() rejects "spyre" devices because it only
    # checks device_type in ("cuda", "xpu").
    #
    # We must patch BOTH mm_common.use_native_matmul AND mm.use_native_matmul
    # because mm.py imports the function at module load time:
    #   from .mm_common import use_native_matmul
    # This creates a local binding that is unaffected by patching the
    # mm_common module attribute. tuned_mm references the local name,
    # so only patching mm_common is invisible to it.
    import torch._inductor.kernel.mm_common as mm_common
    import torch._inductor.kernel.mm as mm
    import torch._inductor.kernel.bmm as bmm
    import torch._inductor.graph as inductor_graph
    import torch._inductor.utils as inductor_utils

    _orig_use_native_matmul = mm_common.use_native_matmul
    _orig_should_assume_input_aligned = inductor_utils.should_assume_input_aligned

    _log = logging.getLogger("torch_spyre.patches")

    def _spyre_aware_use_native_matmul(mat1, mat2):
        _dev = mat1.get_device().type
        if _dev == DEVICE_NAME:
            # tl.dot requires all of M, K, N > 1. When M=1 (vector-matrix
            # multiply), the scheduler can't create the 3D range trees
            # (batch+M+N) that tl.dot needs, causing an assertion failure.
            # Same for K=1 or N=1 — they degenerate to pointwise ops.
            m, k, n = mat1.get_size()[-2], mat1.get_size()[-1], mat2.get_size()[-1]
            m_le1 = V.graph.sizevars.statically_known_leq(m, 1)
            k_le1 = V.graph.sizevars.statically_known_leq(k, 1)
            n_le1 = V.graph.sizevars.statically_known_leq(n, 1)
            _log.debug(
                "use_native_matmul: dev=%s m=%s k=%s n=%s m_le1=%s k_le1=%s n_le1=%s",
                _dev, m, k, n, m_le1, k_le1, n_le1,
            )
            if m_le1 or k_le1 or n_le1:
                return False
            return True
        result = _orig_use_native_matmul(mat1, mat2)
        _log.debug("use_native_matmul: dev=%s -> orig returned %s", _dev, result)
        return result

    def _spyre_aware_should_assume_input_aligned(example_input):
        if getattr(getattr(example_input, "device", None), "type", None) == DEVICE_NAME:
            return torch._inductor.config.assume_aligned_inputs
        return _orig_should_assume_input_aligned(example_input)

    mm_common.use_native_matmul = _spyre_aware_use_native_matmul
    mm.use_native_matmul = _spyre_aware_use_native_matmul
    bmm.use_native_matmul = _spyre_aware_use_native_matmul
    inductor_utils.should_assume_input_aligned = _spyre_aware_should_assume_input_aligned
    inductor_graph.should_assume_input_aligned = _spyre_aware_should_assume_input_aligned

    import torch._inductor.codegen.triton as triton_codegen
    from torch._inductor.codegen.triton import is_unaligned_buffer_name

    _orig_tma_compatibility_checker_cls = (
        triton_codegen.TritonKernel.tma_compatibility_checker_cls
    )

    class SpyreTMACompatibilityChecker(_orig_tma_compatibility_checker_cls):
        """Diagnostic-only TensorDescriptor eligibility for Spyre pointwise capture."""

        def _is_spyre_graph(self):
            return V.graph.get_current_device_or_throw().type == DEVICE_NAME

        def can_use_tma(self):
            if not self._is_spyre_graph():
                return super().can_use_tma()
            if not _descriptor_pointwise_enabled:
                _log.debug(
                    "Spyre TensorDescriptor path disabled; set %s=1 to enable diagnostics",
                    _DESCRIPTOR_POINTWISE_ENV,
                )
                return False
            if not torch._inductor.config.triton.use_tensor_descriptor:
                _log.debug("Spyre TensorDescriptor path requires triton.use_tensor_descriptor")
                return False
            if not torch._inductor.config.assume_aligned_inputs:
                _log.debug("Spyre TensorDescriptor path requires assume_aligned_inputs")
                return False
            if self.for_store and self.kernel.no_x_dim:
                _log.debug("Spyre TensorDescriptor path rejects stores with no_x_dim")
                return False
            if not _spyre_descriptor_shape_allowed(self.buffer_name):
                _log.debug(
                    "Spyre TensorDescriptor path only supports diagnostic shape %s; buffer=%s",
                    _DESCRIPTOR_POINTWISE_SHAPE,
                    self.buffer_name,
                )
                return False
            _log.debug(
                "Spyre TensorDescriptor diagnostic path enabled for buffer=%s store=%s dtype=%s",
                self.buffer_name,
                self.for_store,
                self.dtype,
            )
            return True

        def are_block_parameters_compatible(
            self,
            block_params,
            constant_offset: sympy.Expr | int = 0,
        ):
            if not self._is_spyre_graph():
                return super().are_block_parameters_compatible(
                    block_params,
                    constant_offset=constant_offset,
                )
            if not _descriptor_pointwise_enabled:
                return super().are_block_parameters_compatible(
                    block_params,
                    constant_offset=constant_offset,
                )
            def _resolve_expr(expr):
                value = _static_int(expr)
                if value is not None:
                    return value
                if self.kernel.fixed_config and isinstance(expr, sympy.Expr):
                    substitutions = {
                        symbol: self.kernel.fixed_config[str(symbol)]
                        for symbol in expr.free_symbols
                        if str(symbol) in self.kernel.fixed_config
                    }
                    if substitutions:
                        return _static_int(expr.subs(substitutions))
                return None

            constant_offset_value = _resolve_expr(constant_offset)
            if constant_offset_value is None:
                return False
            if self.buffer_name is not None and is_unaligned_buffer_name(self.buffer_name):
                return False

            innermost_block = _resolve_expr(block_params.block_shape[-1])
            return (
                _static_int(block_params.strides[-1]) == 1
                and all(
                    ((_static_int(stride) or -1) * self.dtype.itemsize) % 16 == 0
                    for stride in block_params.strides[:-1]
                )
                and (constant_offset_value * self.dtype.itemsize) % 16 == 0
                and innermost_block is not None
                and innermost_block * self.dtype.itemsize >= 16
            )

        def can_lift(self):
            if self._is_spyre_graph() and _descriptor_pointwise_enabled:
                return True
            return super().can_lift()

    triton_codegen.TritonKernel.tma_compatibility_checker_cls = (
        SpyreTMACompatibilityChecker
    )

    # Verify patches applied correctly
    _log.debug("mm.use_native_matmul = %s", mm.use_native_matmul)
    _log.debug("mm_common.use_native_matmul = %s", mm_common.use_native_matmul)
    _log.debug("bmm.use_native_matmul = %s", bmm.use_native_matmul)

    with (
        spyre_data_types(),
        enable_spyre_lowerings(triton_path=True),
        enable_spyre_decompositions(decomps=decomps, triton_path=True) as spyre_context_decompositions,
        V.set_real_inputs(example_inputs),
        torch._inductor.config.patch(new_config),
    ):
        _log.debug("lowering[aten.mm.default] = %s", torch._inductor.lowering.lowerings.get(torch.ops.aten.mm.default))
        _log.debug("lowering[aten.addmm.default] = %s", torch._inductor.lowering.lowerings.get(torch.ops.aten.addmm.default))
        _log.debug("decomp[aten.mm.default] = %s", decomps.get(torch.ops.aten.mm.default))
        _log.debug("decomp[aten.addmm.default] = %s", decomps.get(torch.ops.aten.addmm.default))
        _log.debug("decomp[aten.linear.default] = %s", decomps.get(torch.ops.aten.linear.default))
        _log.debug("config.triton.native_matmul = %s", torch._inductor.config.triton.native_matmul)

        try:
            yield spyre_context_decompositions
        finally:
            mm_common.use_native_matmul = _orig_use_native_matmul
            mm.use_native_matmul = _orig_use_native_matmul
            bmm.use_native_matmul = _orig_use_native_matmul
            inductor_utils.should_assume_input_aligned = _orig_should_assume_input_aligned
            inductor_graph.should_assume_input_aligned = _orig_should_assume_input_aligned
            triton_codegen.TritonKernel.tma_compatibility_checker_cls = (
                _orig_tma_compatibility_checker_cls
            )
            joint_graph.pass_patterns[:] = origin_pass
