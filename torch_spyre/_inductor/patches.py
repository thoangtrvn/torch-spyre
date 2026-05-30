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

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import InputType
from torch._inductor.virtualized import V
from torch_spyre.constants import DEVICE_NAME
from typing import Callable, Optional


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
      - Enabling Spyre-specific data type handling (bfloat16 computation)
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
        # Enable native matmul (ops.dot + make_reduction("dot")) for Spyre.
        # Required for TritonKernel to set is_native_matmul=True on the kernel,
        # which gates the tl.dot codegen path (simd.py:440-449).
        "triton.native_matmul": True,
    }

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

    _orig_use_native_matmul = mm_common.use_native_matmul

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

    mm_common.use_native_matmul = _spyre_aware_use_native_matmul
    mm.use_native_matmul = _spyre_aware_use_native_matmul
    bmm.use_native_matmul = _spyre_aware_use_native_matmul

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
            joint_graph.pass_patterns[:] = origin_pass
