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

from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.triton import FixedTritonConfig
from torch._inductor.scheduler import BaseSchedulerNode, Scheduler
from torch._inductor.virtualized import V


class SpyreHeuristics(InductorChoices):
    @staticmethod
    def reduction_split_factor(
        device: torch.device,
        reduction_numel_hint: int,
        numel_hint: int,
        inner_reduction: bool,
    ) -> int:
        """Heuristic to decide the RSPLIT used for split reductions.
        When a reduction has a small number of outputs there is not enough parallelism,
        so we will do the reduction in two phases."""
        return 1

    @staticmethod
    def can_fuse(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        return False

    @staticmethod
    def can_fuse_vertical(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        return False

    @staticmethod
    def can_fuse_horizontal(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        return False

    def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
        """Override TritonKernel block sizes and capture problem dimensions.

        All Spyre kernels bypass Triton autotuning via FixedTritonConfig.
        Without it, Triton autotuning spawns 32 compile workers that deadlock
        on the single VFIO device.

        Spyre codegen has its own cost-model autotuner (no hardware access)
        that handles tiling decisions. Triton block sizes just need to be
        valid stick-aligned values so the Triton IR is well-formed.

        Kernel types and their block size keys:
        - Matmul (tl.dot): YBLOCK/R0_BLOCK/XBLOCK
        - Reduction: XBLOCK/R0_BLOCK
        - Pointwise: XBLOCK only
        """
        is_matmul = (features.is_reduction()
                     and hasattr(features, 'contains_op')
                     and features.contains_op("dot"))
        is_reduction = features.is_reduction()

        # Stick-aligned block sizes: 64 elements = 1 stick (128 bytes / 2 bytes)
        elements_per_stick = 64

        # Extract tiling dict for problem dimensions and block sizes.
        # groups = kernel_args = [tiling_dict] where tiling keys are:
        #   "x" → XBLOCK, "y" → YBLOCK, "r0_" → R0_BLOCK
        tiling = groups[0] if groups and isinstance(groups[0], dict) else {}
        problem_dims = {}
        _ski = getattr(V.graph.sizevars, "statically_known_int", None)

        def _resolve_dim(tiling_key):
            val = tiling.get(tiling_key)
            if val is None:
                return None
            try:
                if _ski is not None:
                    concrete = _ski(val)
                else:
                    concrete = int(V.graph.sizevars.size_hint(val))
                return concrete
            except (TypeError, ValueError):
                return None

        if is_matmul:
            # Matmul: YBLOCK=tile_m, R0_BLOCK=tile_k, XBLOCK=tile_n
            tile_m, tile_k, tile_n = 32, elements_per_stick, elements_per_stick
            m_val = _resolve_dim("y") or 0
            k_val = _resolve_dim("r0_") or 0
            n_val = _resolve_dim("x") or 0
            problem_dims = {"M": m_val, "K": k_val, "N": n_val}

            if m_val >= tile_m and k_val >= tile_k and n_val >= tile_n:
                kernel_kwargs["fixed_config"] = FixedTritonConfig({
                    "YBLOCK": tile_m,
                    "R0_BLOCK": tile_k,
                    "XBLOCK": tile_n,
                })

        elif is_reduction:
            # Reduction: XBLOCK + R0_BLOCK, both stick-aligned
            x_val = _resolve_dim("x") or 0
            r0_val = _resolve_dim("r0_") or 0
            problem_dims = {"X": x_val, "R0": r0_val}

            kernel_kwargs["fixed_config"] = FixedTritonConfig({
                "XBLOCK": elements_per_stick,
                "R0_BLOCK": elements_per_stick,
            })

        else:
            # Pointwise: XBLOCK only, stick-aligned
            # Extract M and N for the codegen's autotuner. Pointwise is tiled
            # as (M, 1, N) where N must be a multiple of elements_per_stick.
            # For 2D grids (y + x), M = y_val, N = x_val.
            # For 1D grids (x only), decompose: N = elements_per_stick, M = x_val / N.
            x_val = _resolve_dim("x") or 0
            y_val = _resolve_dim("y") or 0

            if y_val > 0 and x_val > 0:
                m_val = y_val
                n_val = x_val
            elif x_val > 0:
                # 1D grid: decompose into M rows of N stick-aligned elements
                n_val = elements_per_stick
                m_val = (x_val + n_val - 1) // n_val  # ceil division
            else:
                m_val = 1
                n_val = elements_per_stick

            problem_dims = {"M": m_val, "N": n_val}

            kernel_kwargs["fixed_config"] = FixedTritonConfig({
                "XBLOCK": elements_per_stick,
            })

        # Pass problem dims through FixedTritonConfig → cfg.kwargs →
        # CachingAutotuner._create_compile_options() → options dict →
        # backend.parse_options() → SpyreOptions.problem_m/n/k.
        #
        # NOTE: problem_m/n/k CANNOT go directly in kernel_kwargs because
        # SIMDKernel.__init__() rejects unknown kwargs. They must be embedded
        # inside FixedTritonConfig which is a recognized kwarg. At runtime,
        # FixedTritonConfig.config becomes cfg.kwargs in the CachingAutotuner.
        # A monkey-patch on _create_compile_options (in __init__.py _autoload)
        # copies Spyre-recognized keys from cfg.kwargs into the options dict,
        # so backend.parse_options() receives them.
        if problem_dims and tiling:
            m = problem_dims.get("M", 0)
            n = problem_dims.get("N", 0)
            k = problem_dims.get("K", 0)
            config_dict = kernel_kwargs.get("fixed_config", FixedTritonConfig({})).config.copy()
            if m > 0:
                config_dict["problem_m"] = m
            if n > 0:
                config_dict["problem_n"] = n
            if k > 0:
                config_dict["problem_k"] = k
            kernel_kwargs["fixed_config"] = FixedTritonConfig(config_dict)

        return kernel_kwargs
