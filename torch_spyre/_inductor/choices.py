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

import threading

import torch

from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.triton import FixedTritonConfig
from torch._inductor.scheduler import BaseSchedulerNode, Scheduler
from torch._inductor.virtualized import V


# Module-level store for passing problem dims from Inductor scheduling
# to the Spyre backend compilation stage. Keyed by kernel name to
# avoid race conditions with multiple kernel compilations.
_spyre_problem_dims_store: dict[str, dict[str, int]] = {}
_spyre_problem_dims_lock = threading.Lock()


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

        For native matmul kernels (tl.dot path):
        - Inject FixedTritonConfig with stick-aligned block sizes
        - Extract concrete M/K/N from the tiling dict and store for
          the Spyre backend to read during _make_spyre_ir()
        """
        if not (features.is_reduction()
                and hasattr(features, 'contains_op')
                and features.contains_op("dot")):
            return kernel_kwargs

        # Stick-aligned block sizes for native matmul
        # YBLOCK=tile_m (M), R0_BLOCK=tile_k (K), XBLOCK=tile_n (N)
        # tile_k=64 elements = 1 stick, tile_n=64 elements = 1 stick
        tile_m, tile_k, tile_n = 32, 64, 64

        # Extract problem dims from tiling dict to check compatibility
        # groups is kernel_args = [tiling_dict] where tiling = {"y": M, "x": N, "r0_": K}
        problem_dims = {}
        if groups and isinstance(groups[0], dict):
            tiling = groups[0]
            dim_map = {"M": "y", "N": "x", "K": "r0_"}
            for dim_name, tiling_key in dim_map.items():
                val = tiling.get(tiling_key)
                if val is not None:
                    try:
                        # Prefer statically_known_int() (PyTorch 2.12+)
                        # which returns None for symbolic expressions. We
                        # must NOT inject FixedTritonConfig with heuristic
                        # estimates from size_hint() — a wrong block size
                        # causes Triton assertion failures at runtime.
                        # PyTorch 2.11 lacks statically_known_int() but all
                        # Spyre shapes are currently static, so size_hint()
                        # is safe there. When upgrading to 2.12, remove the
                        # fallback and use only statically_known_int().
                        _ski = getattr(
                            V.graph.sizevars, "statically_known_int", None
                        )
                        if _ski is not None:
                            concrete = _ski(val)
                        else:
                            # PyTorch 2.11: size_hint() is the only option.
                            # Safe for static shapes (our current use case).
                            concrete = int(
                                V.graph.sizevars.size_hint(val)
                            )
                        if concrete is None:
                            # Symbolic expression — skip this dimension.
                            continue
                        problem_dims[dim_name] = concrete
                    except (TypeError, ValueError):
                        pass

        # Only inject FixedTritonConfig if problem dims are compatible.
        # Block sizes must not exceed actual dimensions (e.g., M=16 with
        # YBLOCK=32 would cause a Triton assertion failure).
        m_val = problem_dims.get("M", 0)
        k_val = problem_dims.get("K", 0)
        n_val = problem_dims.get("N", 0)
        if m_val >= tile_m and k_val >= tile_k and n_val >= tile_n:
            kernel_kwargs["fixed_config"] = FixedTritonConfig({
                "YBLOCK": tile_m,
                "R0_BLOCK": tile_k,
                "XBLOCK": tile_n,
            })
        # else: skip FixedTritonConfig, let Triton autotune.
        # Small problems (< stick-aligned block sizes) don't benefit from
        # hardware matmul and should fall back to pointwise decomposition.

        # Store problem dims for the Spyre backend to read in _make_spyre_ir().
        if problem_dims and groups and isinstance(groups[0], dict):
            tiling = groups[0]
            kernel_key = f"{kernel_cls.__name__}_{id(tiling)}"
            with _spyre_problem_dims_lock:
                _spyre_problem_dims_store[kernel_key] = problem_dims
                # Also store as "_latest" for the common single-kernel case.
                # With FixedTritonConfig, autotuning is bypassed so only one
                # compile happens per triton_kernel_kwargs call.
                _spyre_problem_dims_store["_latest"] = problem_dims

        return kernel_kwargs
