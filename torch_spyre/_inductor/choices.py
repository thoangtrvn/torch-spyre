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

import math

import torch

from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.triton import FixedTritonConfig, TRITON_MAX_BLOCK
from torch._inductor.scheduler import BaseSchedulerNode, Scheduler
from torch._inductor.virtualized import V

from torch_spyre._C import get_elem_in_stick


class SpyreFixedTritonConfig(FixedTritonConfig):
    """A FixedTritonConfig that answers EVERY block-size query Inductor makes.

    torch's ``TritonKernel.max_block`` does a raw
    ``self.fixed_config[f"{prefix.upper()}BLOCK"]`` for ANY range-tree prefix
    (via ``_has_constant_mask`` -> ``iteration_ranges_codegen_header``, which
    runs for every tree during kernel construction). The injected key set
    carries EXACTLY the signature-declared keys (the ASTSource binding
    invariant documented on ``triton_kernel_kwargs``), so any kernel whose
    range-tree prefixes exceed the predicted key set died with
    ``KeyError: 'YBLOCK'`` / ``KeyError: 'R0_BLOCK'`` at codegen time. On CUDA
    ``fixed_config is None`` (autotune) and ``max_block`` falls back to
    ``TRITON_MAX_BLOCK`` — this class restores that same fallback on the
    QUERY side only: an undeclared ``<PREFIX>BLOCK`` key resolves to torch's
    own ``TRITON_MAX_BLOCK[prefix]`` (the exact value the CUDA path returns),
    everything else KeyErrors as before.

    Deliberately NOT a key-set change. Everything the config channel
    introspects is untouched:

    - ``.config`` (the dict) is byte-identical, so the runtime channel —
      codegen emits ``config={self.fixed_config.config!r}`` as a plain dict
      literal — and the autotune cache key derived from it are unchanged.
    - ``__eq__`` compares by ``.config`` against ANY FixedTritonConfig
      (subclass or plain), which is exactly the plain class's semantics for
      plain-on-plain pairs; the inherited dataclass ``__eq__`` would compare
      ``other.__class__ is self.__class__`` and make a wrapped config
      compare UNEQUAL to a plain config with the same dict — a silent
      cache-hit surface change. The class stays unhashable, like the plain
      dataclass (eq without frozen).
    - ``__contains__`` stays exact-key: the ``_has_constant_mask``
      ``f"...BLOCK" in self.fixed_config`` short-circuit and the
      exactly-declared-keys binding invariant are preserved. The fallback is
      a property of the query side, not of the key set.

    Boundary: keys that do not name a ``TRITON_MAX_BLOCK`` prefix (e.g.
    ``"RSPLIT"``, read by ``max_rsplit`` on cooperative reductions) still
    raise ``KeyError`` — mirroring ``TRITON_MAX_BLOCK`` there would need
    ``TRITON_MAX_RSPLIT`` semantics this lane does not own.
    """

    def __getitem__(self, item):
        try:
            return self.config[item]
        except KeyError:
            prefix = item.removesuffix("BLOCK")
            if prefix != item and prefix in TRITON_MAX_BLOCK:
                return TRITON_MAX_BLOCK[prefix]
            raise

    def __eq__(self, other):
        # The dataclass-generated __eq__ compares other.__class__ is
        # self.__class__, which would make this subclass compare UNEQUAL to
        # a plain FixedTritonConfig carrying the same dict. Compare by
        # .config against any FixedTritonConfig so the equality surface is
        # identical to the all-plain world (defining __eq__ also leaves this
        # class unhashable, matching the plain dataclass).
        if isinstance(other, FixedTritonConfig):
            return self.config == other.config
        return NotImplemented


class SpyreHeuristics(InductorChoices):
    @staticmethod
    def uuid():
        return "torch_spyre._inductor.choices.SpyreHeuristics.v1"

    @staticmethod
    def _recover_pointwise_shape(features, x_val):
        """Recover the TRUE 2D (M, N) of a 1D-flattened pointwise kernel.

        Inductor collapses a contiguous [M,N] pointwise into a 1D iteration grid
        {"x": M*N}, but does NOT reshape the buffers — the output ComputedBuffer's
        layout still carries [M, N] (Buffer.get_size() returns [*layout.size]).
        Read it back (same "Inductor collapsed the shape" recovery class as the
        #193 reduction dim recovery). Returns (M, N), or (x_val, None) when a 2D
        layout matching x_val can't be found — the caller then falls back to the
        old flat decompose (safe for aligned / M=1; fails safe, never silent-wrong).
        """
        try:
            sizevars = V.graph.sizevars
            helpers = tuple(
                helper
                for helper in (
                    getattr(sizevars, "statically_known_int", None),
                    getattr(sizevars, "guard_int", None),
                    getattr(sizevars, "size_hint", None),
                )
                if helper is not None
            )

            def _to_int(dim):
                for helper in helpers:
                    try:
                        return int(helper(dim))
                    except (TypeError, ValueError, AttributeError, RuntimeError):
                        pass
                return int(dim)

            for node in features.scheduler_nodes():
                buf = getattr(node, "node", None)
                size = buf.get_size() if buf is not None else None
                if size and len(size) == 2:
                    m = _to_int(size[0])
                    n = _to_int(size[1])
                    if m and n and m * n == x_val:
                        return m, n
        except (TypeError, ValueError, AttributeError):
            pass
        return x_val, None

    @staticmethod
    def _kernel_dtype(features):
        """Resolve the dtype of the (unfused) kernel's buffers.

        Spyre fusion is disabled (can_fuse/can_fuse_vertical/can_fuse_horizontal
        all return False), so a Spyre kernel has exactly one buffer dtype; read
        it back from each scheduler node's underlying IR buffer (same
        node.node recovery as _recover_pointwise_shape). Refuses — never
        silently defaults — when no dtype resolves or a node disagrees: the
        elements-per-stick derived from it sizes every block, so a wrong
        guess mistiles the whole kernel.
        """
        dtype = None
        for node in features.scheduler_nodes():
            buf = getattr(node, "node", None)
            get_dtype = getattr(buf, "get_dtype", None)
            node_dtype = get_dtype() if get_dtype is not None else None
            if node_dtype is None:
                continue
            if dtype is None:
                dtype = node_dtype
            elif node_dtype != dtype:
                raise NotImplementedError(
                    "SpyreHeuristics cannot size a mixed-dtype kernel "
                    f"({dtype} vs {node_dtype}) — elements-per-stick is ambiguous"
                )
        if dtype is None:
            raise NotImplementedError(
                "SpyreHeuristics could not resolve a dtype from the kernel's "
                "scheduler nodes — elements-per-stick is undefined"
            )
        return dtype

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

    def _kernel_persistent_reduction(self, kernel_cls, features, kernel_kwargs):
        """The CONSTRUCTED kernel's own loop/persistent decision, evaluated at
        hook time from the same inputs ``SIMDKernel.__init__`` resolves
        (torch/_inductor/codegen/simd.py): ``override_persistent_reduction``
        from kernel_kwargs wins; otherwise ``inside_reduction and
        should_use_persistent_reduction(features, cooperative_reduction)``
        with the kernel's cooperative_reduction
        (``override_cooperative_reduction`` if set, else
        ``should_use_cooperative_reduction(device, numel, reduction_numel)``).

        Why the override must be honored (executed witness: dynamic masked-sum
        pooling, reduction_numel s20): torch's
        ``TritonScheduling.create_kernel_choices`` calls
        ``TritonKernel.apply_feature_required_overrides`` BEFORE this hook,
        and that call FORCES ``override_persistent_reduction=False`` whenever
        ``TritonKernel.has_persistent_RBLOCK(reduction_numel)`` fails — which
        it does for dynamic reduction extents with no static upper bound at
        that point (s20 bounds were [2, int_oo] at override time). The kernel
        is then LOOPED no matter what the heuristics predict. Calling
        ``should_use_persistent_reduction`` directly here can disagree with
        that (our own dim resolution specializes the symbol afterwards, so
        the heuristic then says persistent) — a prediction that can disagree
        with the codegen it predicts is a defect, not a tuning knob.

        The kernel CLASS is also an input: ``create_kernel_choices`` swaps
        kernel_type to ``TritonSplitScanKernel`` for split scans BEFORE the
        hook, and that class overrides ``should_use_persistent_reduction``
        to a hard False (always looped; the scan path sets only
        ``override_cooperative_reduction``, not the persistent override), so
        the heuristic chain below must not run for it. Latent on spyre today
        (scan ops do not reach TritonKernel through the spyre frontend —
        verified compile-only), but the key set is keyed on the kernel's
        decision, not on today's reachability.

        KNOWN UNSUPPORTED (pre-existing, disclosed): (a) split-scan kernels
        assert ``fixed_config is None`` at construction
        (triton_split_scan.py), so hook injection is incompatible with them
        regardless of this gate; (b) ``config.triton.multi_kernel`` (torch
        default 0; torch_spyre never sets it) makes
        ``add_multi_kernel_choices`` construct extra looped kernel variants
        AFTER the hook reusing the SAME fixed_config — a persistent primary's
        config (no R0_BLOCK) would then be shared with a looped variant that
        requires it (KeyError 'R0_BLOCK' in max_block). Neither is reachable
        under spyre defaults.
        """
        if kernel_cls is not None:
            from torch._inductor.codegen.triton_split_scan import (
                TritonSplitScanKernel,
            )

            if issubclass(kernel_cls, TritonSplitScanKernel):
                return False
        override = kernel_kwargs.get("override_persistent_reduction")
        if override is not None:
            return bool(override)
        cooperative = kernel_kwargs.get("override_cooperative_reduction")
        if cooperative is None:
            cooperative = (
                features.is_reduction()
                and self.should_use_cooperative_reduction(
                    V.graph.get_current_device_or_throw(),
                    features.numel,
                    features.reduction_numel,
                )
            )
        return features.is_reduction() and self.should_use_persistent_reduction(
            features, cooperative
        )

    def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
        """Override TritonKernel block sizes and capture problem dimensions.

        All Spyre kernels bypass Triton autotuning via FixedTritonConfig.
        Without it, Triton autotuning spawns 32 compile workers that deadlock
        on the single VFIO device.

        Spyre codegen has its own cost-model autotuner (no hardware access)
        that handles tiling decisions. Triton block sizes just need to be
        valid stick-aligned values so the Triton IR is well-formed.

        INVARIANT: the fixed_config must contain EXACTLY the *_BLOCK keys that the
        generated kernel declares as constexpr parameters — no more, no fewer.
        Triton's ASTSource binds each config key via `arg_names.index(key)`, so a key
        the kernel does not declare raises `ValueError: '<KEY>' is not in list` (and a
        missing required key would leave the block size unset). The concrete kernel does
        not exist at this hook (kernel_cls is the CLASS, not an instance), so we cannot
        read the signature directly — instead we derive the declared keys from the SAME
        inputs the constructed kernel resolves its loop/persistent decision from
        (`_kernel_persistent_reduction`: torch's `override_persistent_reduction` /
        `override_cooperative_reduction` kernel_kwargs, set by
        `apply_feature_required_overrides` BEFORE this hook, then the heuristic
        chain `SIMDKernel.__init__` itself runs). Calling the heuristic directly
        with `cooperative_reduction=False` is NOT sufficient: the pre-hook
        override can force the kernel looped for dynamic reduction extents that
        `should_use_persistent_reduction` would call persistent (executed
        witness: dynamic masked-sum pooling — see `_kernel_persistent_reduction`).

        Kernel types and the block keys their signature declares:
        - Matmul (tl.dot): YBLOCK/XBLOCK always; R0_BLOCK only when NOT persistent
          (K fits one block → baked in-body); ZBLOCK only when the kernel is 3-range
          (batched bmm: the tiling dict carries a "z" range tree). A 3-range kernel
          declares one `<PREFIX>BLOCK` constexpr per range-tree root, so the z tree
          means ZBLOCK is a required key — omitting it left the constexpr unbound and
          crashed ast_to_ttir on `tl.program_id(0) * ZBLOCK`.
        - Reduction: XBLOCK always; R0_BLOCK ONLY when NOT persistent. A persistent
          reduction (reduction extent fits one block) bakes R0_BLOCK as an in-body
          `tl.constexpr` and does NOT declare it as a parameter — so R0_BLOCK is gated
          on the kernel's own persistent decision (`_kernel_persistent_reduction`).
          Injecting it unconditionally was the #193 'R0_BLOCK' is not in list break.
        - Pointwise: XBLOCK only. Always blocked in x → XBLOCK always declared.
        """
        is_matmul = (
            features.is_reduction()
            and hasattr(features, "contains_op")
            and features.contains_op("dot")
        )
        is_reduction = features.is_reduction()

        # Stick-aligned block sizes. Elements per stick is dtype-dependent
        # (one stick is a fixed 128 bytes: 64 elements at 16-bit, 128 at
        # 8-bit, 32 at 32-bit — torch_spyre._C.get_elem_in_stick). Derive it
        # from the kernel's dtype instead of hardcoding the 16-bit value,
        # which silently mistiled every non-16-bit kernel.
        elements_per_stick = get_elem_in_stick(self._kernel_dtype(features))

        # Extract tiling dict for problem dimensions and block sizes.
        # groups = kernel_args = [tiling_dict] where tiling keys are:
        #   "x" → XBLOCK, "y" → YBLOCK, "r0_" → R0_BLOCK
        tiling = groups[0] if groups and isinstance(groups[0], dict) else {}
        problem_dims = {}
        _ski = getattr(V.graph.sizevars, "statically_known_int", None)
        _guard_int = getattr(V.graph.sizevars, "guard_int", None)
        _size_hint = getattr(V.graph.sizevars, "size_hint", None)

        def _resolve_dim(tiling_key):
            val = tiling.get(tiling_key)
            if val is None:
                return None
            try:
                if _ski is not None:
                    concrete = _ski(val)
                elif _guard_int is not None:
                    concrete = _guard_int(val)
                elif _size_hint is not None:
                    concrete = int(_size_hint(val))
                else:
                    concrete = int(val)
                return concrete
            except (TypeError, ValueError, AttributeError, RuntimeError):
                return None

        if is_matmul:
            # Matmul (tl.dot): YBLOCK/XBLOCK always; R0_BLOCK ONLY for looped (non-persistent)
            # kernels — SAME persistence split as the reduction branch below. When the
            # K-contraction fits one block (K<=64), Inductor makes the tl.dot reduction
            # PERSISTENT and bakes R0_BLOCK as an in-body `tl.constexpr` — it is NOT a
            # signature parameter, so injecting it makes Triton's ASTSource do
            # `arg_names.index("R0_BLOCK")` on a signature that lacks it ->
            # `ValueError: 'R0_BLOCK' is not in list` (HW-observed for sp1_2x64x64: signature
            # was {XBLOCK,YBLOCK} only). Gate R0_BLOCK on the kernel's own
            # persistent decision (`_kernel_persistent_reduction` — the pre-hook
            # override_persistent_reduction plus the heuristic chain
            # SIMDKernel.__init__ resolves) so the fixed_config key-set matches the
            # generated signature exactly — the same rule that fixed reduction #193.
            #
            # Separately, the prior `m_val>=32` guard installed NO fixed_config for SP5's
            # small-M matmul (M=2/4) -> Inductor KeyError:'XBLOCK' (_has_constant_mask ->
            # max_block("X") -> fixed_config["XBLOCK"]). Dropping that guard (install
            # unconditionally within is_matmul) fixes small-M routing; block >= dim is masked
            # (same as the reduction branch's XBLOCK=64 for small x_val).
            #
            # Block sizes stay stick-const (YBLOCK=32 matches codegen MICRO_M=32; do NOT
            # clamp below it, or the Triton block desyncs from the seg-relative tile).
            # Systematic key-derivation from the tiling-dict prefixes is deferred (#178).
            tile_m, tile_k, tile_n = 32, elements_per_stick, elements_per_stick
            m_val = _resolve_dim("y") or 0
            k_val = _resolve_dim("r0_") or 0
            n_val = _resolve_dim("x") or 0
            problem_dims = {"M": m_val, "K": k_val, "N": n_val}
            persistent = self._kernel_persistent_reduction(
                kernel_cls, features, kernel_kwargs
            )
            block_config = {"YBLOCK": tile_m, "XBLOCK": tile_n}
            if not persistent:
                block_config["R0_BLOCK"] = tile_k
            # 3-range (batched bmm) matmul: the tiling dict carries a "z" range
            # tree, and every range-tree root declares a `<PREFIX>BLOCK`
            # constexpr parameter — so ZBLOCK is REQUIRED here, not optional.
            # Omitting it left the ZBLOCK constexpr unbound in Triton's
            # ast_to_ttir (ZBLOCK resolved to None; `tl.program_id(0) *
            # ZBLOCK` crashed with AttributeError "'NoneType' object has no
            # attribute 'type'", surfacing from Inductor as a ZBLOCK
            # KeyError). The value 1 is Inductor's own bmm policy —
            # torch/_inductor/codegen/triton.py keeps the batch dimension
            # unblocked ("we force ZBLOCK to be always 1 during autotune";
            # the 2d-dot reshape path also rewrites ZBLOCK out of the block
            # shape) — so this key is a signature-shape obligation, not a
            # tiling decision.
            if "z" in tiling:
                # PREDICTION-DRIFT NOTE (the #193 class): this gate is keyed on
                # the TILING DICT carrying "z", not on an Inductor decision
                # function (unlike the R0_BLOCK gate above, which resolves the
                # kernel's own decision via `_kernel_persistent_reduction` —
                # the bare heuristic call CAN drift from the codegen, see that
                # helper's docstring). If a 3-range
                # matmul ever surfaces whose tiling dict LACKS "z" (an Inductor
                # schedule change re-rooting the batch range, or a new prefixed
                # range family), the ZBLOCK key silently goes missing again and
                # the compile dies at ast_to_ttir with the ZBLOCK-unbound
                # AttributeError. Check on any such failure: print the tiling
                # dict keys here and compare against the range-tree prefixes in
                # torch/_inductor/codegen/triton.py's iteration_ranges (the
                # same comparison the fixed_config key-set must match).
                block_config["ZBLOCK"] = 1
            kernel_kwargs["fixed_config"] = SpyreFixedTritonConfig(block_config)

        elif is_reduction:
            # Reduction: XBLOCK always; R0_BLOCK ONLY for looped (non-persistent) kernels.
            # Inductor generates two reduction kernel shapes:
            #   - persistent (reduction extent fits one block): signature declares only
            #     XBLOCK; R0_BLOCK is baked as an in-body `tl.constexpr` and is NOT a
            #     kernel parameter. Injecting R0_BLOCK here makes Triton's ASTSource do
            #     `arg_names.index("R0_BLOCK")` on a signature that lacks it →
            #     `ValueError: 'R0_BLOCK' is not in list` (the #193 frontend break).
            #   - looped (large reduction extent): signature declares BOTH XBLOCK and
            #     R0_BLOCK (the `for r0_offset in range(...)` form) — R0_BLOCK is required.
            # The gate is the kernel's OWN decision (`_kernel_persistent_reduction`):
            # torch's `apply_feature_required_overrides` runs BEFORE this hook and can
            # force `override_persistent_reduction=False` (looped) for dynamic
            # reduction extents with no static upper bound — the bare heuristic call
            # would then predict persistent and leave the looped kernel's R0_BLOCK
            # unbound (the r0-persistent-prediction-drift defect).
            x_val = _resolve_dim("x") or 0
            r0_val = _resolve_dim("r0_") or 0
            problem_dims = {"X": x_val, "R0": r0_val}

            persistent = self._kernel_persistent_reduction(
                kernel_cls, features, kernel_kwargs
            )
            block_config = {"XBLOCK": elements_per_stick}
            if not persistent:
                block_config["R0_BLOCK"] = elements_per_stick
            kernel_kwargs["fixed_config"] = SpyreFixedTritonConfig(block_config)

        else:
            # Pointwise: XBLOCK only, stick-aligned
            # Extract M and N for the codegen's autotuner. Pointwise is tiled
            # as (M, 1, N) where N must be a multiple of elements_per_stick.
            # For 2D grids (y + x), M = y_val, N = x_val.
            # For 1D grids (x only), Inductor collapsed [M,N] to a flat numel
            # x_val=M*N, THROWING AWAY the per-row stick padding the physical
            # device tensor still carries (spyre_tensor_impl.cpp:170 row-pads to
            # M*ceil(N/64) sticks; D2H reads all of them, :286). Decomposing as
            # m_val=ceil(x_val/64) makes the kernel write only ceil(M*N/64)
            # contiguous sticks — SHORT by the trailing row-tail sticks for ragged
            # N (N%64!=0) AND M>1, which then read 0 on HW (#166: add=11/mul=28/
            # sub=7). Recover the TRUE (M,N) from the output buffer layout (Inductor
            # flattens the iteration grid, NOT the buffer size) and flatten over the
            # PADDED stick count so the kernel walks every physical stick. Row-major
            # stick order is contiguous (row r, sub-stick s -> r*ceil(N/64)+s), so a
            # contiguous M*ceil(N/64)-stick walk is correct — only the COUNT was
            # wrong. Mirrors the #193 reduction shape recovery for the same
            # "Inductor collapsed the shape" class.
            x_val = _resolve_dim("x") or 0
            y_val = _resolve_dim("y") or 0

            if y_val > 0 and x_val > 0:
                m_val = y_val
                n_val = x_val
            elif x_val > 0:
                # 1D grid: recover the true 2D (M, N) from the output buffer layout.
                true_m, true_n = self._recover_pointwise_shape(features, x_val)
                n_val = elements_per_stick
                if (
                    true_n is not None
                    and true_n % elements_per_stick != 0
                    and true_m > 1
                ):
                    # Ragged N with M>1: flatten over the ROW-PADDED physical stick
                    # count so the kernel writes M*ceil(N/64) sticks (== the device
                    # tensor), not the flat ceil(M*N/64). This is the ONLY case that
                    # changes vs prior behavior (#166 fix).
                    m_val = true_m * math.ceil(true_n / elements_per_stick)
                else:
                    # M=1, or N a stick multiple: flat == row-padded already.
                    # Unchanged (byte-identical to prior behavior — verified for
                    # [1,100],[2,64],[128,768],[512,4096]).
                    m_val = (x_val + n_val - 1) // n_val  # ceil division
            else:
                m_val = 1
                n_val = elements_per_stick

            problem_dims = {"M": m_val, "N": n_val}

            # Pointwise: XBLOCK always; YBLOCK/ZBLOCK when the tiling dict
            # carries those range trees. Every range-tree root declares a
            # `<PREFIX>BLOCK` constexpr parameter (the same signature law the
            # matmul branch's ZBLOCK gate follows), so a 2D-grid pointwise
            # kernel (e.g. a broadcast multiply [B,S,H]*[B,S,1] scheduling y+x
            # trees) REQUIRES YBLOCK in the fixed_config — omitting it made
            # Inductor's _has_constant_mask -> max_block("y") die with
            # KeyError: 'YBLOCK' (MiniLM pooled-embedding repro). Keyed on the
            # tiling dict (the scheduled range trees), not a shape guess, so
            # it cannot drift from the generated signature. Values are
            # stick-aligned; block >= dim is masked, same as XBLOCK.
            # Wrapped in SpyreFixedTritonConfig (fixed-config-fallback lane):
            # query-side TRITON_MAX_BLOCK fallback for any undeclared
            # <PREFIX>BLOCK, .config byte-identical to the plain class.
            pointwise_block_config = {"XBLOCK": elements_per_stick}
            if "y" in tiling:
                pointwise_block_config["YBLOCK"] = elements_per_stick
            if "z" in tiling:
                # Inductor's own policy: the batch dimension stays unblocked.
                pointwise_block_config["ZBLOCK"] = 1
            kernel_kwargs["fixed_config"] = SpyreFixedTritonConfig(
                pointwise_block_config
            )

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
        config_dict = kernel_kwargs.get(
            "fixed_config", FixedTritonConfig({})
        ).config.copy()
        if problem_dims and tiling:
            m = problem_dims.get("M", 0)
            n = problem_dims.get("N", 0)
            k = problem_dims.get("K", 0)
            if m > 0:
                config_dict["problem_m"] = m
            if n > 0:
                config_dict["problem_n"] = n
            if k > 0:
                config_dict["problem_k"] = k

        from . import config as spyre_config

        # T-SR-39: SENCORES core-budget bridge. The single validated
        # config read (config.sencores) flows to SpyreOptions.num_cores
        # through the SAME serialized FixedTritonConfig channel as
        # problem_m/n/k — kernel kwargs are function arguments, so
        # async_compile worker processes see the scheduling process's
        # value. An env-read default_factory in the fork's SpyreOptions
        # would NOT survive reliably across those workers (the
        # _spyre_problem_dims_store defect class) and is the forbidden
        # pattern. Unconditional: explicit provenance on every compile,
        # and the default (32) matches the fork field default so the
        # options cache key is unchanged when SENCORES is unset.
        config_dict["num_cores"] = spyre_config.sencores

        # T-SR-40: torch-level ttir_parse_backend channel. None (default)
        # injects nothing — the fork's own default ("legacy") stands and
        # the options cache key is unchanged. "structured" is auto-paired
        # with require_stage0_admission=True: the T-SR-38 fence makes
        # structured-without-admission the convicted non-authority-shell
        # trap, so this channel cannot express it (the fork's
        # parse_options refusal remains the backstop for direct callers).
        parse_backend = spyre_config.ttir_parse_backend
        if parse_backend is not None:
            if parse_backend not in ("legacy", "structured"):
                raise ValueError(
                    "torch_spyre._inductor.config.ttir_parse_backend must be "
                    f"'legacy', 'structured', or None (fork default), got "
                    f"{parse_backend!r}"
                )
            config_dict["ttir_parse_backend"] = parse_backend
            if parse_backend == "structured":
                config_dict["require_stage0_admission"] = True

        route_request = {
            "_descriptor_normal_route_request_operation": (
                spyre_config.descriptor_normal_route_operation
            ),
            "_descriptor_normal_route_request_rollout": (
                spyre_config.descriptor_normal_route_rollout
            ),
            "_descriptor_normal_route_request_kill_switch": (
                spyre_config.descriptor_normal_route_kill_switch
            ),
            "_descriptor_normal_route_request_authority_fingerprint": (
                spyre_config.descriptor_normal_route_authority_fingerprint
            ),
        }
        if route_request["_descriptor_normal_route_request_operation"] != "off":
            from torch._inductor.async_compile import get_compile_threads
            from sentient_codegen.backend.descriptor_unary_route_authority import (
                resolve_descriptor_unary_normal_route_authority,
            )

            if get_compile_threads() != 1:
                raise RuntimeError(
                    "descriptor normal route requires same-process Inductor "
                    "compilation with TORCHINDUCTOR_COMPILE_THREADS=1"
                )
            resolve_descriptor_unary_normal_route_authority(
                route_request["_descriptor_normal_route_request_operation"],
                route_request["_descriptor_normal_route_request_authority_fingerprint"],
            )
            config_dict.update(route_request)
        kernel_kwargs["fixed_config"] = SpyreFixedTritonConfig(config_dict)

        return kernel_kwargs
