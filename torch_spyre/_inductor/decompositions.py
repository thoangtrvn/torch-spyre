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

import math
from typing import Optional, Union, Sequence, Callable, TypeVar
from typing_extensions import ParamSpec
import torch
from torch.utils import _pytree as pytree
import torch._decomp as decomp

from torch_spyre.constants import DEVICE_NAME
from .errors import Unsupported
from . import customops  # noqa: F401

import threading

# A module-level lock to make the CM thread-safe
_decompositions_lock = threading.RLock()

# Dictionary for Spyre-specific decompositions
spyre_decompositions: dict = {}

# Exclude specific Inductor default decompositions on Spyre.
# Some Inductor decompositions do not work reliably on the Spyre backend yet.
# We disable them here and rely on implicit fallbacks to eager ops instead. Once
# the blocking issues are resolved, these exclusions can be removed.
spyre_decompositions_to_exclude = [
    torch.ops.aten.triu,
    torch.ops.aten.tril,
    # Exclude aten.mm decomposition so that aten.mm reaches tuned_mm, which
    # uses the native matmul path (ops.dot + make_reduction("dot")) when
    # use_native_matmul() returns True. The in-tree decomposition at
    # torch/_inductor/decomposition.py:424 decomposes mm into
    # unsqueeze+mul+sum for M==1 cases, which prevents tuned_mm from ever
    # being called and leaves extern_kernels.mm in the compiled function.
    torch.ops.aten.mm,
    # Same for bmm — exclude decomposition so tuned_bmm handles it natively.
    torch.ops.aten.bmm,
]

# Dict for Spyre-specific decompositions to be registered via DispatchKey
spyre_decompositions_via_dispatchkey: dict = {}

# Module-level Library objects kept alive permanently so that the registered
# PrivateUse1 / AutogradPrivateUse1 kernels are never unregistered by garbage collector.
# (torch.library.Library uses weakref.finalize → m.reset() on GC, which would
# silently remove the kernels from the C++ dispatcher.)
_spyre_autograd_lib = None
_spyre_lib = None
_dispatchkey_kernels_registered = False

_T = TypeVar("_T")
_P = ParamSpec("_P")


def register_spyre_decomposition(
    ops: Union[torch._ops.OperatorBase, list],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Register decompositions specifically for Spyre device.
    These will only be active when compiling for the Spyre device.

    For ``aten`` ops, this also registers a PrivateUse1 dispatch kernel
    (via ``register_spyre_decompositions_via_dispatchkey``) so that
    eager-mode dispatch on a Spyre tensor reaches the Spyre implementation.
    This is necessary for ops with CompositeImplicitAutograd (CIA) in
    upstream PyTorch, and harmless for non-CIA ops.
    """

    def decorator(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        # 1. Register in the Spyre decomposition table (for compile mode / make_fx)
        decomp.register_decomposition(ops, spyre_decompositions)(fn)

        # 2. For aten ops, also register via PrivateUse1 dispatch key (for eager mode).
        #    Non-aten ops (e.g. spyre::compact) are custom Spyre ops that don't need
        #    PrivateUse1 kernel registration.
        #    Skip ops that already have a PrivateUse1 kernel (e.g. from eager.py) to
        #    avoid registration conflicts.
        ops_list = ops if isinstance(ops, list) else [ops]
        aten_ops = [
            op
            for op in ops_list
            if getattr(op, "namespace", None) == "aten"
            and not torch._C._dispatch_has_kernel_for_dispatch_key(
                op._name, "PrivateUse1"
            )
        ]
        if aten_ops:
            register_spyre_decompositions_via_dispatchkey(aten_ops)(fn)

        return fn

    return decorator


# Decompositions that produce spyre.* custom ops, which then trigger SDSC-specific
# lowerings. These must be excluded in the Triton path so that Inductor's standard
# decompositions produce standard aten ops that TritonScheduling can handle.
_TRITON_INCOMPATIBLE_DECOMPOSITIONS = frozenset({
    torch.ops.aten.layer_norm.default,   # → spyre.layernormscale + spyre.layernormnorm
    torch.ops.aten.rms_norm.default,     # → spyre.exx2
    torch.ops.aten.gelu.default,         # → spyre.gelu
    torch.ops.aten.softplus.default,     # → spyre.softplus
    torch.ops.aten.topk,                 # → spyre.topkvalue + spyre.topkindex
})


# Context manager that enables spyre specific decompositions in addition to PyTorch in-tree decompositions
@contextmanager
def enable_spyre_decompositions(
    decomps: Optional[dict[torch._ops.OperatorBase, Callable]] = None,
    *,
    triton_path: bool = False,
):
    """
    CM that enables Spyre decompositions:
      - Temporarily adds relevant Spyre decompositions to provided decomposition table `decomps`
      - Restore original decompositions table on exit

    This CM is reentrant and safe under nested usage.

    Args:
        decomps: Decomposition table to modify. Maps operator overloads to their
            decomposition implementations. Defaults to PyTorch Inductor's global
            decomposition registry (torch._inductor.decomposition.decompositions).
        triton_path: If True, exclude decompositions that produce spyre.* custom
            ops. In the Triton path, Inductor's standard decompositions produce
            standard aten ops (e.g., aten.native_layer_norm instead of
            spyre.layernormscale), which TritonScheduling compiles to TTIR
            with standard tt.reduce operations.
    """
    if decomps is None:
        decomps = torch._inductor.decomposition.decompositions

    with _decompositions_lock:
        from torch_spyre.ops.fallbacks import fallback_ops
        from torch._ops import OpOverload, OpOverloadPacket

        # Helper function to remove ops from decompositions
        def _fetch_and_remove_op(ops):
            _removed = {}
            for op in ops:
                if isinstance(op, OpOverloadPacket):
                    for overload_name in op.overloads():
                        opo = getattr(op, overload_name)
                        op_ret = decomps.pop(opo, None)
                        if op_ret is not None:
                            _removed[opo] = op_ret
                elif isinstance(op, OpOverload):
                    op_ret = decomps.pop(op, None)
                    if op_ret is not None:
                        _removed[op] = op_ret
            return _removed

        # 1. Add/override spyre-specific decompositions
        excluded_decomps = _TRITON_INCOMPATIBLE_DECOMPOSITIONS if triton_path else frozenset()
        saved_intree_decompositions = {}
        for (
            spyre_decompositions_op,
            spyre_decompositions_impl,
        ) in spyre_decompositions.items():
            if spyre_decompositions_op in excluded_decomps:
                continue
            if spyre_decompositions_op in decomps:
                saved_intree_decompositions[spyre_decompositions_op] = decomps[
                    spyre_decompositions_op
                ]
            decomps[spyre_decompositions_op] = spyre_decompositions_impl

        # Attach to the function so we can restore on last exit
        enable_spyre_decompositions._saved_decompositions = saved_intree_decompositions

        # 2. Remove selected decompositions from Inductor's registry for spyre
        _removed_decompositions_to_exclude = _fetch_and_remove_op(
            spyre_decompositions_to_exclude
        )

        # Attach to the function so we can restore on last exit
        enable_spyre_decompositions._removed_decompositions_to_exclude = (
            _removed_decompositions_to_exclude
        )

        # 3. Remove selected decompositions for fallback ops defined in fallbacks.py
        _removed_decompositions_fallback_ops = _fetch_and_remove_op(fallback_ops)

        # Attach to the function so we can restore on last exit
        enable_spyre_decompositions._removed_decompositions_fallback_ops = (
            _removed_decompositions_fallback_ops
        )

        try:
            yield decomps
        finally:
            # Inverse order compared to when entering the context manager

            # 1. Revert selected decompositions that have been marked for fallback ops
            removed_decompositions_fallback_ops = getattr(
                enable_spyre_decompositions,
                "_removed_decompositions_fallback_ops",
                {},
            )
            [
                torch._decomp._add_op_to_registry(decomps, op, fn)
                for op, fn in removed_decompositions_fallback_ops.items()
            ]

            # 2. Revert selected decompositions that have been removed from Inductor's registry for spyre
            removed_decompositions_to_exclude = getattr(
                enable_spyre_decompositions,
                "_removed_decompositions_to_exclude",
                {},
            )
            [
                torch._decomp._add_op_to_registry(decomps, op, fn)
                for op, fn in removed_decompositions_to_exclude.items()
            ]

            # 3. Reset the saved in-tree lowerings if needed
            saved_intree_decompositions = getattr(
                enable_spyre_decompositions, "_saved_decompositions", {}
            )
            for (
                spyre_decompositions_op,
                spyre_decompositions_impl,
            ) in spyre_decompositions.items():
                if spyre_decompositions_op in saved_intree_decompositions:
                    decomps[spyre_decompositions_op] = saved_intree_decompositions[
                        spyre_decompositions_op
                    ]
                else:
                    decomps.pop(spyre_decompositions_op, None)

            # Clean up
            enable_spyre_decompositions._saved_decompositions = {}
            enable_spyre_decompositions._removed_decompositions_to_exclude = {}
            enable_spyre_decompositions._removed_decompositions_fallback_ops = {}


def _register_spyre_dispatchkey_kernels_permanently():
    """
    Permanently register PrivateUse1 / AutogradPrivateUse1 kernels for all ops
    in ``spyre_decompositions_via_dispatchkey``.

    This must be called once before any eager-mode dispatch can reach the Spyre
    kernels (typically from ``_SpyreImpl._lazy_init()``).  It is idempotent:
    subsequent calls are no-ops.

    The ``Library`` objects are stored in module-level globals so they are never
    garbage-collected (and therefore never unregistered from the C++ dispatcher).

    After registration, ``OPWrapper.__call__`` uses ``torch.compiler.is_compiling()``
    to route dispatch: inside a ``torch.compile`` context the Spyre function is called
    directly; outside (eager mode) the pre-compiled wrapper is used.
    """
    global _spyre_autograd_lib, _spyre_lib, _dispatchkey_kernels_registered

    if _dispatchkey_kernels_registered:
        return

    from torch.library import Library, fallthrough_kernel

    _spyre_autograd_lib = Library("aten", "IMPL", "AutogradPrivateUse1")
    _spyre_lib = Library("aten", "IMPL", "PrivateUse1")

    for op, wrapper_cls in spyre_decompositions_via_dispatchkey.items():
        # Autograd key: fall through so that the PrivateUse1 kernel is reached.
        _spyre_autograd_lib.impl(op._name, fallthrough_kernel)
        # PrivateUse1 key: the OPWrapper dispatches to spyre_fn.
        _spyre_lib.impl(op._name, wrapper_cls)

    _dispatchkey_kernels_registered = True


def register_spyre_decompositions_via_dispatchkey(
    ops: Union[torch._ops.OperatorBase, list],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Register decompositions specifically for Spyre device via the PyTorch dispatcher
    This replaces the need for global patching of operations in order to enable them for
    eager mode.
    """

    def decomposition_decorator(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        class OPWrapper:
            def __init__(self, op, spyre_fn):
                self.op = op
                self.spyre_fn = spyre_fn
                # Pre-compile once so that repeated eager-mode calls reuse the
                # same compiled entry point rather than constructing a new
                # torch.compile wrapper on every invocation.
                self._compiled_fn = torch.compile(spyre_fn, dynamic=False)

            def __call__(self, *args, **kwargs):
                # We are about to execute the op on spyre, hence the inputs are expected to be on spyre
                if any(
                    isinstance(x, torch.Tensor)
                    and getattr(x.device, "type", None) != DEVICE_NAME
                    for x in (pytree.tree_leaves(args) + pytree.tree_leaves(kwargs))
                ):
                    args_device = [
                        x.device if isinstance(x, torch.Tensor) else None
                        for x in (pytree.tree_leaves(args) + pytree.tree_leaves(kwargs))
                    ]
                    raise RuntimeError(
                        f"Spyre decomposition function called with inputs being on a different device! Args devices: {args_device=}"
                    )

                # Inside a torch.compile context (make_fx tracing, Inductor
                # lowering, etc.) call the function directly — wrapping it in
                # another torch.compile call would be incorrect.
                if torch.compiler.is_compiling():
                    return self.spyre_fn(*args, **kwargs)
                else:
                    # Eager mode: use the pre-compiled wrapper.
                    return self._compiled_fn(*args, **kwargs)

        def register(op):
            spyre_decompositions_via_dispatchkey[op] = OPWrapper(op, fn)

        # To handle allowing multiple aten_ops at once
        pytree.tree_map_(register, ops)
        return fn

    return decomposition_decorator


# TODO (imaihal): Inductor applies constant folding to torch.full, which allocates
# a one-element Spyre tensor. This currently fails because Spyre does not handle
# single-element tensors well.
# Ref: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/fx_passes/joint_graph.py#L324-L335
#
# Implement ones via identity broadcast: create a size-1 tensor (ones_scalar), expand to
# target size, then clone (identity) to materialize. Clone op with identity is merged.
@register_spyre_decomposition([torch.ops.aten.ones.default])
def ones_decomp(
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[torch.memory_format] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    # Fail loudly rather than silently materialize contiguous strides for a
    # channels-last request (the expand/clone route below is contiguous-only).
    assert memory_format in (
        None,
        torch.contiguous_format,
    ), f"doesn't support memory_format={memory_format}"
    scalar = torch.ops.spyre.ones_scalar(device, dtype=dtype)
    return scalar.reshape(()) if not size else scalar.expand(size).clone()


@register_spyre_decomposition([torch.ops.aten.new_ones.default])
def new_ones_decomp(
    self: torch.Tensor,
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[torch.memory_format] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    assert memory_format in (
        None,
        torch.contiguous_format,
    ), f"doesn't support memory_format={memory_format}"
    dev = device if device is not None else self.device
    dt = dtype if dtype is not None else self.dtype
    scalar = torch.ops.spyre.ones_scalar(dev, dtype=dt)
    return scalar.reshape(()) if not size else scalar.expand(size).clone()


# To avoid constant folding, we introduce a custom op `spyre::full` that runs
# torch.full on CPU and copies the result to Spyre. Remove this workaround once
# Spyre supports one-element tensors.
@register_spyre_decomposition([torch.ops.aten.full])
def full_decomp(
    size: list[Union[int, torch.SymInt]],
    fill_value: torch.types.Number,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    return torch.ops.spyre.full(size, fill_value, device, dtype=dtype)


@register_spyre_decomposition([torch.ops.aten.logical_not])
def logical_not_decomp(input: torch.Tensor) -> torch.Tensor:
    # Currently falling back to torch.zeros_like for dtypes other than bool
    # This is needed until scalar False/0.0 or constant tensor [False]/[0.0] is supported
    if input.dtype is torch.bool:
        zero = torch.ne(input, input)
    else:
        zero = torch.zeros_like(input)
    return torch.eq(input, zero)


@register_spyre_decomposition([torch.ops.aten.addmm.default, torch.ops.aten.addmm.out])
def addmm_decomp(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: Union[int, float] = 1,
    alpha: Union[int, float] = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Decompose addmm into basic operations: out = beta * input + alpha * (mat1 @ mat2)
    """
    # Compute matrix multiplication using matmul to handle batched tensors
    mm_result = mat1 @ mat2

    # Apply alpha scaling if needed
    if alpha != 1:
        mm_result = alpha * mm_result

    # Apply beta scaling and add input if needed
    if beta == 0:
        result = mm_result
    elif beta == 1:
        result = input + mm_result
    else:
        result = beta * input + mm_result

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out

    return result


@register_spyre_decomposition([torch.ops.aten.mean.dim])
def mean_dim_decomp(
    input: torch.Tensor,
    dim,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Decompose mean(dim) into sum(dim) * (1/count) — a MULTIPLY, not a divide.

    Why this decomposition (task #53): on the live torch.compile→Triton path,
    ``aten.mean.dim`` is in ``lowering._TRITON_INCOMPATIBLE_LOWERINGS``, so the
    dedicated ``lower_mean`` (SpyreReduction "mean") does NOT fire. Inductor's
    DEFAULT decomposition splits mean into a ``sum`` reduction (HW-verified) plus a
    trailing ``divf``-by-count kernel — and ``divf`` classifies as
    ``pointwise_unsupported`` in the codegen (no divide datapath). By folding the
    reciprocal at compile time and emitting ``sum * (1/count)`` (an ``arith.mulf``
    by a compile-time constant), the trailing kernel becomes the SCALAR-AFFINE
    ``mul`` subform, which is HW-verified single- and multi-stick (#56, Stages
    1/2/3). Registering this decomp ALSO stops Inductor emitting the
    ``aten.mean.dim`` fallback that otherwise re-dispatches through the eager
    torch.compile wrap → infinite recursion (the "RecursionError" symptom); one
    change closes both failure modes.

    This is a plain-aten decomposition (emits ``sum`` + ``mul``, no ``spyre.*``
    custom op), so it is NOT in ``_TRITON_INCOMPATIBLE_DECOMPOSITIONS`` and DOES
    fire on the Triton path — exactly where the default sum+div split was losing.

    ⚠️ DL16 precision: ``1/count`` is DL16-exact only for power-of-2 ``count``
    (e.g. count=8→0.125, 512→2^-9). For a non-power-of-2 reduced extent the scale
    is inexact and the result carries the scalar-affine round-to-nearest tolerance
    (scalar-affine.md §6c), not ``max_diff=0``. This matches ``torch.mean``'s own
    fp16 behaviour (it also scales by a rounded reciprocal), so it is a precision
    property, not a correctness bug.
    """
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    summed = torch.sum(input, dim=dims, keepdim=keepdim, dtype=dtype)
    size = input.shape
    count = 1
    for d in dims:
        count *= size[d]
    return summed * (1.0 / count)


###############################################################################################
##                           Spyre decompositions for aten ops                               ##
###############################################################################################
# For aten ops, ``register_spyre_decomposition`` automatically registers both a
# decomposition table entry (for compile mode / make_fx) and a PrivateUse1
# dispatch kernel (for eager mode).  The latter is essential for ops with
# CompositeImplicitAutograd (CIA) in upstream PyTorch (e.g. rms_norm,
# layer_norm), and harmless for non-CIA ops (e.g. gelu, softplus).
@register_spyre_decomposition([torch.ops.aten.rms_norm.default])
def spyre_rms_norm(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_rms_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )

    mean = torch.mean(input * input, dim=-1, keepdim=True)
    rsqrt_inp = torch.rsqrt(mean + eps)
    output = input * rsqrt_inp
    if weight is not None:
        output = output * weight
    return output


@register_spyre_decomposition([torch.ops.aten.layer_norm.default])
def spyre_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_layer_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


@register_spyre_decomposition([torch.ops.aten.topk])
def spyre_topk(
    input: torch.Tensor,
    k: int,
    dim: Optional[int] = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k > 4:
        raise Unsupported("Topk is not supported for this config")
    return torch.ops.spyre.topkvalue(input, k, dim), torch.ops.spyre.topkindex(
        input, k, dim
    )


@register_spyre_decomposition([torch.ops.aten.embedding.default])
def spyre_embedding(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """Keep aten.embedding OPAQUE under torch.compile — route it to the custom
    spyre::embedding IBR-gather op instead of letting Inductor decompose it into a
    generic indexed-load (gather) Triton kernel.

    Without this, torch.compile lowers aten.embedding to `tt.load` at a
    per-token-computed pointer (addptr + bounds tt.assert + negative-index select),
    a fused kernel with NO float compute that codegen classifies as
    pointwise_unsupported and fails loud. The custom op (torch_spyre/ops/embedding.py)
    is the HW-verified IBR gather (all target LLM d_models, C=1 and C>=2 multi-chunk);
    routing here makes the compiled path reuse it. register_fake on the custom op
    supplies the output shape for FakeTensor tracing, so the op stays opaque (a fusion
    barrier — correct for bring-up; fusing the gather with its consumer is a later
    optimization).

    NOTE: intentionally NOT added to _TRITON_INCOMPATIBLE_DECOMPOSITIONS. Ops in that
    set (gelu/softplus/layer_norm) are EXCLUDED on the active triton_path so they fall
    back to standard aten; embedding must do the OPPOSITE — survive on the triton path
    so the opaque spyre::embedding op replaces the gather. padding_idx is
    forward-irrelevant (backward-only); the training flags are rejected by the op.
    """
    if scale_grad_by_freq:
        raise NotImplementedError("spyre embedding: scale_grad_by_freq unsupported")
    if sparse:
        raise NotImplementedError("spyre embedding: sparse unsupported")
    return torch.ops.spyre.embedding(weight, indices)


@register_spyre_decomposition([torch.ops.aten._softmax.default])
def spyre_softmax(
    self: torch.Tensor,
    dim: int,
    half_to_float: bool = False,
) -> torch.Tensor:
    """Keep ``aten._softmax`` OPAQUE under torch.compile — route it to the custom
    ``spyre::softmax`` op instead of letting Inductor lower it to a Triton kernel.

    WHY THE GENERIC LOWERING CANNOT WORK HERE, and it is not about fusion: the device
    softmax binary addresses THREE HBM segments (scratch=0, x=1, out=2), but the Triton
    launcher binds ``xlat[i] = tensor_args[i]`` positionally from the kernel's own
    arguments and has no workspace concept, so a two-arg softmax kernel trips the #86
    guard (``declared_segment_count 3 > 2 launched``) — correctly, because the third
    segment would store to unmapped memory. The custom op allocates its own scratch and
    launches all three tensors. The general workspace facility is #113.

    Note this is where Inductor would otherwise DECOMPOSE softmax into
    amax/sub/exp/sum/div and fuse them; ``ttir_analyzer._match_softmax`` exists to
    pattern-match that back, but recovering the op only to be unable to bind its
    scratch does not help. Staying opaque is the shorter path.

    CONSEQUENCE: softmax becomes a fusion barrier (correct for bring-up; a flash-style
    attention kernel needs an ONLINE single-pass softmax, which is #114 — a separate
    op, not a relaxation of this one).

    Intentionally NOT in ``_TRITON_INCOMPATIBLE_DECOMPOSITIONS``: like embedding, this
    must SURVIVE on the triton path so the opaque op replaces the generic lowering.
    Ops in that set do the opposite (they fall back to standard aten).

    ENVELOPE: admission is decided in ``torch_spyre/ops/softmax.py`` (this
    decomposition adds no shape check of its own) — see that module's docstring
    for the current witnessed-shape summary and its own pointer to
    ``sentient_codegen.encoder.softmax_dispatch.SOFTMAX_WITNESSED_SHAPES``, the
    actual source of truth.  Everything outside that set raises
    ``SoftmaxEnvelopeError`` and the error PROPAGATES (no CPU fallback, #70).
    """
    if half_to_float:
        raise NotImplementedError(
            "spyre softmax: half_to_float=True (fp32 output from a half input) is "
            "unsupported — the device kernel produces DL16.")
    return torch.ops.spyre.softmax(self, dim)


@register_spyre_decomposition([torch.ops.aten.gelu.default])
def spyre_gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    return torch.ops.spyre.gelu(input, approximate)


@register_spyre_decomposition([torch.ops.aten.softplus.default])
def spyre_softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    return torch.ops.spyre.softplus(input, beta, threshold)


@register_spyre_decomposition([torch.ops.aten.linear.default])
def spyre_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    weight = weight.transpose(-1, -2)
    while weight.dim() < input.dim():
        weight = torch.unsqueeze(weight, 0)
    out = input @ weight
    if bias is not None:
        out = out + bias
    return out


@register_spyre_decomposition(
    [torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default]
)
def spyre__sdpa_overrideable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = query.size(0)
    num_heads = query.size(1)
    num_kvheads = key.size(1)
    max_seqlen_q = query.size(2)
    max_seqlen_kv = key.size(2)

    query = query.clone(memory_format=torch.contiguous_format)
    key = key.clone(memory_format=torch.contiguous_format)
    value = value.clone(memory_format=torch.contiguous_format)

    scaling_factor = scale
    if scaling_factor is None:
        scaling_factor = 1.0 / math.sqrt(query.shape[-1])
    scaling_factor = math.sqrt(scaling_factor)

    query = query * scaling_factor
    key = key * scaling_factor

    expansion = num_heads // num_kvheads
    if expansion != 1:
        key = key.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        value = value.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
    key_t = key.transpose(-2, -1).clone(memory_format=torch.contiguous_format)

    attn = torch.matmul(query, key_t)

    if is_causal:
        assert attn_bias is None
        attn_bias = torch.full_like(attn, float("-inf"))
        attn_bias = attn_bias.triu(diagonal=1)

    if attn_bias is not None:
        attn = attn + attn_bias

    # TODO (aviros): Switch to _safe_softmax
    attn = torch.softmax(attn, -1)

    if dropout_p > 0.0:
        # TODO(aviros): Implement
        raise Unsupported("Attention dropout not implemented for Spyre")

    # Unused for now
    logsumexp = torch.empty(
        (batch_size, num_heads, max_seqlen_q), dtype=torch.float32, device="spyre"
    )
    philox_seed = torch.empty((1,), dtype=torch.float16, device="spyre")
    philox_offset = torch.empty((1,), dtype=torch.float16, device="spyre")

    # B, H, S, E
    out = torch.matmul(attn, value)

    # B, S, H, E
    # This is needed to maintain the API promise from SDPA (attn needs to have same size+stride as q)
    out = out.transpose(1, 2).clone(memory_format=torch.contiguous_format)

    # Returns (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
    return (
        out.transpose(1, 2),
        logsumexp,
        None,
        None,
        max_seqlen_q,
        max_seqlen_kv,
        philox_seed,
        philox_offset,
        None,
    )


## TODO(imaihal): Need to fix scalar tensor shape mismatch during Spyre-to-CPU transfer.
## See: https://github.com/torch-spyre/torch-spyre/issues/1172
## This will be enabled after solving this.
# @register_spyre_decomposition([torch.ops.aten.max.default])
# def spyre_max_default_decomp(input):
#    """
#    Decompose torch.max(input) with conditional CPU fallback for int64.
#
#    For int64 tensors, use custom op spyre::max_default_int64_fallback which has
#    a CPU fallback registered in fallbacks.py.
#    For other dtypes (float16, float32, etc.), use amax.
#    """
#    if input.dtype == torch.int64:
#        # Use custom op with CPU fallback to avoid recursive decomposition
#        # Returns a scalar (0D) tensor
#        return torch.ops.spyre.max_default_int64_fallback(input)
#    else:
#        # Use amax for supported dtypes (can run on Spyre)
#        # Returns a scalar (0D) tensor
#        return torch.ops.aten.amax(input)


@register_spyre_decomposition([torch.ops.aten.max.dim])
def spyre_max_dim_decomp(input, dim, keepdim=False):
    """
    Decompose torch.max(input, dim) with conditional CPU fallback for int64.

    For int64 tensors, use custom op spyre::max_dim_int64_fallback which has
    a CPU fallback registered in fallbacks.py.
    For other dtypes (float16, float32, etc.), decompose into amax and argmax operations.

    Returns a named tuple (values, indices) as expected by torch.max.

    # TODO (imaihal): Decomposed into torch.topk with k=1 to obtain both values and indices,
    #  or implement argmax in the backend compiler to get indices
    """
    if input.dtype == torch.int64:
        # Use custom op with CPU fallback to avoid recursive decomposition
        return torch.ops.spyre.max_dim_int64_fallback(input, dim, keepdim)
    else:
        # Use amax and argmax for supported dtypes (can run on Spyre)
        values = torch.ops.aten.amax(input, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmax(input, dim=dim, keepdim=keepdim)
        return torch.return_types.max((values, indices))


@register_spyre_decomposition([torch.ops.aten.cat.default])
def decompose_cat(
    tensors: list[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    orig_decomp = torch._inductor.decomposition.cat(tensors, dim)
    if orig_decomp == NotImplemented:
        expanded_size = 0
        for t in tensors:
            expanded_size += t.size(dim)
        output_size = list(tensors[0].size())
        output_size[dim] = expanded_size
        output = tensors[0].new_empty(output_size)
        offset = 0
        for input in tensors:
            output = torch.ops.spyre.overwrite_f(
                input=input, output=output, dims=[dim], offsets=[offset]
            )
            offset += input.size(dim)
        return output
    else:
        return orig_decomp


@register_spyre_decomposition([torch.ops.aten.constant_pad_nd.default])
def pad_decomp(
    input: torch.Tensor,
    pad: list[int],
    value: float = 0,
) -> torch.Tensor:
    # pad is in reverse dim order: (left_last, right_last, left_2nd_last, right_2nd_last, ...)
    n_dims_padded = len(pad) // 2

    # Negative pad values (cropping) require reading from a non-zero storage
    # offset or a sub-stick position, neither of which the SFP supports.
    if any(p < 0 for p in pad):
        raise Unsupported(
            f"constant_pad_nd: negative padding (cropping) is not supported on "
            f"Spyre (pad={pad})"
        )

    # Left-padding on the last (stick) dimension shifts the output start address
    # by `left` elements. The hardware can only express this in whole sticks, so
    # `left` must be a multiple of the stick size (64 fp16 elements).
    # Sub-stick left-padding on the last dimension is tracked in:
    # https://github.com/torch-spyre/torch-spyre/issues/1464
    last_dim_left = pad[0]
    if last_dim_left > 0:
        elems_per_stick = 128 // input.element_size()
        if last_dim_left % elems_per_stick != 0:
            raise Unsupported(
                f"constant_pad_nd: sub-stick left-padding on the last dimension is "
                f"not supported on Spyre (pad={pad}, left={last_dim_left}, "
                f"stick_size={elems_per_stick})"
            )

    # Build the padded output shape and collect which dimensions need padding.
    scalar = torch.ops.spyre.full([1], value, input.device, dtype=input.dtype)
    output_size = list(input.size())
    dims: list[int] = []
    offsets: list[int] = []
    for i in range(n_dims_padded - 1, -1, -1):
        left = pad[2 * i]
        right = pad[2 * i + 1]
        if left + right == 0:
            continue
        dim = input.dim() - 1 - i
        output_size[dim] += left + right
        dims.append(dim)
        offsets.append(left)

    if not dims:
        return input

    output = scalar.expand(output_size).clone()
    output = torch.ops.spyre.overwrite_f(
        input=input, output=output, dims=dims, offsets=offsets
    )
    return output


def _conv2d_one_by_one_core(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
) -> torch.Tensor:
    """Alias a gated 1x1 convolution to a matmul (+ separate bias add).

    The hardware has no convolution datapath; a 1x1/s1/p0/d1/groups=1
    convolution is a pure per-spatial-position channel mix — exactly a
    matmul over the channel dimension — so it rewrites to ops the
    silicon-witnessed native-matmul route already handles (the aten.mm
    exclusion below -> triton.native_matmul -> tt.dot):

        out = weight.view(O, C) @ input.view(N, C, H*W)   # (N,O,H*W)
        out = out.view(N, O, H, W)
        out = out + bias.view(1, O, 1, 1)   # SEPARATE pointwise add

    The bias stays a separate pointwise add by design: the matmul_bias
    composite admission is witnessed only at the census geometry
    (M=32/K=64/N=64), and widening that witness is a deliberate,
    hardware-probed decision — not something this alias makes silently.

    Everything outside the gate raises ``Unsupported`` (refuse-loud —
    no CPU fallback, no extern kernel, no silent contiguity assumption):
    strided/padded/dilated/grouped/transposed 1x1, any kernel > 1,
    conv1d/conv3d inputs (``aten.convolution`` is the funnel for all
    three), sub-stick channel extents, mismatched dtypes between
    input/weight/bias, non-1D or wrong-length bias, and non-contiguous
    input/weight.

    ATen parameter-list shorthand: a traced RAW op call
    (``torch.ops.aten.conv2d(x, w, None, [1], [0], [1], 1)``) delivers
    the stride/padding/dilation lists exactly as written, so a len-1
    list can reach this decomposition un-normalized (eager ``F.conv2d``
    normalizes via CompositeImplicitAutograd, the traced path does
    not). Len-1 lists are canonicalized to the spatial rank (2) BEFORE
    the equality gate so the ATen shorthand does not false-refuse;
    any other wrong length refuses loud naming the observed value.

    Layout contract (NCHW):
      - Stick alignment: the channel extent C must be a multiple of the
        stick size in elements (128B / element_size — 64 elements at
        DL16, 32 at fp32); a sub-stick C would put the matmul's K
        dimension off stick boundaries.
      - Contiguity disposition: dense NCHW/OIHW tensors fresh from
        allocation are contiguous (the reachable case, pinned by test);
        a non-contiguous input (slice/permute) is REFUSED with the
        observed strides named — ``.contiguous()`` is never called
        silently because it would both pay a hidden copy and hide the
        layout from the compiler.

    ``aten.convolution`` and ``aten.conv2d`` both route here: the
    compile path funnels conv2d -> convolution, while eager dispatch and
    pre-decomposition traces can present either overload.
    """
    if transposed:
        raise Unsupported(
            f"conv2d 1x1 alias: transposed convolution (transposed="
            f"{transposed}) — only the 1x1/stride-1/no-padding/no-dilation/"
            f"groups-1 form aliases to matmul")
    if groups != 1:
        raise Unsupported(
            f"conv2d 1x1 alias: grouped convolution (groups={groups}) — "
            f"only groups=1 aliases to matmul")
    # Rank check BEFORE the parameter gates: a conv1d/conv3d-shaped input
    # must be told it is the wrong rank, not get a stride refusal naming a
    # rank-3 list it legitimately carries.
    if input.dim() != 4 or weight.dim() != 4:
        raise Unsupported(
            f"conv2d 1x1 alias: only 4D NCHW convolution aliases to "
            f"matmul, got input.dim()={input.dim()}, "
            f"weight.dim()={weight.dim()} (conv1d/conv3d are refused)")
    # Canonicalize the ATen len-1 shorthand ([1]/[0]/[1] == [1, 1]/[0, 0]/
    # [1, 1] at spatial rank 2) BEFORE the equality gate. Any other wrong
    # length refuses loud rather than reaching the gate with a shape it
    # cannot honestly compare.
    def _spatial_list(values, name):
        _values = [int(v) for v in values]
        if len(_values) == 1:
            _values = _values * 2
        if len(_values) != 2:
            raise Unsupported(
                f"conv2d 1x1 alias: {name}={list(values)} has length "
                f"{len(list(values))} — a 4D convolution takes exactly 2 "
                f"values per parameter (the ATen len-1 shorthand is "
                f"canonicalized; conv1d/conv3d are refused)")
        return _values

    _stride = _spatial_list(stride, "stride")
    _padding = _spatial_list(padding, "padding")
    _dilation = _spatial_list(dilation, "dilation")
    _output_padding = _spatial_list(output_padding, "output_padding")
    if _stride != [1, 1] or _padding != [0, 0] or _dilation != [1, 1]:
        raise Unsupported(
            f"conv2d 1x1 alias: stride={_stride}, padding={_padding}, "
            f"dilation={_dilation} — only stride=(1, 1)/padding=(0, 0)/"
            f"dilation=(1, 1) aliases to matmul")
    if any(p != 0 for p in _output_padding):
        raise Unsupported(
            f"conv2d 1x1 alias: output_padding={_output_padding} is only "
            f"meaningful for transposed convolution, which is refused")
    out_channels, in_channels, kh, kw = weight.shape
    if kh != 1 or kw != 1:
        raise Unsupported(
            f"conv2d 1x1 alias: kernel ({kh}, {kw}) — only 1x1 aliases "
            f"directly to matmul (general kernels route through "
            f"_conv2d_im2col_core; reaching this arm with k>1 means a "
            f"direct call bypassing the _conv2d_core router)")
    n, c_in, h, w = input.shape
    if c_in != in_channels:
        raise Unsupported(
            f"conv2d 1x1 alias: input channels {c_in} != weight input "
            f"channels {in_channels}")
    elems_per_stick = 128 // input.element_size()
    if in_channels % elems_per_stick != 0:
        raise Unsupported(
            f"conv2d 1x1 alias: channel extent C={in_channels} is not "
            f"stick-aligned ({elems_per_stick} elements per "
            f"{input.dtype} stick) — the matmul K dimension would sit off "
            f"stick boundaries")
    if not input.is_contiguous():
        raise Unsupported(
            f"conv2d 1x1 alias: input is not dense-contiguous NCHW "
            f"(stride={tuple(input.stride())}, shape={tuple(input.shape)}) "
            f"— non-contiguous layouts are refused, never silently copied")
    if not weight.is_contiguous():
        raise Unsupported(
            f"conv2d 1x1 alias: weight is not dense-contiguous OIHW "
            f"(stride={tuple(weight.stride())}, shape={tuple(weight.shape)}) "
            f"— non-contiguous layouts are refused, never silently copied")
    # Dtype uniformity: plain tensor ops would silently PROMOTE a mixed
    # dtype (e.g. an f64 bias over fp32 tensors) where the conv2d
    # reference refuses — the alias must match the reference, refuse
    # loud naming both dtypes.
    if weight.dtype != input.dtype:
        raise Unsupported(
            f"conv2d 1x1 alias: weight dtype {weight.dtype} != input "
            f"dtype {input.dtype} — mixed dtypes are refused, never "
            f"silently promoted")
    # Dtype IDENTITY, not just uniformity: torch_spyre maps bfloat16 to
    # DataFormats.BFLOAT16 (torch_spyre/_C.py:113) — the 1p5 format. On a
    # 1p0 target a bf16 tensor's bits are consumed as DL16 (silent-wrong),
    # and nothing downstream refuses the mapping, so this layer must.
    # Same seam precedent as torch_spyre/ops/softmax.py's dtype refusal.
    if input.dtype is torch.bfloat16:
        raise Unsupported(
            f"conv2d 1x1 alias: dtype {input.dtype} unsupported (1p0 is "
            f"DL16, i.e. torch.float16; bfloat16 is 1p5)")
    if bias is not None:
        if bias.dim() != 1:
            raise Unsupported(
                f"conv2d 1x1 alias: bias must be 1-dimensional of length "
                f"{out_channels}, got shape {tuple(bias.shape)} — matching "
                f"the conv2d reference, which refuses non-1D bias")
        if bias.dtype != input.dtype:
            raise Unsupported(
                f"conv2d 1x1 alias: bias dtype {bias.dtype} != input "
                f"dtype {input.dtype} — mixed dtypes are refused, never "
                f"silently promoted")
        if bias.numel() != out_channels:
            raise Unsupported(
                f"conv2d 1x1 alias: bias has length {bias.numel()}, "
                f"expected {out_channels} (one per output channel)")

    # The gated rewrite. The 2D weight sits on the LEFT of the batched
    # matmul (matmul((O, C), (N, C, P)) broadcasts the weight over the
    # batch — the same stride-0-batch bmm class as the witnessed linear
    # path's (B, S, d) @ (d, d)). .view() (not .reshape()) so any layout
    # surprise raises instead of silently copying; the contiguity gates
    # above are what make the views legal.
    spatial = h * w
    out = weight.view(out_channels, in_channels) @ input.view(
        n, in_channels, spatial)
    out = out.view(n, out_channels, h, w)
    if bias is not None:
        # Separate pointwise add (default D-disposition) — never fold
        # into a matmul_bias composite here.
        out = out + bias.view(1, out_channels, 1, 1)
    return out


def _conv2d_im2col_core(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
) -> torch.Tensor:
    """Rewrite a gated general convolution to HOST im2col + matmul
    (+ separate bias add).

    The hardware has no convolution datapath, and the device-side
    gather is vetoed twice over: L3 EAR strides are stick-units (128B),
    so a 1-2 element sliding-window hop is inexpressible in the L3->LX
    path, and the analyzer's dot-operand provenance guard refuses
    gathered dot operands by design. The im2col (unfold) therefore runs
    on HOST — a documented disposition, not a silent CPU fallback of a
    device-capable op — and only the matmul rides the silicon-witnessed
    native-matmul route:

        cols = im2col(x, (kh, kw), padding=(ph, pw))   # (N, K, L)
        cols = pad(cols, K -> K_pad)                    # whole-stick K
        w2   = pad(weight.view(O, K), K -> K_pad)       # zero-pad bill
        out  = w2 @ cols                                # (N, O, L)
        out  = out.view(N, O, H_out, W_out)
        out  = out + bias.view(1, O, 1, 1)              # SEPARATE add

    The K zero-pad is structural zeros in BOTH the weight rows and the
    im2col columns, so the matmul is numerically exact; the bill
    (K_pad - K elements) is the im2col amplification term on record in
    the conv pathway catalog.

    ENVELOPE (fail-loud BEFORE any tensor is built — this seam owns
    the refusal, it is never silently inherited from the matmul
    route's downstream shape checks): ks >= 9 refuses (single-ImpSync
    clean-K caps at ks=8; the chunked-K mechanism is not landed), and
    ks*ns > 127 refuses (14-bit L3 LBR single-kernel fit bound — the
    matmul route would NOT refuse this). The bounds live in
    ``sentient_codegen.scheduler.conv_predicate`` (single source of
    truth); this decomposition delegates to it and refuses loud if
    that authority is unreachable (never silently skips the check).

    Gate (v0): stride=1, dilation=1, groups=1, not transposed,
    padding 0 or kernel//2 per dim (symmetric-valid), 4D NCHW,
    dense-contiguous input/weight, a window that fits at least once.
    Everything else raises ``Unsupported``.
    """
    if transposed:
        raise Unsupported(
            f"conv2d im2col: transposed convolution (transposed="
            f"{transposed}) — only the stride-1/no-dilation/groups-1 "
            f"form rewrites to host im2col + matmul")
    if groups != 1:
        raise Unsupported(
            f"conv2d im2col: grouped convolution (groups={groups}) — "
            f"only groups=1 rewrites to host im2col + matmul")
    # Rank check BEFORE the parameter gates (the fold's rank-first
    # rule): a conv1d/conv3d-shaped input must be told it is the wrong
    # rank, not get a stride refusal naming the rank-3 list it
    # legitimately carries.
    if input.dim() != 4 or weight.dim() != 4:
        raise Unsupported(
            f"conv2d im2col: only 4D NCHW convolution rewrites to host "
            f"im2col + matmul, got input.dim()={input.dim()}, "
            f"weight.dim()={weight.dim()} (conv1d/conv3d are refused)")
    # Canonicalize the ATen len-1 shorthand ([1]/[0]/[1] == [1, 1]/
    # [0, 0]/[1, 1] at spatial rank 2) BEFORE the equality gate — a
    # traced RAW op call delivers the lists un-normalized (eager
    # F.conv2d normalizes via CompositeImplicitAutograd, the traced
    # path does not). Any other wrong length refuses loud naming the
    # observed value (same fold convention as the 1x1 alias).
    def _spatial_list(values, name):
        _values = [int(v) for v in values]
        if len(_values) == 1:
            _values = _values * 2
        if len(_values) != 2:
            raise Unsupported(
                f"conv2d im2col: {name}={list(values)} has length "
                f"{len(list(values))} — a 4D convolution takes exactly 2 "
                f"values per parameter (the ATen len-1 shorthand is "
                f"canonicalized; conv1d/conv3d are refused)")
        return _values

    _stride = _spatial_list(stride, "stride")
    _padding = _spatial_list(padding, "padding")
    _dilation = _spatial_list(dilation, "dilation")
    _output_padding = _spatial_list(output_padding, "output_padding")
    if _stride != [1, 1] or _dilation != [1, 1]:
        raise Unsupported(
            f"conv2d im2col: stride={_stride}, dilation={_dilation} — "
            f"only stride=(1, 1)/dilation=(1, 1) rewrites to host "
            f"im2col + matmul (strided and dilated convolution are out "
            f"of v0 scope by tier row)")
    if any(p != 0 for p in _output_padding):
        raise Unsupported(
            f"conv2d im2col: output_padding={_output_padding} is only "
            f"meaningful for transposed convolution, which is refused")
    out_channels, in_channels, kh, kw = weight.shape
    n, c_in, h, w = input.shape
    if c_in != in_channels:
        raise Unsupported(
            f"conv2d im2col: input channels {c_in} != weight input "
            f"channels {in_channels}")
    # Per-dim symmetric-valid padding: 0 or kernel//2 (the "same"
    # extent for odd kernels; 3x3 -> p=1, 5x5 -> p=2). Anything else —
    # including asymmetric or arbitrary padding — refuses loud.
    if _padding != [0, 0] and (
            _padding[0] not in (0, kh // 2)
            or _padding[1] not in (0, kw // 2)):
        raise Unsupported(
            f"conv2d im2col: padding={_padding} — only padding=0 or "
            f"padding=kernel//2 per dim (kernel=({kh}, {kw})) rewrites "
            f"to host im2col + matmul; arbitrary or asymmetric padding "
            f"is out of v0 scope by tier row")
    h_out = h + 2 * _padding[0] - kh + 1
    w_out = w + 2 * _padding[1] - kw + 1
    if h_out < 1 or w_out < 1:
        raise Unsupported(
            f"conv2d im2col: kernel ({kh}, {kw}) over input extent "
            f"({h}, {w}) with padding={_padding} leaves a degenerate "
            f"window (output extent ({h_out}, {w_out})) — the window "
            f"must fit at least once")
    if not input.is_contiguous():
        raise Unsupported(
            f"conv2d im2col: input is not dense-contiguous NCHW "
            f"(stride={tuple(input.stride())}, shape={tuple(input.shape)}) "
            f"— non-contiguous layouts are refused, never silently copied")
    if not weight.is_contiguous():
        raise Unsupported(
            f"conv2d im2col: weight is not dense-contiguous OIHW "
            f"(stride={tuple(weight.stride())}, shape={tuple(weight.shape)}) "
            f"— non-contiguous layouts are refused, never silently copied")
    # Dtype uniformity and bias shape (the fold conventions, same
    # contract as the 1x1 alias): plain tensor ops would silently
    # PROMOTE a mixed dtype where the conv2d reference refuses, and a
    # non-1D or wrong-length bias must refuse named instead of dying
    # inside the .view() or being leniently accepted.
    if weight.dtype != input.dtype:
        raise Unsupported(
            f"conv2d im2col: weight dtype {weight.dtype} != input "
            f"dtype {input.dtype} — mixed dtypes are refused, never "
            f"silently promoted")
    if bias is not None:
        if bias.dim() != 1:
            raise Unsupported(
                f"conv2d im2col: bias must be 1-dimensional of length "
                f"{out_channels}, got shape {tuple(bias.shape)} — matching "
                f"the conv2d reference, which refuses non-1D bias")
        if bias.dtype != input.dtype:
            raise Unsupported(
                f"conv2d im2col: bias dtype {bias.dtype} != input "
                f"dtype {input.dtype} — mixed dtypes are refused, never "
                f"silently promoted")
        if bias.numel() != out_channels:
            raise Unsupported(
                f"conv2d im2col: bias has length {bias.numel()}, "
                f"expected {out_channels} (one per output channel)")
        if not bias.is_contiguous():
            raise Unsupported(
                f"conv2d im2col: bias is non-contiguous (strides "
                f"{tuple(bias.stride())} for shape "
                f"{tuple(bias.shape)}) — matching the input/weight "
                f"contiguity contract; the decomposition never "
                f"silently .contiguous()-copies")

    # The envelope boundary: fail-loud BEFORE any tensor is built.
    # Delegated to the codegen-side predicate (single source of truth,
    # same delegation shape as softmax's witnessed-shape set); an
    # unreachable authority refuses loud, never silently skips.
    try:
        from sentient_codegen.scheduler.conv_predicate import (
            UnsupportedConvIm2colError,
            assert_conv_im2col_envelope,
        )
    except ImportError as exc:
        raise Unsupported(
            f"conv2d im2col: the envelope admission predicate "
            f"(sentient_codegen.scheduler.conv_predicate) is not "
            f"importable — the envelope check is never silently "
            f"skipped, so the route refuses (source env.sh so codegen "
            f"is on the path)") from exc
    try:
        # dtype IDENTITY, not element width: 16 bits cannot distinguish
        # 1p0's DL16 (torch.float16) from 1p5's bf16, and the predicate
        # is the authority that refuses the latter NAMED.
        assert_conv_im2col_envelope(
            in_channels=in_channels, kh=kh, kw=kw,
            out_channels=out_channels, dtype=input.dtype)
    # Typed catch ONLY: a bug inside the predicate must surface as
    # itself, never masquerade as an envelope refusal.
    except UnsupportedConvIm2colError as exc:
        raise Unsupported(f"conv2d im2col envelope: {exc}") from exc

    # The gated rewrite. im2col's column ordering (channel-major, then
    # kernel row, then kernel column) matches weight.view(O, K)'s
    # (O, C, kh, kw) -> (O, C*kh*kw) flattening exactly, so the mm IS
    # the convolution. K is zero-padded to a whole stick in BOTH
    # operands (structural zeros — numerically exact; the bill is
    # K_pad - K elements per weight row and per im2col column); the 2D
    # weight sits on the LEFT of the broadcast batched matmul (the same
    # stride-0-batch bmm class as the 1x1 alias and the witnessed
    # linear path). .view() (not .reshape()) so any layout surprise
    # raises instead of silently copying; the contiguity gates above
    # are what make the views legal.
    elems_per_stick = 128 // input.element_size()
    k = in_channels * kh * kw
    k_padded = ((k + elems_per_stick - 1) // elems_per_stick) \
        * elems_per_stick
    cols = torch.nn.functional.unfold(
        input, (kh, kw), padding=(_padding[0], _padding[1]))
    if k_padded != k:
        # Zero-pad K by appending structural-zero columns to BOTH
        # operands. Built with new_zeros + slice-copy rather than
        # aten.constant_pad_nd deliberately: the Spyre pad
        # decomposition routes through spyre::overwrite_f (a device op
        # with no CPU kernel), which would make every offline trace of
        # this decomposition on CPU tensors crash; the slice-copy form
        # is backend-neutral, lowers to the same materialization under
        # Inductor, and keeps the offline make_fx receipt honest
        # against the unmodified production decomposition table.
        n_patches = cols.shape[2]
        cols_padded = cols.new_zeros((n, k_padded, n_patches))
        cols_padded[:, :k] = cols
        w2 = weight.view(out_channels, k)
        w2_padded = w2.new_zeros((out_channels, k_padded))
        w2_padded[:, :k] = w2
        cols, w2 = cols_padded, w2_padded
    else:
        w2 = weight.view(out_channels, k)
    out = w2 @ cols
    out = out.view(n, out_channels, h_out, w_out)
    if bias is not None:
        # Separate pointwise add (default D-disposition) — never fold
        # into a matmul_bias composite here.
        out = out + bias.view(1, out_channels, 1, 1)
    return out


def _conv2d_core(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
) -> torch.Tensor:
    """Route a convolution by kernel extent: the 1x1 form keeps its
    cheaper direct-matmul alias (no unfold), every other kernel
    rewrites to host im2col + matmul. Both cores refuse loud on
    everything outside their gates."""
    if weight.dim() == 4 and weight.shape[2] == 1 and weight.shape[3] == 1:
        return _conv2d_one_by_one_core(
            input, weight, bias, stride, padding, dilation, transposed,
            output_padding, groups)
    return _conv2d_im2col_core(
        input, weight, bias, stride, padding, dilation, transposed,
        output_padding, groups)


@register_spyre_decomposition([torch.ops.aten.convolution.default])
def conv2d_decomp(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    transposed: bool = False,
    output_padding: Sequence[int] = (0, 0),
    groups: int = 1,
) -> torch.Tensor:
    """``aten.convolution``-signature wrapper (the compile-path funnel
    for conv1d/conv2d/conv3d): all nine arguments positional. See
    ``_conv2d_core`` for the kernel-extent routing, the gated 1x1 ->
    matmul alias, and the general-conv host im2col rewrite."""
    return _conv2d_core(
        input, weight, bias, stride, padding, dilation, transposed,
        output_padding, groups)


@register_spyre_decomposition([torch.ops.aten.conv2d.default])
def conv2d_alias_decomp(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """``aten.conv2d``-signature wrapper (no transposed/output_padding
    — a different positional order than ``aten.convolution``, hence a
    separate entry point): eager dispatch and pre-decomposition traces
    can present this overload directly. Same gated core."""
    return _conv2d_core(
        input, weight, bias, stride, padding, dilation, False, (0, 0),
        groups)


@register_spyre_decomposition([torch.ops.aten.bitwise_not])
def bitwise_not(input: torch.Tensor) -> torch.Tensor:
    if input.dtype is torch.bool:
        return torch.logical_not(input)
    else:
        neg_one = torch.ops.aten.full_like(input, -1)
        return torch.ops.aten.bitwise_xor(input, neg_one)


@register_spyre_decomposition([torch.ops.aten.bitwise_and])
def bitwise_and(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    if input1.dtype is torch.bool and input2.dtype is torch.bool:
        return torch.ops.aten.logical_and(input1, input2)
    else:
        return torch.ops.aten.bitwise_not(
            torch.ops.aten.bitwise_or(
                torch.ops.aten.bitwise_not(input1), torch.ops.aten.bitwise_not(input2)
            )
        )


###############################################################################################
##                           Register custom kernels for Spyre.                              ##
###############################################################################################
# Kernels are registered permanently in the C++ dispatcher by
# ``_register_spyre_dispatchkey_kernels_permanently()`` (idempotent).
# Once registered, ``OPWrapper.__call__`` uses ``torch.compiler.is_compiling()``
# to route dispatch: inside a ``torch.compile`` context the Spyre function is
# called directly; outside (eager mode) the pre-compiled wrapper is used.
# Note: This has to stay at the end of the file.
_register_spyre_dispatchkey_kernels_permanently()
