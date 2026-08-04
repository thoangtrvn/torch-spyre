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
"""Device softmax(dim=-1) — the routed attention softmax (#112).

WHY THIS IS A CUSTOM OP AND NOT AN INDUCTOR LOWERING
----------------------------------------------------
The generic (Inductor -> Triton -> TTIR -> codegen) path CANNOT bind this kernel's
tensors.  The Triton launcher binds ``xlat[i] = tensor_args[i]`` positionally from the
*kernel's own* arguments and has no workspace concept
(``triton-spyre`` ``third_party/spyre/backend/driver.py``), while this binary
addresses THREE HBM segments:

    launch_kernel_from_bytes(binary, [scratch, x, out])
        segment 0 = scratch   three ping-ponging intermediate buffers
        segment 1 = x         the activation
        segment 2 = out       the result

A Triton softmax kernel has two tensor args, so ``declared_segment_count (3) >
len(tensor_args) (2)`` and the #86 guard refuses — correctly, since the third segment
would otherwise store to unmapped memory (silent 0x7b1b).  ``hbm_scratch_bytes`` does
not help: it is specifically *cross-core partial-result* scratch and nothing allocates
it today.  Building the general workspace facility is #113.

So softmax takes embedding's route: an opaque custom op that allocates its own
auxiliary buffer and launches all three tensors itself.  Consequence, stated plainly:
softmax is a FUSION BARRIER.  That costs nothing reachable today — Spyre's SDPA is
already decomposed to ``matmul -> softmax(dim=-1) -> matmul`` with the score matrix
materialised (``_inductor/decompositions.py`` ``spyre__sdpa_overrideable``), and a
flash-style kernel would need an ONLINE (single-pass) softmax, which this two-pass
five-packet kernel is not.  That is #114, a separate op, not a change here.

THE 3-TENSOR POSITIONAL BINDING IS THE VERIFIED ONE
---------------------------------------------------
``[scratch, x, out]`` is exactly the placement ``softmax_packet_placement`` derives and
the full-binary byte-match locks, and exactly what the silicon KAT
(``scripts/softmax_dim_neg1_hw_kat.py``) launched to get 3.21-3.91 ULP_DL16 with zero
outliers.  Reusing it unchanged is what makes that evidence transfer to this path
instead of becoming a new unknown.

ENVELOPE: ONE SHAPE, FAIL-LOUD
------------------------------
Only ``M=1024, N=64, C=32`` DL16 is admitted, enforced in codegen by
``SOFTMAX_WITNESSED_SHAPES``.  Everything else raises ``SoftmaxEnvelopeError`` (a
``NotImplementedError``, so it PROPAGATES and fails the compile).  There is
deliberately no CPU fallback (#70): a silent host softmax is worse than a hard error.
This is NOT servable attention yet — ``[1, seq]`` decode and ``[seq, seq]`` prefill are
both refused.  Widening needs a golden or an HW-KAT, never a relaxed guard.
"""
from __future__ import annotations

import torch

from torch_spyre._C import launch_kernel_from_bytes

__all__ = ["softmax", "SoftmaxHostError"]

#: Element width per torch dtype.  1p0 is DL16 (1-6-9) — NEVER "FP16", and never
#: bfloat16 (that is 1p5/BF16).  Extend per generation, not per convenience.
_DTYPE_BITS = {torch.float16: 16}

#: Rows per core in the witnessed envelope: the plan's `M == 32*C` law means every core
#: owns exactly 32 rows.  Used to DERIVE the core count from M rather than hardcoding
#: 32 — the shape pin in codegen refuses anything this does not resolve correctly.
_ROWS_PER_CORE = 32


class SoftmaxHostError(RuntimeError):
    """Raised when sentient_codegen is unreachable, so no binary can be built."""


def _resolve_num_cores(m: int) -> int:
    """Core count from the envelope law ``M == 32*C`` (so ``C = M/32``).

    Clamped to >= 1 only so a tiny M produces a clean envelope refusal downstream
    rather than a confusing division-by-zero inside the plan.  This is NOT a widening:
    the witnessed-shape pin in ``softmax_dispatch`` still has the final say.
    """
    return max(1, m // _ROWS_PER_CORE)


def _build_launch_plan(m: int, n: int, num_cores: int, dtype_bits: int):
    """Fetch the cached host-side launch plan from codegen.

    Imported lazily and guarded the same way ``_embedding_host`` guards its codegen
    import, so a missing PYTHONPATH is a clear message rather than an ImportError deep
    inside a torch.compile trace.
    """
    try:
        from sentient_codegen.encoder.softmax_dispatch import (
            build_softmax_launch_plan,
        )
    except ImportError as e:
        raise SoftmaxHostError(
            "sentient_codegen not importable — source env.sh so codegen is on "
            "PYTHONPATH (required to build the softmax binary)."
        ) from e

    return build_softmax_launch_plan(
        m, n, num_cores=num_cores, dtype_bits=dtype_bits)


@torch.library.custom_op("spyre::softmax", mutates_args=(), device_types="spyre")
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """softmax over the trailing dimension of a 2-D ``[M, N]`` DL16 activation."""
    if x.device.type != "spyre":
        raise ValueError(f"spyre softmax: input must be on spyre, got {x.device}")
    if x.dtype not in _DTYPE_BITS:
        raise NotImplementedError(
            f"spyre softmax: dtype {x.dtype} unsupported (1p0 is DL16, i.e. "
            f"torch.float16; bfloat16 is 1p5)")
    # 2-D ONLY, deliberately. Flattening leading dims of a contiguous [B,H,S,S] into
    # [B*H*S, S] is arithmetically exact for a row-wise softmax, but the DEVICE stick
    # layout of a 4-D tensor is not verified to equal that of the flattened 2-D one,
    # and this op must not assume a layout it has not seen. Refuse instead of guessing.
    if x.dim() != 2:
        raise NotImplementedError(
            f"spyre softmax: only 2-D [M, N] is routed, got shape {tuple(x.shape)}. "
            f"Higher-rank inputs need a verified device-layout equivalence for the "
            f"leading-dim flatten; reshape on the host or extend the op deliberately.")

    # Normalise the axis, then require it to be the TRAILING one. dim=0 is a different
    # datapath entirely (per-lane across-stick) and the only softmax pattern for it in
    # the tree is the legacy dim=0 fixture, which is non-functional. Never silently
    # treat a dim=0 request as dim=-1.
    axis = dim + x.dim() if dim < 0 else dim
    if axis != x.dim() - 1:
        raise NotImplementedError(
            f"spyre softmax: only the trailing axis (dim=-1) is routed, got dim={dim} "
            f"(axis {axis} of {x.dim()}). A cross-lane dim=0 softmax is a separate, "
            f"unbuilt datapath — refusing rather than reducing the wrong axis.")

    m, n = int(x.shape[0]), int(x.shape[1])
    plan = _build_launch_plan(m, n, _resolve_num_cores(m), _DTYPE_BITS[x.dtype])

    # The binary reads the activation as contiguous sticks.
    xc = x.contiguous()

    # DEVICE-RESIDENT, HANDLES KEPT. `t.to("spyre")` returns a NEW tensor, so the
    # kernel writes to that one; passing a converted tensor inline discards the only
    # handle to the written memory and reads back untouched zeros — which looks exactly
    # like "the kernel never wrote". That cost one box run on the standalone KAT.
    scratch = torch.zeros((plan.scratch_sticks, plan.elems_per_stick),
                          dtype=x.dtype, device=x.device)
    out = torch.zeros((m, n), dtype=x.dtype, device=x.device)

    # Positional XLAT binding: seg0=scratch, seg1=x, seg2=out. See module docstring.
    launch_kernel_from_bytes(plan.binary, [scratch, xc, out])
    return out


@softmax.register_fake
def _(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Shape/dtype for FakeTensor tracing. softmax is shape-preserving.
    return torch.empty_like(x)


@torch.library.register_kernel("aten::_softmax", ["spyre"])
def _aten__softmax_spyre(self, dim, half_to_float):
    """Route ``aten::_softmax(Tensor self, int dim, bool half_to_float)`` on spyre.

    ``half_to_float=True`` asks for an fp32 output from a half input (Inductor's
    mixed-precision softmax). This kernel produces DL16 and cannot honour that, so it
    refuses rather than returning the wrong dtype.
    """
    if half_to_float:
        raise NotImplementedError(
            "spyre softmax: half_to_float=True (fp32 output from a half input) is "
            "unsupported — the device kernel produces DL16.")
    return torch.ops.spyre.softmax(self, dim)
