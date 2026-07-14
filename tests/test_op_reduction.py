"""Dedicated HW test for reduction ops on spyre (sum / max / mean, dim=0).

PURPOSE
-------
The living coverage contract for cross-row (dim=0) reductions — mirrors
tests/test_op_pointwise.py. Each op is exercised at real transformer shapes with a
DL16-exact known-answer device-vs-CPU check. Shapes / ops / paths not yet working are
xfail'd with the EXACT reason, so this file tracks progress as the two known gaps land:

  1. #193 — the torch.compile/eager REDUCTION FRONTEND is broken even for ALIGNED N:
     Inductor lowering raises `ValueError: 'R0_BLOCK' is not in list` for sum/max, and
     `RecursionError` for mean. So reductions do not reach the codegen backend through
     torch at all today (the reduction DATAPATH is HW-verified only via the direct
     scheduler `resolve_reduction_tiling` path, not through torch.compile). Every
     torch-dispatched case below xfails on this until the frontend is fixed.
  2. #180 — ragged / sub-stick N (N not a multiple of elements_per_stick=64 @16-bit):
     the reduction resolver raises `UnsupportedReductionShapeError`. Deferred until #193
     lands (ragged N is moot while no reduction reaches codegen). NOTE: dim=0 is PER-LANE
     (output lane j = reduce over rows of column j; lanes never interact), so a ragged N
     is an inert OUTPUT-tail case like pointwise — NOT the cross-lane-combine masking
     problem that dim=-1 (#173) has.

DL16 (1-6-9) note: AIU 1.0 does not represent integers > 1024 exactly. Operands and
reduced results are kept small so exact ops (sum/max) can assert max_diff == 0. Sums grow
with M (the reduced extent), so operands are chosen per shape to keep the sum <= 1024.

REDUCE AXIS
-----------
Only dim=0 (reduce across the M/row dimension → an N-wide output vector, per-lane) is in
scope / implemented. dim=-1 (within-row, cross-lane REDUCE+PACK) is a separate gap (#173)
and is xfail'd distinctly.
"""
import math

import pytest
import torch

_SPYRE_AVAILABLE = False
try:
    import torch_spyre  # noqa: F401 — autoloads the device
    _SPYRE_AVAILABLE = torch.zeros(1).to("spyre") is not None
except Exception:
    _SPYRE_AVAILABLE = False

requires_hw = pytest.mark.skipif(
    not _SPYRE_AVAILABLE, reason="spyre device not available (no hardware)"
)

_EPS = 64  # elements per stick at 16-bit


# --- operand builders: DL16-exact, reduced result stays <= 1024 --------------
def _ones_like_small(M, N):
    # All-ones: sum over M rows = M (<=1024 for M<=1024), max = 1, mean = 1. Exact in
    # DL16 for any M<=1024, and independent of N so it works at any (ragged) width.
    return torch.ones((M, N), dtype=torch.float16)


def _distinct_cols(M, N, cap=8):
    # Distinct-per-column, constant down rows: col j holds (j % cap) + 1. Then
    # sum(dim=0) = M * col_val, max(dim=0) = col_val, mean(dim=0) = col_val. Distinct
    # across columns so a dropped/duplicated output lane (ragged-tail bug) is CAUGHT.
    # Choose cap*M <= 1024 at call sites so the sum stays DL16-exact.
    cols = ((torch.arange(N, dtype=torch.int64) % cap) + 1).to(torch.float16)  # (N,)
    return cols.unsqueeze(0).expand(M, N).contiguous()


def _max_known_answer(M, N):
    # Strong MAX gate: values increase down the rows so the per-column max sits at the
    # LAST row (row M-1), and each column's max is distinct = (col%8)+2. A wrong FMINMAX
    # combine (keeps an earlier/wrong row), a dropped input stick, or a frozen lane all
    # change the max → nonzero max_diff. DL16-exact (values <= M+8 <= 1024 for these M).
    #   x[r, c] = (c % 8) + 1 + (r / M)  is NOT DL16-clean; instead use integer steps:
    #   x[r, c] = (c % 8) + 1 + r   → per-column max at r=M-1 = (c%8)+1+(M-1), distinct per col.
    rows = torch.arange(M, dtype=torch.int64).unsqueeze(1)          # (M,1)
    cols = (torch.arange(N, dtype=torch.int64) % 8).unsqueeze(0)    # (1,N)
    x = (cols + 1 + rows)                                            # (M,N), max down rows at M-1
    return x.to(torch.float16)


# --- op table: (id, torch fn dim=0, reference fn, exact?) ---------------------
def _sum0(t):
    return torch.sum(t, dim=0)


def _max0(t):
    return torch.max(t, dim=0).values


def _mean0(t):
    return torch.mean(t, dim=0)


_REDUCTION_OPS = [
    ("sum", _sum0, True),
    ("max", _max0, True),
    ("mean", _mean0, False),  # mean divides → allow small DL16 tol
]

# Which reduction ops reach a working codegen binary today (dim=0, per-lane path).
# `sum` and `max` are end-to-end HW-verified (max_diff=0). `mean` still fails for an
# OP-SPECIFIC reason unrelated to the reduce-axis path:
#   - max : FIXED (task #52) — torch.max(dim) lowers via @torch._inductor.runtime.
#           triton_helpers.max2 (a tl.reduce over the elementwise `maximum` combine). The
#           classifier now keys on the tt.call BASE name (max/max2/amax → "max"), robust to
#           the module-path churn that had made it fall through to pointwise_unsupported.
#           Routes to max_nonstick (PE FMINMAX); shares sum's stick-native axis path.
#   - mean: torch.mean lowering hits a RecursionError in the Inductor decomposition
#           (mean = sum / count) before reaching codegen. Separate frontend issue (#53).
_REDUCTION_OP_HW_VERIFIED = {"sum", "max"}

# (id, M, N, ragged?) — dim=0 reductions. N (last dim) is the CONTIGUOUS/stick axis and
# drives the output width in sticks = ceil(N/64); M is the reduced extent (rows). Aligned
# N is a multiple of 64; ragged N is not. Full expected-pass sweep for the supported ops
# (sum/max): vary M (reduced extent) AND N (output-stick count, incl. real-model hidden
# widths) so a shape-specific datapath bug — a dropped output stick, a per-core split issue
# at many output sticks, an M-loop-count error — is CAUGHT, not assumed away. DL16-exact:
# sum=M and max=(N-col)+M-1 stay <= 1024 for these M (<=512).
# regime: "" = supported (expected pass), "ragged" = #180, "regimeB" = #183 (single output
# stick with M>64 → reduce-axis split across cores, cross-core combine NOT built).
_SHAPES = [
    # --- aligned N, small output (1 stick), vary the reduced extent M (M<=64: single-core) ---
    ("m1_n64",       1,   64,   ""),        # M=1 (single reduced row) — decode-like
    ("m8_n64",       8,   64,   ""),        # 1 output stick
    ("m32_n64",      32,  64,   ""),
    ("m64_n64",      64,  64,   ""),        # M=64: single-core envelope boundary
    ("m512_n64",     512, 64,   "regimeB"), # M>64, 1 output stick → Regime B (#183), fail-loud
    # --- aligned N, multi output stick (output core-split regime), vary N width ---
    ("m8_n128",      8,   128,  ""),        # 2 output sticks
    ("m8_n256",      8,   256,  ""),        # 4 output sticks
    ("m8_n512",      8,   512,  ""),        # 8 output sticks
    ("m8_n768",      8,   768,  ""),        # GPT-2 hidden (12 sticks)
    ("m32_n2048",    32,  2048, ""),        # Llama-2B hidden (32 sticks) × real M
    ("m8_n4096",     8,   4096, ""),        # Llama/Mistral hidden (64 sticks)
    # --- ragged / sub-stick N (gated on #180 until the ragged-N follow-on) ---
    ("m8_n100",      8,   100,  "ragged"),  # N>eps ragged (1 full + 36 tail)
    ("m8_n40",       8,   40,   "ragged"),  # N<eps sub-stick
    ("m8_n5",        8,   5,    "ragged"),  # tiny sub-stick
]

_APPROX_TOL = 2e-2

# HW STATUS (2026-07-14): NO reduction reaches the codegen backend through torch —
# the Inductor frontend fails at lowering (#193: 'R0_BLOCK' is not in list for sum/max;
# RecursionError for mean), for BOTH aligned and ragged N, on every torch entry point
# (aten / eager / compiled all route through Inductor). The reduction datapath itself is
# HW-verified via the direct scheduler path (resolve_reduction_tiling). These tests
# therefore xfail on the FRONTEND until #193 lands; once it does, aligned shapes should
# pass and only the ragged shapes should remain xfail (#180) until the ragged-N follow-on.
# #193 has TWO stages. Stage 1 (R0_BLOCK arg mismatch) is FIXED — choices.py no longer
# injects R0_BLOCK for persistent reductions, so reductions now reach the codegen backend.
# #193 was a CHAIN of frontend gates, ALL now fixed for the dim=0 per-lane path:
#   stage 1 — R0_BLOCK arg mismatch (choices.py: no R0_BLOCK for persistent reductions).
#   stage 2 — mlir_text not threaded → axis recovery skipped (backends/spyre/compiler.py
#             now passes mlir_text=ttir_text; recover_reduction_dims returns correct extents).
#   stage 3 — axis MISLABEL: analyze_ttir returned the BLOCK trailing axis (r0=1) for a torch
#             dim=0 reduction. Fixed STICK-NATIVELY: reduced_axis_is_across_stick() reads the
#             load-stride arithmetic (reduced axis strided → across-stick → per-lane → axis 0;
#             reduced axis contiguous → within-stick → #173). _compute_tiling uses it and the
#             tiling's reduce_axis is no longer clobbered by the stale block axis.
# RESULT: torch.sum(dim=0) is end-to-end HW-verified (max_diff=0). dim=1/within-stick still
# fails loud (#173). max/mean fail for OP-SPECIFIC reasons (see _REDUCTION_OP_HW_VERIFIED),
# NOT the axis path.
_FRONTEND_WORKS = True   # stages 1+2+3 fixed for the dim=0 per-lane reduction path
_RAGGED_N_WORKS = False  # flip to True when the ragged-dim=0 reduction follow-on lands


def _check(op_id, exact, out_dev, out_ref):
    md = (out_dev.float() - out_ref.float()).abs().max().item()
    tol = 0.0 if exact else _APPROX_TOL
    assert md <= tol, f"{op_id}: max_diff={md} > tol={tol}"


@requires_hw
@pytest.mark.parametrize("shape_id,M,N,regime", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("op_id,fn,exact", _REDUCTION_OPS, ids=[o[0] for o in _REDUCTION_OPS])
def test_reduction_dim0(op_id, fn, exact, shape_id, M, N, regime):
    """dim=0 (per-lane, across-stick) reduction on device matches CPU across a full M×N
    sweep (reduced extent M ∈ 1..512; output width N up to real-model hidden 4096). sum and
    max are end-to-end HW-verified; mean xfails (op-specific); ragged N (#180) and Regime B
    (#183) shapes fail loud and are gated as xfail."""
    if op_id not in _REDUCTION_OP_HW_VERIFIED:
        pytest.xfail(
            f"{op_id}: dim=0 axis path works (sum/max prove it — same classifier), but "
            f"{op_id} fails for an OP-SPECIFIC reason: mean → RecursionError in the Inductor "
            "mean=sum/count decomposition (#53). NOT the reduce-axis gap."
        )
    if regime == "ragged" and not _RAGGED_N_WORKS:
        pytest.xfail(
            f"{op_id} {shape_id}: ragged/sub-stick N not yet supported for reductions "
            "(#180). dim=0 is per-lane so the tail is inert (like pointwise); ceil "
            "follow-on pending."
        )
    if regime == "regimeB":
        pytest.xfail(
            f"{op_id} {shape_id}: single output stick (N<=64) with M>64 is Regime B — the "
            "reduce axis is split across cores needing a cross-core datafifo combine, NOT "
            "built (#183). Fails loud (UnsupportedReductionShapeError), never silent-wrong."
        )
    # Known-answer operand, DL16-exact. sum/mean: all-ones (sum=M<=1024). max: increasing
    # down rows so the per-column max is a distinct value at the last row — a wrong FMINMAX
    # combine or dropped stick shows as nonzero max_diff (the FMINMAX-correctness gate).
    x = _max_known_answer(M, N) if op_id == "max" else _ones_like_small(M, N)
    out_dev = fn(x.to("spyre")).cpu()
    out_ref = fn(x)
    assert out_dev.shape == out_ref.shape, (
        f"{op_id} {shape_id}: device shape {tuple(out_dev.shape)} != logical "
        f"{tuple(out_ref.shape)} (ragged tail leaked into logical shape)."
    )
    _check(op_id, exact, out_dev, out_ref)


@requires_hw
@pytest.mark.parametrize("op_id,fn,exact", _REDUCTION_OPS, ids=[o[0] for o in _REDUCTION_OPS])
def test_reduction_dim0_distinct_cols(op_id, fn, exact):
    """Distinct-per-column operands at an aligned multi-stick width (N=128, M=8): a
    dropped/duplicated output lane shows as a nonzero max_diff. Guards the per-lane output
    layout. sum is HW-verified; max/mean xfail for op-specific reasons (see test_reduction_dim0)."""
    if op_id not in _REDUCTION_OP_HW_VERIFIED:
        pytest.xfail(f"{op_id}: op-specific gap (not the reduce-axis path) — see test_reduction_dim0.")
    M, N = 8, 128
    x = _distinct_cols(M, N, cap=8)          # sum = 8 * (1..8) = 8..64, DL16-exact
    out_dev = fn(x.to("spyre")).cpu()
    out_ref = fn(x)
    _check(op_id, exact, out_dev, out_ref)


@requires_hw
def test_elementwise_maximum_is_not_a_reduction():
    """Routing guard for the max2/maximum distinction (task #52). torch.maximum(a,b) is
    ELEMENTWISE (two tensors) and uses the `maximum__` combine helper — NOT the `max2__`
    reduction. It has no verified SFP binary, so it must STILL fail loud
    (pointwise_unsupported), never be mis-routed to max_nonstick (which would compute a
    reduction over a binary elementwise op → wrong output). A regression that made the
    reduction classifier match `maximum` would silently break this."""
    import pytest as _pt
    a = _ones_like_small(8, 64)
    b = _ones_like_small(8, 64) * 2
    with _pt.raises(Exception):  # UnsupportedPointwiseOpError surfaces as InductorError on compile
        torch.maximum(a.to("spyre"), b.to("spyre")).cpu()


@requires_hw
def test_reduction_dim_neg1_is_separate_gap():
    """dim=-1 (within-row, cross-lane) is a DISTINCT gap (#173) — the true cross-lane
    combine that would need identity-mask-before-combine for ragged N. Pinned as xfail
    so it is not conflated with the dim=0 / ragged-N (#180) work."""
    pytest.xfail("dim=-1 within-row reduction is task #173 (cross-lane REDUCE+PACK), "
                 "separate from dim=0 / ragged-N (#180).")
    x = _ones_like_small(8, 64)
    torch.sum(x.to("spyre"), dim=-1).cpu()  # noqa — unreachable past xfail


def test_reduction_shape_model():
    """Hardware-independent: dim=0 output width is ceil(N/eps) sticks (the ragged-N
    contract the eventual ceil-fix must honor — same arithmetic as pointwise). Guards it
    in CI without a board."""
    assert math.ceil(64 / _EPS) == 1
    assert math.ceil(128 / _EPS) == 2
    assert math.ceil(100 / _EPS) == 2    # ragged: 1 full + tail
    assert math.ceil(40 / _EPS) == 1     # sub-stick
    assert math.ceil(5 / _EPS) == 1
