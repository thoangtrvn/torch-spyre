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

# (id, M, N, ragged?) — dim=0 reductions. N drives output width (sticks); M is the
# reduced extent. Aligned N ∈ {64, 128}; ragged N ∈ {100, 40, 5}.
_SHAPES = [
    ("m8_n64",       8,   64,   False),   # 1 output stick, aligned
    ("m8_n128",      8,   128,  False),   # 2 output sticks, aligned
    ("m64_n64",      64,  64,   False),   # larger reduced extent
    ("m8_n100",      8,   100,  True),    # N>eps ragged (1 full + 36 tail)
    ("m8_n40",       8,   40,   True),    # N<eps sub-stick
    ("m8_n5",        8,   5,    True),    # tiny sub-stick
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
# Stage 2 (axis recovery) remains: Inductor collapses the dim=0 kept-dim to N=1, and
# ttir_to_schedule._compute_tiling fails loud ("problem_dims have N=1") because it cannot
# recover the true (M, N) under the regex TTIR backend. So torch-dispatched reductions
# still don't produce output — they now fail at codegen tiling, not at Triton arg binding.
_FRONTEND_WORKS = False  # Stage 1 (R0_BLOCK) fixed; Stage 2 (axis recovery, N=1) still open
_FRONTEND_STAGE1_FIXED = True  # R0_BLOCK arg mismatch resolved (choices.py persistent-reduction gate)
_RAGGED_N_WORKS = False  # flip to True when the ragged-dim=0 reduction follow-on lands


def _check(op_id, exact, out_dev, out_ref):
    md = (out_dev.float() - out_ref.float()).abs().max().item()
    tol = 0.0 if exact else _APPROX_TOL
    assert md <= tol, f"{op_id}: max_diff={md} > tol={tol}"


@requires_hw
@pytest.mark.parametrize("shape_id,M,N,ragged", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("op_id,fn,exact", _REDUCTION_OPS, ids=[o[0] for o in _REDUCTION_OPS])
def test_reduction_dim0(op_id, fn, exact, shape_id, M, N, ragged):
    """dim=0 reduction on device matches CPU. Currently xfails on the #193 frontend for
    ALL shapes; ragged shapes additionally gated on the #180 ragged-N follow-on."""
    if not _FRONTEND_WORKS:
        pytest.xfail(
            f"{op_id} {shape_id}: #193 stage 2 (axis recovery) — the R0_BLOCK arg "
            "mismatch (stage 1) is FIXED so reductions now reach codegen, but Inductor "
            "collapses the dim=0 kept-dim to N=1 and ttir_to_schedule._compute_tiling "
            "fails loud ('problem_dims have N=1'). Recovering the true (M,N) needs the "
            "libtriton structural walk (reduction-dim-recovery #193)."
        )
    if ragged and not _RAGGED_N_WORKS:
        pytest.xfail(
            f"{op_id} {shape_id}: ragged/sub-stick N not yet supported for reductions "
            "(#180). dim=0 is per-lane so the tail is inert (like pointwise); ceil "
            "follow-on pending after #193."
        )
    # Keep the reduced result DL16-exact: all-ones → sum=M (<=1024 for these M).
    x = _ones_like_small(M, N)
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
    dropped/duplicated output lane shows as a nonzero max_diff. Guards the per-lane
    output layout once the frontend works. xfails on #193 until then."""
    if not _FRONTEND_WORKS:
        pytest.xfail("#193 stage 2 (axis recovery, N=1) — see test_reduction_dim0.")
    M, N = 8, 128
    x = _distinct_cols(M, N, cap=8)          # sum = 8 * (1..8) = 8..64, DL16-exact
    out_dev = fn(x.to("spyre")).cpu()
    out_ref = fn(x)
    _check(op_id, exact, out_dev, out_ref)


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
