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
"""Living coverage CONTRACT for softmax(dim=-1) — the attention softmax — on spyre.

STATUS (2026-08-04): ROUTED and HW-VERIFIED at exactly ONE shape.
================================================================================
`torch.softmax(x, dim=-1)` at M=1024, N=64, C=32, DL16 routes end-to-end to the
five-packet device kernel through the `spyre::softmax` custom op, and is HW-verified
eager AND torch.compile at 3.28 ULP_DL16 with zero outliers — including all 3630
fp16-subnormal outputs, measurable only after the #111 readback fix. Both entry
points reproduced the bytes-launch KAT's number at the same input range, which is
what establishes that the routed path runs the same proven bytes (#112).

ONE SHAPE IS NOT SUPPORT. Attention needs `[seq, seq]` prefill and `[1, seq]` decode;
both are REFUSED, and `1x2048` is refused outright (the cross-lane reduce is one stick
per row, N==64). This file therefore tests TWO things of equal weight:
  (a) the witnessed shape computes correctly through both entry points, and
  (b) every OTHER shape FAILS LOUD rather than silently computing something.
(b) is not filler. The routed path admits only shapes in
`sentient_codegen.encoder.softmax_dispatch.SOFTMAX_WITNESSED_SHAPES`, and the reason
that pin exists is a real defect caught during #112: `SoftmaxPlan`'s constraints are
NECESSARY BUT NOT SUFFICIENT — it enforces `M == 32*C`, which M=32/C=1, M=64/C=2 and
M=512/C=16 all satisfy, and it BUILT all three. None has a golden; none has ever run
on silicon. Without (b) a future edit re-opens that hole and looks gated while doing it.

WHY THE TOLERANCE IS NOT max_diff == 0
--------------------------------------
Unlike matmul's D1/D3 known-answers, softmax has no DL16-exact reference: the exp is a
first-order Cody-Waite reconstruction in DL16 (1-6-9), so the honest gate is a relative
ULP bound. #97 has NOT established the silicon NFE floor, so 4 ULP_DL16 is GUIDANCE
carried over from the KAT, not a derived limit.

The input range is chosen so NO output is fp16-subnormal (x in +-2 => min |ref| ~8e-4,
well above 2^-14). That is deliberate: it keeps this regression test clear of the
fp16-READBACK floor entirely, so a failure here means the kernel or the route, never
the instrument. The subnormal population IS covered — by
`scripts/softmax_torch_path_hw_kat.py`, which does the three-way population split with
a binade of guard. Do not "improve" this test by widening the range; you would
reintroduce the measurement ambiguity that made #110 report a false refutation.

CODEGEN-DIRECT HW VERIFICATION PROVES THE BINARY, NOT THE ROUTE. The bytes-launch KAT
(`scripts/softmax_dim_neg1_hw_kat.py`) gates the emitted program; only these tests and
the routed KAT gate dispatch, the custom op, the decomposition, the envelope guard, the
scratch allocation and the 3-tensor bind. A green bytes-launch KAT is NOT evidence for
this file.

WHEN DOES A NEW SHAPE FLIP GREEN?
  Land a byte-match golden OR an HW-KAT for it, add it to SOFTMAX_WITNESSED_SHAPES in
  codegen, add its id to _HW_VERIFIED_SOFTMAX, AND update the membership lock below --
  all in the same change. Never relax a guard to admit a shape.

Cross-ref: codegen/docs/op-milestones/softmax.md (§4 verification, §6 open gaps),
torch_spyre/ops/softmax.py (why a custom op and not an Inductor lowering: the generic
TTIR path cannot bind the third HBM segment -- #113).
"""
import pytest
import torch

_SPYRE_AVAILABLE = False
try:
    import torch_spyre  # noqa: F401 — autoloads the "spyre" device
    _SPYRE_AVAILABLE = torch.zeros(1).to("spyre") is not None
except Exception:
    _SPYRE_AVAILABLE = False

requires_hw = pytest.mark.skipif(
    not _SPYRE_AVAILABLE, reason="spyre device not available (no hardware)"
)

#: DL16 is 1-6-9: 9 explicit mantissa bits, so one ULP at a binade is 2^-9.
ULP_DL16 = 2.0 ** -9

#: Carried from the HW-KAT. GUIDANCE, not a derived floor (#97 is open).
REL_TOL_ULPS = 4.0

#: fp16 min normal. The input range below keeps every output above this, so the
#: #111-class readback floor cannot contaminate the measurement.
FP16_MIN_NORMAL = 2.0 ** -14


# --- entry points (mirror test_op_matmul.py: EAGER + explicit torch.compile) ---------
def _softmax(x):
    return torch.softmax(x, dim=-1)


# --- the case table -----------------------------------------------------------------
# (case_id, M, N, C, expect)  --  expect is "ok" or "refuse".
# C is DOCUMENTATION of the routed core count (softmax derives C = M/32 from the
# envelope law M == 32*C); a user writes torch.softmax and never picks C.
_OK, _REFUSE = "ok", "refuse"

_SOFTMAX_CASES = [
    # THE witnessed shape: byte-match golden + silicon, eager and compiled.
    ("witnessed_1024x64_c32", 1024, 64, 32, _OK),

    # --- refused: structurally UNBUILDABLE, each for a DIFFERENT honest reason -----
    # second-operand byte distance is CONFOUNDED in the only golden (admits M == 32*C)
    ("refuse_2048x64_confounded", 2048, 64, 32, _REFUSE),
    # row unroll u=4 must divide rows_per_core=23; no remainder-row tail is built
    ("refuse_736x64_no_row_tail", 736, 64, 32, _REFUSE),
    # the cross-lane reduce is ONE STICK PER ROW (N==64); N=2048 is 32 sticks/row.
    # This is the decode shape -- refusing it is why softmax is not servable yet.
    ("refuse_1x2048_decode", 1, 2048, 32, _REFUSE),

    # --- refused: structurally BUILDABLE but UNWITNESSED (the dangerous class) -----
    # Every SoftmaxPlan constraint passes and the dispatch WOULD build a binary; there
    # is no golden and no silicon run. If these ever return a tensor, the
    # witnessed-shape pin has been removed and unverified binaries are shipping.
    ("refuse_512x64_unwitnessed", 512, 64, 16, _REFUSE),
    ("refuse_64x64_unwitnessed", 64, 64, 2, _REFUSE),
]

#: Ids HW-verified on silicon through the TORCH path (eager AND torch.compile).
#: Populated 2026-08-04: 3.28 ULP_DL16, 0 outliers, 3630 subnormal outputs clean,
#: both entry points matching the bytes-launch KAT at the same range.
_HW_VERIFIED_SOFTMAX = {"witnessed_1024x64_c32"}


def _make_input(M, N):
    """A well-conditioned DL16-exact input whose softmax has NO subnormal output.

    Small integers scaled by 1/8: integers <= 1024 are exact in DL16 and x/8 is a
    dyadic rational DL16 represents exactly, so the INPUT contributes no rounding and
    a failure is attributable to the kernel or the route. Range +-2 keeps
    min |softmax| ~8e-4, three binades above 2^-14.
    """
    n = M * N
    vals = (torch.arange(n, dtype=torch.int64) % 33) - 16      # -16..16
    return (vals.reshape(M, N).to(torch.float16) / 8.0)        # -2.0..2.0


def _check_softmax(case_id, entry, M, N, run):
    """Assert the routed softmax matches a CPU fp32 reference within the ULP guide."""
    x = _make_input(M, N)
    ref = torch.softmax(x.to(torch.float32), dim=-1)

    out = run(x.to("spyre")).cpu()   # NEVER .float().cpu() — device-dtype conversion
    assert out.shape == ref.shape, (
        f"{case_id}/{entry}: device shape {tuple(out.shape)} != {tuple(ref.shape)}")

    g = out.to(torch.float32)
    # A dataflow fault (nothing stored / wrong segment / never launched) reads as an
    # all-zero buffer. Call it what it is instead of reporting a huge max_diff.
    assert g.abs().max() > 0, (
        f"{case_id}/{entry}: output is entirely ZERO — that is a DATAFLOW fault, not "
        f"an accuracy result")
    # An identity passthrough would look plausible on near-uniform rows.
    assert not torch.equal(out, x), (
        f"{case_id}/{entry}: output == input; the op returned its operand")

    # The measurement must be clear of the fp16 readback floor by construction.
    assert ref.min() >= FP16_MIN_NORMAL, (
        f"{case_id}/{entry}: reference contains fp16-subnormal outputs "
        f"(min={float(ref.min()):.3e}); this test's range must keep every output "
        f"normal so the fp16 readback floor cannot contaminate it")

    rel = ((g - ref).abs() / ref.abs()).max().item() / ULP_DL16
    assert rel <= REL_TOL_ULPS, (
        f"{case_id}/{entry}: max_rel={rel:.2f} ULP_DL16 > {REL_TOL_ULPS} guide")

    # Independent of the reference: softmax rows sum to 1. Sensitive to any per-row
    # scale error (so it exonerates the sum/reciprocal path), blind by construction to
    # small-element error — state the axis, do not read it as corroboration.
    row_err = (g.sum(-1) - 1.0).abs().max().item() / ULP_DL16
    assert row_err <= 2.0 * REL_TOL_ULPS, (
        f"{case_id}/{entry}: row-sum error {row_err:.2f} ULP_DL16 — a per-row scale bug")


def _check_refused(case_id, entry, M, N, run):
    """An out-of-envelope shape must FAIL LOUD, never silently compute.

    NotImplementedError (which SoftmaxEnvelopeError subclasses) is the contract: it
    PROPAGATES rather than falling back to host compute (#70). Asserting the type
    rather than 'any exception' matters — Inductor treats its own `Unsupported` as
    'fall back', which for softmax would mean a silent CPU result.
    """
    x = _make_input(M, N)
    with pytest.raises(NotImplementedError):
        run(x.to("spyre")).cpu()


# ---------------------------------------------------------------------------
# Entry point 1: EAGER (`torch.softmax(x, -1)`), via the per-op torch.compile wrap.
# ---------------------------------------------------------------------------
@requires_hw
@pytest.mark.parametrize(
    "case_id,M,N,C,expect", _SOFTMAX_CASES, ids=[c[0] for c in _SOFTMAX_CASES]
)
def test_softmax_eager(case_id, M, N, C, expect):
    if expect == _REFUSE:
        _check_refused(case_id, "eager", M, N, _softmax)
        return
    if case_id not in _HW_VERIFIED_SOFTMAX:
        pytest.xfail(
            f"{case_id}: softmax(dim=-1) at M={M} N={N} C={C} is not HW-verified "
            f"through the torch path. Add a golden or an HW-KAT, extend "
            f"SOFTMAX_WITNESSED_SHAPES, then add the id here and to the lock.")
    _check_softmax(case_id, "eager", M, N, _softmax)


# ---------------------------------------------------------------------------
# Entry point 2: explicit torch.compile (full Inductor capture; the decomposition must
# keep aten._softmax OPAQUE so the custom op replaces the generic lowering).
# ---------------------------------------------------------------------------
@requires_hw
@pytest.mark.parametrize(
    "case_id,M,N,C,expect", _SOFTMAX_CASES, ids=[c[0] for c in _SOFTMAX_CASES]
)
def test_softmax_compiled(case_id, M, N, C, expect):
    compiled = torch.compile(_softmax, dynamic=False)
    if expect == _REFUSE:
        _check_refused(case_id, "compiled", M, N, compiled)
        return
    if case_id not in _HW_VERIFIED_SOFTMAX:
        pytest.xfail(f"{case_id}: not HW-verified through the compiled torch path.")
    _check_softmax(case_id, "compiled", M, N, compiled)


# ---------------------------------------------------------------------------
# HW-independent guards (run everywhere, incl. CI without a board).
# ---------------------------------------------------------------------------
def test_softmax_verified_membership_locked():
    """Contract lock: EXACTLY one shape is HW-verified through the torch path.

    Pins the verified set so a future edit cannot silently widen it past what has
    actually passed on silicon. Widening requires a golden or an HW-KAT for the new
    shape, an entry in SOFTMAX_WITNESSED_SHAPES, and an edit here — same change.
    """
    assert _HW_VERIFIED_SOFTMAX == {"witnessed_1024x64_c32"}, (
        "Softmax verified set changed. Only M=1024 N=64 C=32 DL16 has passed on "
        "silicon (2026-08-04, 3.28 ULP_DL16, eager + torch.compile). Confirm on "
        "hardware before widening, then update this lock."
    )


def test_codegen_witnessed_set_agrees_with_this_contract():
    """The codegen pin and this test table must not drift apart.

    Two independent statements of 'what is admitted' will diverge eventually unless
    something compares them. This is that something: every _OK case must be in
    codegen's SOFTMAX_WITNESSED_SHAPES, and every _REFUSE case must NOT be.
    """
    try:
        from sentient_codegen.encoder.softmax_dispatch import (
            SOFTMAX_WITNESSED_SHAPES,
        )
    except ImportError:
        pytest.skip("sentient_codegen not importable (source env.sh)")

    for case_id, M, N, C, expect in _SOFTMAX_CASES:
        key = (M, N, C, 16)
        if expect == _OK:
            assert key in SOFTMAX_WITNESSED_SHAPES, (
                f"{case_id} is an _OK case here but is NOT in codegen's "
                f"SOFTMAX_WITNESSED_SHAPES — the route would refuse it")
        else:
            assert key not in SOFTMAX_WITNESSED_SHAPES, (
                f"{case_id} is a _REFUSE case here but codegen ADMITS it — one of the "
                f"two has been widened without the other")


def test_softmax_is_not_claimed_beyond_one_shape():
    """Guard the honesty of the coverage claim itself.

    Attention needs [seq,seq] prefill and [1,seq] decode. Both are refused today, and
    the decode shape is in the table as a _REFUSE case precisely so that 'softmax
    works' can never be read as 'attention softmax works'. If someone makes decode
    pass, they must move it deliberately — not by deleting this.
    """
    decode = [c for c in _SOFTMAX_CASES if c[0] == "refuse_1x2048_decode"]
    assert decode and decode[0][4] == _REFUSE, (
        "the [1, seq] decode shape must remain a documented REFUSAL until it is built "
        "and HW-verified — it is the shape that makes softmax servable for attention")
