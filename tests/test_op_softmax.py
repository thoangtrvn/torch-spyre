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

STATUS (2026-09-08): this file's own ULP-vs-CPU-fp32 table is NOT the routed-shape
allow-list -- read it as the stricter accuracy contract for the shapes it actually
tests, not as a census of what is routed.
================================================================================
The routed-shape source of truth is
`sentient_codegen.encoder.softmax_dispatch.SOFTMAX_WITNESSED_SHAPES`, surfaced through
`torch_spyre/ops/softmax.py` (read that module's docstring for the current set --
as of codegen steps 27/28 it is three `N=64` shapes plus one exact `N=128` shape).
This file encodes a DIFFERENT, stricter measurement (a real CPU-fp32 ULP reference,
see below) for only the shapes in `_HW_VERIFIED_SOFTMAX`, currently `M=1024` and
`M=2048`, both `N=64,C=32,DL16`. A shape being routed elsewhere does not make it green
here; a shape untested here (e.g. the N=128 shape above) does not mean it is unrouted
-- check codegen for admission, this file only for its own ULP evidence.

`torch.softmax(x, dim=-1)` at M=1024, N=64, C=32, DL16 routes end-to-end to the
five-packet device kernel through the `spyre::softmax` custom op, and is HW-verified
eager AND torch.compile at 3.28 ULP_DL16 with zero outliers — including all 3630
fp16-subnormal outputs, measurable only after the #111 readback fix. Both entry
points reproduced the bytes-launch KAT's number at the same input range, which is
what establishes that the routed path runs the same proven bytes (#112).

MORE ROUTED SHAPES DOES NOT MEAN ATTENTION IS SUPPORTED. Attention needs `[seq, seq]`
prefill and `[1, seq]` decode; both are REFUSED, and `1x2048` is refused. The DEEP
architectural reason decode/attention are not servable is that the cross-lane reduce
this file's witnessed shapes use is one stick per row, N==64 (a separate N=128 shape is
routed and torch-route HW-verified elsewhere, per the source of truth above, just not by
this file's ULP table) -- but as of 2026-09-08, `SoftmaxPlan(1,2048,1,16)` itself
constructs without error, so the CODE PATH THIS TEST ACTUALLY EXERCISES refuses `1x2048`
via the same generic witnessed-shape pin as `refuse_64x64_unwitnessed_pin` below, not via
a demonstrated N=2048-specific structural wall -- see that case's own comment before
changing either. This file therefore tests TWO things of equal weight for the shapes in
its own table:
  (a) the witnessed shapes compute correctly through both entry points, and
  (b) every OTHER case in this table FAILS LOUD rather than silently computing something.
(b) is not filler. The routed path admits only shapes in
`sentient_codegen.encoder.softmax_dispatch.SOFTMAX_WITNESSED_SHAPES`, and the reason
that pin exists is a real defect caught during #112: `SoftmaxPlan`'s constraints are
NECESSARY BUT NOT SUFFICIENT — it enforces `M == 32*C`, which M=32/C=1, M=64/C=2 and
M=512/C=16 (note: C=16 here, NOT the witnessed `M=512,N=64,C=32` shape above -- a
different, unrelated shape) all satisfy, and it BUILT all three. None of those three
has a golden; none has ever run on silicon. Without (b) a future edit re-opens that
hole and looks gated while doing it.

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

# (case_id, M, N, routed_C, expect, reason_substr)
#
# `routed_C` IS 32 FOR EVERY SHAPE THAT HAS ONE, and that is the corrected law: all four
# tracked dim=-1 goldens are `_32cores`, and the quantity that varies between them is
# rows_per_core (1024->32, 2048->64, 736->23, 1x2048->1). C is the CONSTANT; rows is
# derived as M/C. The first version of this table had it inverted (C = M/32), which is
# observationally identical at M=1024 (rows == 32 AND C == 32) and therefore survived a
# byte-match, a silicon HW-KAT and review — the 32x32 small-constant aliasing.
#
# Historical note, corrected 2026-09-08: this comment used to say only M=1024 was
# admissible and that no structurally-buildable pin-only refusal remained. That is
# stale on BOTH counts. M=2048 (2026-08-05, Band A) and M=512 (2026-08-17, #45 R1
# Part A) were each individually HW-witnessed and added to codegen's
# SOFTMAX_WITNESSED_SHAPES. Separately, `64x64` and the `1x2048` decode shape were
# found (2026-09-08) to currently construct fine but remain refused by the SAME
# generic witnessed-shape pin, NOT a structural row_unroll/N-specific reason as their
# names once implied -- that is a LIVE case of exactly the class this paragraph used
# to say was empty. The table below distinguishes the one genuinely structural
# refusal it currently checks (`736x64`: row_unroll divisibility) from the two
# generic pin-only refusals (`64x64`, `1x2048`) -- see each case's own comment for
# the live-message evidence, not this paragraph's history.
#
# Every reason below was verified against the real dispatch at the corrected C — not
# predicted. `reason_substr` pins WHICH guard fires so a rename cannot silently re-point
# a case at a different mechanism (the defect the box run caught).
_SOFTMAX_CASES = [
    # THE witnessed shape: byte-match golden + silicon, eager and compiled.
    ("witnessed_1024x64_c32", 1024, 64, 32, _OK, None),

    # rows=64: WITNESSED (Band A). The second-operand distance confound was RESOLVED --
    # decoded rows-independent (e*128=8192, identical in the rows=32 AND rows=64 goldens),
    # byte-matched (28416 B) AND silicon HW-KAT PASS (3.74 ULP, 0 bad, 3x clean, 2026-08-05).
    ("witnessed_2048x64_c32", 2048, 64, 32, _OK, None),

    # rows=16: WITNESSED since 2026-08-17 (commit 072deaf / #45 R1 Part A) --
    # silicon HW-KAT PASS, max_diff==0 vs a captured anchor reference output. NOT
    # byte-matched to a tracked golden binary, unlike M=1024/M=2048 (see
    # torch_spyre/ops/softmax.py's own docstring for that distinction). This case
    # was `refuse_512x64_unwitnessed_pin` / `_REFUSE` / "admits only" until
    # 2026-09-08 -- that was correct THEN (codegen had not yet witnessed this
    # shape) and went stale the moment codegen's admission landed without this
    # table being updated in the same change, exactly the discipline
    # "WHEN DOES A NEW SHAPE FLIP GREEN?" above asks for. Left OUT of
    # _HW_VERIFIED_SOFTMAX below deliberately -- this file's own ULP-vs-CPU-fp32
    # protocol has not been run for this shape, so the torch-path tests xfail for
    # it until someone actually runs and adds it. Do not add it speculatively.
    ("witnessed_512x64_c32", 512, 64, 32, _OK, None),

    # rows=23: u=4 does not divide 23 (the mvloopcnt trip is rows/u) and no remainder-row
    # tail is built. A REAL capability limit -- a genuine structural refusal, not the
    # witnessed-shape pin -- and note a golden for this shape EXISTS, so it becomes
    # byte-match-gatable the moment the tail is built. Reason pin corrected 2026-09-08
    # (was "must divide", which has never matched codegen's actual message): the live
    # message's own per-case suffix is "Underlying reason: mvloopcnt: rows_per_core=23
    # not divisible by row_unroll=4" -- "not divisible" is the discriminating substring,
    # confirmed absent from every OTHER case's message (verified directly, not assumed).
    ("refuse_736x64_no_row_tail", 736, 64, 32, _REFUSE, "not divisible"),

    # rows=2: NOT actually a structural row_unroll=2 refusal today (renamed + reason
    # corrected 2026-09-08 -- was `refuse_64x64_unwitnessed_unroll` / "unwitnessed",
    # which has never matched codegen's actual message, confirmed by direct history
    # search). codegen's own message says plainly: "The shape is structurally
    # buildable -- every SoftmaxPlan constraint passes -- but it is not present in
    # SOFTMAX_WITNESSED_SHAPES". That is the SAME generic witnessed-shape-pin class
    # `witnessed_512x64_c32` above used to be in before it was witnessed -- this case
    # just hasn't been witnessed (no golden, no HW-KAT), not blocked by a demonstrated
    # row_unroll structural wall.
    ("refuse_64x64_unwitnessed_pin", 64, 64, 32, _REFUSE,
     "not present in SOFTMAX_WITNESSED_SHAPES"),

    # N != 64, M=1, decode shape. THIS IS THE DECODE SHAPE -- attention is not yet
    # servable, and that remains true (see module docstring). But the reason_substr
    # pin here must reflect what THIS test path actually exercises, not the deep
    # architectural reason: corrected 2026-09-08 (was "one stick per row", which is
    # boilerplate present in EVERY refusal message including `refuse_736x64_no_row_tail`'s
    # genuinely-different structural one -- confirmed directly -- so it was passing for
    # the wrong reason, exactly the failure mode this file's own design says it guards
    # against). `SoftmaxPlan(1,2048,1,16)` itself constructs without error (confirmed:
    # codegen's message says "structurally buildable... every SoftmaxPlan constraint
    # passes"), so today this refuses via the SAME generic witnessed-shape pin as
    # `refuse_64x64_unwitnessed_pin` above, not a demonstrated N=2048-specific wall in
    # this code path. If a future change makes SoftmaxPlan itself reject N=2048
    # structurally, update this reason to match that new, more specific message.
    ("refuse_1x2048_decode", 1, 2048, 1, _REFUSE,
     "not present in SOFTMAX_WITNESSED_SHAPES"),
]

#: The corrected law: C is CONSTANT at 32 (clamped for M<32), rows is derived. Mirrored
#: from sentient_codegen.encoder.softmax_dispatch.resolve_softmax_cores, which is where
#: the derivation now lives BECAUSE the wrong version was unreachable by any test here.
_SOFTMAX_CORES = 32

#: Ids HW-verified on silicon (bytes-launch KAT; torch-path eager+compiled where run).
#: M=1024 populated 2026-08-04 (3.28 ULP_DL16, 0 outliers, 3630 subnormal outputs clean).
#: M=2048 (Band A, rows=64) populated 2026-08-05 (3.74 ULP_DL16, 0 bad, 3x clean unpadded
#: bytes-launch KAT). NOTE: M=2048 silicon was proven via the bytes-launch KAT
#: (scripts/softmax_dim_neg1_hw_kat.py KAT_M=2048); the torch-path eager/compiled twin at
#: M=2048 should be box-run to confirm routing before treating the torch entry as verified.
_HW_VERIFIED_SOFTMAX = {"witnessed_1024x64_c32", "witnessed_2048x64_c32"}


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


def _check_refused(case_id, entry, M, N, run, reason_substr):
    """An out-of-envelope shape must FAIL LOUD, for the REASON this case claims.

    Two assertions, and the second is the one that earns its keep:

    (1) TYPE. NotImplementedError (which SoftmaxEnvelopeError subclasses) is the
        contract: it PROPAGATES rather than falling back to host compute (#70).
        Asserting the type rather than "any exception" matters — Inductor treats its own
        `Unsupported` as "fall back", which for softmax would mean a silent CPU result.
        It also matters because it CAUGHT A REAL DEFECT: M=2048 refused with a raw
        ValueError from the C>32 core guard, which no NotImplementedError handler would
        have caught. The dispatch now gives every refusal one type.

    (2) REASON. Without this, a case passes as long as SOMETHING refuses — and the first
        version of this table proved that is not enough: two cases were named after
        guards that cannot fire on the routed path, and passed anyway. Pinning the
        message means a rename cannot silently re-point a case at a different mechanism.
    """
    x = _make_input(M, N)
    with pytest.raises(NotImplementedError) as exc:
        run(x.to("spyre")).cpu()
    msg = str(exc.value).lower()
    assert "softmax" in msg, (
        f"{case_id}/{entry}: refusal does not name the op the user called: {exc.value}")
    assert reason_substr.lower() in msg, (
        f"{case_id}/{entry}: refused, but NOT for the {reason_substr!r} reason this case "
        f"claims. Either the case name/mechanism is wrong or a different guard now fires "
        f"first. Message: {exc.value}")


# ---------------------------------------------------------------------------
# Entry point 1: EAGER (`torch.softmax(x, -1)`), via the per-op torch.compile wrap.
# ---------------------------------------------------------------------------
@requires_hw
@pytest.mark.parametrize(
    "case_id,M,N,C,expect,reason", _SOFTMAX_CASES, ids=[c[0] for c in _SOFTMAX_CASES]
)
def test_softmax_eager(case_id, M, N, C, expect, reason):
    if expect == _REFUSE:
        _check_refused(case_id, "eager", M, N, _softmax, reason)
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
    "case_id,M,N,C,expect,reason", _SOFTMAX_CASES, ids=[c[0] for c in _SOFTMAX_CASES]
)
def test_softmax_compiled(case_id, M, N, C, expect, reason):
    compiled = torch.compile(_softmax, dynamic=False)
    if expect == _REFUSE:
        _check_refused(case_id, "compiled", M, N, compiled, reason)
        return
    if case_id not in _HW_VERIFIED_SOFTMAX:
        pytest.xfail(f"{case_id}: not HW-verified through the compiled torch path.")
    _check_softmax(case_id, "compiled", M, N, compiled)


# ---------------------------------------------------------------------------
# HW-independent guards (run everywhere, incl. CI without a board).
# ---------------------------------------------------------------------------
def test_softmax_verified_membership_locked():
    """Contract lock: EXACTLY the shapes that have passed on silicon are HW-verified.

    Pins the verified set so a future edit cannot silently widen it past what has
    actually passed on silicon. Widening requires a golden or an HW-KAT for the new
    shape, an entry in SOFTMAX_WITNESSED_SHAPES, and an edit here — same change.
    """
    assert _HW_VERIFIED_SOFTMAX == {"witnessed_1024x64_c32", "witnessed_2048x64_c32"}, (
        "Softmax verified set changed. Silicon-passed shapes: M=1024 N=64 C=32 "
        "(2026-08-04, 3.28 ULP) and M=2048 N=64 C=32 (2026-08-05, 3.74 ULP, 0 bad, "
        "3x clean, bytes-launch HW-KAT). Confirm on hardware before widening, then "
        "update this lock."
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

    for case_id, M, N, C, expect, _reason in _SOFTMAX_CASES:
        key = (M, N, C, 16)
        if expect == _OK:
            assert key in SOFTMAX_WITNESSED_SHAPES, (
                f"{case_id} is an _OK case here but is NOT in codegen's "
                f"SOFTMAX_WITNESSED_SHAPES — the route would refuse it")
        else:
            assert key not in SOFTMAX_WITNESSED_SHAPES, (
                f"{case_id} is a _REFUSE case here but codegen ADMITS it — one of the "
                f"two has been widened without the other")


def test_table_routed_C_matches_the_ops_derivation():
    """The `routed_C` column must be what the op actually derives, not a wish.

    This column has now been wrong TWICE, in opposite directions, which is why it is
    asserted rather than documented. First it declared C=32 everywhere while the code
    derived C=M/32 (so 2048 really ran at C=64). Then the code itself was found to have
    solved `M == 32*C` for the wrong variable — C is the CONSTANT (32, per all four
    `_32cores` goldens) and rows is derived. A column that silently disagrees with the
    code is worse than no column: it reads as verified fact.

    Cross-checked against the single source of truth, `resolve_softmax_cores` in codegen,
    by `test_codegen_and_contract_agree_on_the_core_law` below.
    """
    for case_id, M, N, C, _expect, _reason in _SOFTMAX_CASES:
        derived = min(_SOFTMAX_CORES, M)
        assert C == derived, (
            f"{case_id}: table says routed C={C} but the corrected law derives "
            f"C={derived} from M={M} (C = min(32, M) — C is CONSTANT, rows is what "
            f"varies). Fix the table, not the law.")


def test_refusal_reasons_are_discriminating():
    """No two refusal reasons may be substrings of each other's messages.

    A reason that matches every message proves nothing — the same confound that made
    `"row"` useless in the codegen test (it appears in all three structural reasons).
    Here the risk is subtler: several cases share `"structurally buildable"` legitimately
    (they are the same class), but a reason must never be so generic that it would also
    match a DIFFERENT class's message.
    """
    specific = {r for _, _, _, _, e, r in _SOFTMAX_CASES if e == _REFUSE and r}
    # After Band A: M=2048/rows=64 is WITNESSED (no longer a "confounded" refusal). After
    # #45 R1 Part A (2026-08-17): M=512/rows=16 is ALSO witnessed, so the witnessed-shape
    # PIN's OLD reason string ("admits only") no longer has a live case. Corrected
    # 2026-09-08 (review round 3): the remaining refusals split into exactly ONE genuine
    # structural limit (736, u∤rows, "not divisible") and TWO cases that share the SAME
    # generic witnessed-shape-pin message today (64 and the decode shape) -- neither
    # "must divide" nor "unwitnessed" nor "one stick per row" ever matched codegen's
    # actual message (confirmed directly against live dispatch, and by searching this
    # repo's history for where those substrings might have come from -- nowhere). Two
    # DIFFERENT cases sharing one reason string is honest here, not a discrimination
    # failure: they dedupe to one element of `specific`, and that element is still
    # disjoint from "not divisible" -- the real distinction the pairwise check below
    # protects is structural-limit vs generic-pin, not "one string per case".
    assert "confounded" not in specific, (
        "a case still claims the 'confounded' reason, but the second-operand distance "
        "confound was RESOLVED (rows-independent) and M=2048 is now witnessed.")
    assert "admits only" not in specific, (
        "a case still claims the witnessed-shape-PIN's OLD reason string, but that "
        "wording was retired 2026-09-08 in favor of the live message's own "
        "'not present in SOFTMAX_WITNESSED_SHAPES' -- if codegen's wording changes "
        "again, re-verify against the LIVE message, don't guess.")
    assert {"not divisible", "not present in SOFTMAX_WITNESSED_SHAPES"} <= specific
    # No reason may be a substring of another, or a case could pass on the wrong guard.
    # (This CAUGHT the "witnessed" ⊂ "unwitnessed" clash when 512's reason was first set to
    # "witnessed". Also caught, review round 2/3: "one stick per row" is boilerplate
    # present in EVERY refusal message including 736's genuinely-different structural
    # one -- it was passing 1x2048's test for the wrong reason. Any new reason_substr
    # MUST be checked against the live message text before being added here, not
    # written by analogy to what a case's NAME suggests.)
    for a in specific:
        for b in specific:
            if a != b:
                assert a not in b, f"reason {a!r} is a substring of {b!r} — not discriminating"


def test_codegen_and_contract_agree_on_the_core_law():
    """The core law has ONE home (codegen); this asserts the mirror here matches it.

    `_SOFTMAX_CORES` and the table's routed_C column are a copy. The original defect was
    a law living where no test could reach it, so the cure is not just moving it but
    checking the copy against it.
    """
    try:
        from sentient_codegen.encoder.softmax_dispatch import (
            SOFTMAX_CORES,
            resolve_softmax_cores,
        )
    except ImportError:
        pytest.skip("sentient_codegen not importable (source env.sh)")
    assert _SOFTMAX_CORES == SOFTMAX_CORES
    for case_id, M, _N, C, _e, _r in _SOFTMAX_CASES:
        assert C == resolve_softmax_cores(M), (
            f"{case_id}: table C={C} disagrees with codegen's "
            f"resolve_softmax_cores({M})={resolve_softmax_cores(M)}")


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
