"""Dedicated HW test for pointwise (element-wise) ops on spyre.

PURPOSE
-------
One file per supported op, enumerating every shape it should handle — driven by the
real transformer element-wise shapes ([seq, d_model]) — with a DL16-exact
known-answer device-vs-CPU check. Shapes / ops not yet HW-verified are xfail/skip
with the exact reason, so this file is the living coverage contract.

DL16 (1-6-9) note: AIU 1.0 does not represent integers > 1024 exactly. All operands
and results here are kept small (<= 1024) so the check is exact (max_diff == 0),
never a spurious DL16 rounding artifact. See the project DL16 note.

POINTWISE OPS covered (register_torch_compile_kernel, torch_spyre/ops/eager.py):
  unary : relu, neg, abs, exp, tanh, sigmoid, sqrt, reciprocal
  binary: add, mul, sub, div, maximum
Each is exercised at the real transformer element-wise shapes.

SHAPE MODEL
-----------
Pointwise ops are element-wise over a [M, N] tensor; the codegen tiles N into
64-element sticks (N=elems/stick) and M rows across cores. The relevant real shapes
are the transformer residual/activation tensors [seq, d_model]:
  [1, 64]           smallest (1 stick)      — sanity
  [128, 768]        GPT-2 hidden            — supported
  [128, 4096]       Llama/Mistral hidden    — supported
  [512, 4096]       longer seq              — supported
  [2048, 4096]      max seq × hidden        — LX scratchpad capacity boundary (xfail:
                    LBR0 pin — large single-tile tensor exceeds LX; needs tiling/
                    chunk-large-tensors, a tracked codegen follow-on)
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


# --- op table: (id, callable, is_binary, input-domain builder) ---------------
# The input builder returns DL16-exact operand(s) whose op result stays <= 1024.
def _small_pos(shape):
    # values 1..8, exact in DL16, keeps products/sums small.
    # NOTE: build the arange in int64 and take `% 8` BEFORE casting to float16.
    # A float16 arange overflows at n>65504 (fp16 can't represent large integers →
    # inf/NaN), which silently NaN'd multi-stick shapes (512×4096 = 2M elements).
    # int64 arange is exact at any n; the values after `%8+1` are 1..8, DL16-exact.
    n = int(torch.tensor(shape).prod())
    return ((torch.arange(n, dtype=torch.int64) % 8 + 1).to(torch.float16).reshape(shape))


def _signed_small(shape):
    # DL16-exact values spanning negatives AND positives: (i%8)-4 → -4..3, so relu's
    # max(x,0) clamp is genuinely exercised (roughly half the elements clamp to 0).
    # int64 arange then cast (fp16 arange overflows to NaN at n>65504 — see _small_pos).
    n = int(torch.tensor(shape).prod())
    return ((torch.arange(n, dtype=torch.int64) % 8 - 4).to(torch.float16).reshape(shape))


def _distinct_per_stick(shape, period, offset=0):
    # DL16-exact operand whose value is DISTINCT PER STICK with a period that must NOT
    # divide the 64-element stick width. This de-periodizes the operand so a per-stick /
    # per-core FEED bug (a load pointer that doesn't advance across sticks/cores) is
    # CAUGHT, not masked.
    #
    # WHY THIS EXISTS (task #33): the `_small_pos`/`_signed_small` builders use periods
    # 8 and 4 — both DIVIDE the 64-element stick — so every stick carries identical
    # values. A real sub bug (multi-core input_y PatchInit read `b` from stick 0 on every
    # core → device computed a[stick] - b[stick % sticks_per_core]) was INVISIBLE to those
    # operands and shipped as "HW-verified" until a de-periodized probe exposed it. Any
    # period coprime-to-64 (7, 13, 20, …) makes value(stick s) = ((s % period) + 1 +
    # offset), so a frozen/misindexed stick shows up as a nonzero max_diff. Result stays
    # <= 1024 (DL16-exact). See codegen pointwise.md §12.
    M, N = shape
    n_sticks = (N + _EPS - 1) // _EPS
    stick_of_col = torch.arange(N, dtype=torch.int64) // _EPS          # (N,) stick index
    vals = (stick_of_col % period) + 1 + offset                        # distinct per stick
    row = vals.to(torch.float16).unsqueeze(0)                          # (1, N)
    return row.expand(M, N).contiguous()


_UNARY_OPS = [
    # relu is BARE (no `x-4`): a fused sub+relu (`torch.relu(x-4)`) lowers to an
    # Inductor `maximum__` helper over a subf result — a MULTI-OP fused elementwise
    # chain the classifier does NOT yet handle generically (it faults; see memory
    # generic-fused-elementwise-gap). Bare relu on a SIGNED input (built by
    # _signed_small, so negatives are present WITHOUT an in-graph sub) is the
    # HW-verified path. Fused relu(x-4) is covered by an explicit xfail below.
    ("relu",       lambda x: torch.relu(x)),              # bare max(x,0); needs signed input
    ("neg",        lambda x: torch.neg(x)),
    ("abs",        lambda x: torch.abs(x - 4)),
    ("tanh",       lambda x: torch.tanh(x / 8)),           # bounded, exact-ish domain
    ("sigmoid",    lambda x: torch.sigmoid(x / 8)),
    ("sqrt",       lambda x: torch.sqrt(x)),               # x in 1..8
]
_BINARY_OPS = [
    ("add",        lambda a, b: a + b),
    ("mul",        lambda a, b: a * b),
    ("sub",        lambda a, b: a - b),
    ("maximum",    lambda a, b: torch.maximum(a, b)),
]

# (id, M, N, capacity_boundary)
# NOTE: add/mul are HW-verified for the SINGLE-STICK-per-core path (s1x64). The
# multi-stick shapes (N>=768 → >=12 sticks/row → burst LXLU) exercise the looped-LDI
# burst path, a documented deferred divergence (not yet byte-matched/HW-verified).
_SHAPES = [
    ("s1x64",        1,    64,   False),
    ("gpt2_128x768", 128,  768,  False),
    ("llama_128x4096", 128, 4096, False),
    ("seq512x4096",  512,  4096, False),
    ("maxseq_2048x4096", 2048, 4096, True),   # LX capacity boundary
]

# Single-stick shapes (N == elements_per_stick) where add/mul are HW-verified.
_SINGLE_STICK_SHAPES = {"s1x64"}

# Tolerance: unary transcendentals (tanh/sigmoid/sqrt) go through DL16 + a device
# approximation, so allow a small tol; exact integer ops (relu/neg/abs/add/mul/sub/
# maximum on <=1024 values) must be max_diff == 0.
_EXACT_OPS = {"relu", "neg", "abs", "add", "mul", "sub", "maximum"}
_APPROX_TOL = 2e-2


def _check(op_id, out_dev, out_ref):
    md = (out_dev - out_ref).abs().max().item()
    tol = 0.0 if op_id in _EXACT_OPS else _APPROX_TOL
    assert md <= tol, f"{op_id}: max_diff={md} > tol={tol}"


# HW STATUS (2026-07-10, HW-measured): per-op support, not a blanket flag.
#   SUPPORTED (torch.compile HW-verified, max_diff==0): binary add, mul. These
#   byte-match the deeptools golden op_add load layout (3-LBR-block + REUSE two-FMA);
#   see codegen commit "add/mul WORK ON HW" + .git/sdd/POINTWISE_TORCH_PATH_BROKEN_
#   2026-07-09.md.
#   sub (binary) + neg (unary) added 2026-07-10: REUSE two-FMA + FNMS (sub: a - REUSE(b)*1;
#   neg: -(x*1 - 0)), grounded in deeptools add_mul_sub_fwd.smc:343/345. HW-verified on the
#   single-stick path (known-answer: 5-2=3, neg([1,2,3,4])=[-1,-2,-3,-4], max_diff=0). The
#   sub operand order was HW-corrected (LXLU sends input_y=b first → REUSE=b, FIFO=a; an
#   initial src0/src2 swap computed b-a, fixed to a-b).
#   NOT YET SUPPORTED (Part-A fail-loud — codegen raises UnsupportedPointwiseOpError,
#   no verified SFP binary): relu/abs/tanh/sigmoid/sqrt (unary) and maximum (binary).
#   relu/maximum/abs need the FMINMAX submode or a relu mode-bit (not yet wired);
#   tanh/sigmoid/sqrt need the FEST transcendental path. Marked xfail until built + HW-verified.
#   The >8192-stick shapes (boundary=True) stay xfail on the LX-capacity /
#   chunk-large-tensors gap regardless of op.
#   relu (unary) added 2026-07-13: single sfp.fminmax(MAX, S2=ZERO) = max(x,0), byte-matches
#   the deeptools op_relu_5/op_relu_128_768 goldens. Fixes the aten.relu→identity mis-map that
#   was COPYING negatives (HW-confirmed COPIED_INPUT before the fix).
#   MULTI-STICK MULTI-CORE now HW-VERIFIED (2026-07-13, task #12 bug-3 fix, commit 5b59031):
#   neg/relu at 128×768 (spc=48, 2 chunks), 128×4096 (8 chunks), 512×4096 all max_diff=0 on
#   silicon. Root cause was the unary multi-chunk OUTPUT-STORE base offset (chunk LAR = raw
#   cumulative, not output_base+cumulative → chunk>0 read the INPUT staging region and copied
#   un-transformed input). So neg/relu are no longer single-stick-only.
_SUPPORTED_OPS = {"add", "mul", "sub", "neg", "relu"}
# Ops HW-verified across the MULTI-STICK shapes (not just s1x64), max_diff=0 on silicon:
#   add/mul   — via the Fix A (unrolled) + Fix B (looped L3) chunked-prefill work
#     (HW-verified 128×768, 512×4096, 2048×2048/4096 across 32 cores).
#   neg/relu  — via the task-#12 bug-3 fix (2026-07-13, commit 5b59031). neg/relu decode
#     at full hidden width (1×4096) additionally needed the 2026-07-14 UR-unroll-alias
#     emitter fix (commit 137ed55) — it HUNG before that (codegen pointwise.md §11).
#   sub       — CORRECTED 2026-07-14 (task #33, commit 8fe89bb): sub multi-stick was
#     silently WRONG until then — the per-core input_y (b) EAR1 PatchInit fold was gated
#     to add/mul only, so every core read b from stick 0 (device computed
#     a[stick]-b[stick % sticks_per_core]). The PERIODIC operands above (periods 8/4, both
#     divide the 64-stick width) MASKED it — every stick was identical. Fixed + re-verified
#     with distinct-per-stick operands (see test_pointwise_binary_distinct_per_stick, the
#     de-periodized guard that catches this class). pointwise.md §12.
# The only remaining xfail is the boundary shape (2048×4096, LX-capacity — see `boundary`).
_MULTISTICK_VERIFIED_OPS = {"add", "mul", "sub", "neg", "relu"}


@requires_hw
@pytest.mark.parametrize("shape_id,M,N,boundary", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("op_id,fn", _UNARY_OPS, ids=[o[0] for o in _UNARY_OPS])
def test_pointwise_unary(op_id, fn, shape_id, M, N, boundary):
    """Unary pointwise op on device matches CPU across real transformer shapes.

    relu MUST see negatives to exercise the max(x,0) clamp (a positive-only input makes
    relu a no-op and would NOT catch the identity-mis-map or the multi-chunk dropped-
    compute bug). _signed_small provides them for relu; other unary ops keep _small_pos.
    """
    if op_id not in _SUPPORTED_OPS:
        pytest.xfail(
            f"{op_id}: no HW-verified SFP binary yet — codegen fails loud "
            "(UnsupportedPointwiseOpError, Part A). Build+verify its binary to enable."
        )
    if shape_id not in _SINGLE_STICK_SHAPES and op_id not in _MULTISTICK_VERIFIED_OPS:
        pytest.xfail(
            f"{op_id} {shape_id}: multi-stick path not yet HW-verified for this unary op "
            "(neg/relu ARE multi-stick-verified; others' burst looped-LDI is deferred)."
        )
    if boundary:
        pytest.xfail(
            f"{shape_id}: [{M},{N}] single-tile exceeds LX scratchpad (LBR0 pin) — "
            "needs tiling / chunk-large-tensors (tracked codegen follow-on)."
        )
    # relu needs signed input (clamp); build it WITHOUT an in-graph sub (fused sub+relu
    # hits the separate generic-fused-elementwise classifier gap).
    x = _signed_small((M, N)) if op_id == "relu" else _small_pos((M, N))
    out_dev = fn(x.to("spyre")).cpu()
    out_ref = fn(x)
    _check(op_id, out_dev, out_ref)


@requires_hw
@pytest.mark.parametrize("shape_id,M,N,boundary", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("op_id,fn", _BINARY_OPS, ids=[o[0] for o in _BINARY_OPS])
def test_pointwise_binary(op_id, fn, shape_id, M, N, boundary):
    """Binary pointwise op on device matches CPU across real transformer shapes.

    add/mul/sub are HW-verified multi-stick multi-core (Fix A/B chunked-prefill:
    128×768, 512×4096, 2048×2048/4096 all max_diff=0). maximum fails loud (no binary).
    """
    if op_id not in _SUPPORTED_OPS:
        pytest.xfail(
            f"{op_id}: no HW-verified SFP binary yet — codegen fails loud "
            "(UnsupportedPointwiseOpError, Part A). maximum tracked as a follow-on."
        )
    if shape_id not in _SINGLE_STICK_SHAPES and op_id not in _MULTISTICK_VERIFIED_OPS:
        pytest.xfail(
            f"{op_id} {shape_id}: multi-stick path not yet HW-verified for this op."
        )
    if boundary:
        pytest.xfail(
            f"{shape_id}: [{M},{N}] single-tile exceeds LX scratchpad (LBR0 pin) — "
            "needs tiling / chunk-large-tensors (tracked codegen follow-on)."
        )
    a = _small_pos((M, N))
    b = (_small_pos((M, N)) % 4 + 1)  # 1..4, keeps sums/products <= 1024
    out_dev = fn(a.to("spyre"), b.to("spyre")).cpu()
    out_ref = fn(a, b)
    _check(op_id, out_dev, out_ref)


# --- de-periodized (distinct-per-stick) binary regression (task #33) ----------
# Multi-stick binary shapes with operands whose per-stick period does NOT divide the
# 64-element stick, so a per-stick / per-core input FEED bug is CAUGHT, not masked. This
# is the guard that would have caught the sub input_y PatchInit bug (multi-core `b` read
# from stick 0 → a[stick]-b[stick%sticks_per_core]); the periodic `_small_pos` operands
# (periods 8/4, both divide 64) made every stick identical and hid it. See pointwise.md §12.
#
# NON-COMMUTATIVE op (sub) is essential: a+b/a*b absorb an operand swap/misfeed that a-b
# cannot, so `sub` is the sensitive canary. Each op gets DISTINCT periods on the two
# operands (both coprime-to-64) so neither a nor b can be silently frozen.
_DEPERIODIZED_BINARY = [
    ("add", lambda a, b: a + b),
    ("mul", lambda a, b: a * b),
    ("sub", lambda a, b: a - b),
]
_DEPERIODIZED_SHAPES = [
    ("decode_1x4096", 1, 4096),     # M=1 wide → 32-core, small sticks-per-core
    ("multi_128x4096", 128, 4096),  # multi-core multi-stick prefill chunk
]


@requires_hw
@pytest.mark.parametrize("shape_id,M,N", _DEPERIODIZED_SHAPES, ids=[s[0] for s in _DEPERIODIZED_SHAPES])
@pytest.mark.parametrize("op_id,fn", _DEPERIODIZED_BINARY, ids=[o[0] for o in _DEPERIODIZED_BINARY])
def test_pointwise_binary_distinct_per_stick(op_id, fn, shape_id, M, N):
    """Binary op with distinct-per-stick operands (period coprime-to-64) must be exact.

    Guards against per-stick/per-core feed bugs that periodic operands mask. a: period 20,
    b: period 7 — both coprime to the 64-stick width, and mul stays <= 20*7=140 <= 1024
    (DL16-exact), sub result in [-6, 19], add in [2, 27] — all DL16-exact.
    """
    a = _distinct_per_stick((M, N), period=20)   # 1..20 per stick
    b = _distinct_per_stick((M, N), period=7)     # 1..7 per stick
    out_dev = fn(a.to("spyre"), b.to("spyre")).cpu()
    out_ref = fn(a, b)
    md = (out_dev - out_ref).abs().max().item()
    assert md == 0.0, (
        f"{op_id} {shape_id} distinct-per-stick: max_diff={md} (expected 0.0). "
        "A nonzero diff here means an operand's per-stick/per-core feed is frozen or "
        "misindexed (the task-#33 sub PatchInit bug class)."
    )


# ---------------------------------------------------------------------------
# Ragged N (N not a multiple of elements_per_stick) — sub-stick support.
#
# WHY: N that is not a whole number of sticks (N%64 != 0 at 16-bit) is the sub-stick /
# ragged-N case. Inductor flattens [M,N] to a 1D grid and the torch-spyre bridge decomposes
# it as N'=64, M'=ceil(M·N/64) LOGICAL sticks.
#
# THE M>1 CATCH (root-caused + FIXED this session, #167): the naive 1D-flatten wrote
# ceil(M·N/64) LOGICAL sticks, but `.to("spyre")` row-pads each row to M·ceil(N/64) PHYSICAL
# sticks. When N%64 != 0 AND M>1, physical > logical, so the trailing physical sticks were
# NEVER WRITTEN and read back as 0 — the ragged tail dropped real data (it did NOT land inertly
# in padding; that earlier claim was wrong). FINGERPRINT (direct mechanism proof, 3×200): the
# wrong elements were cols 192-199 = the 200%64=8 partial tail of rows 1,2 ONLY; row 0 was
# correct (it aligns by coincidence) — exactly the "physical row-pad sticks unwritten" mechanism.
#
# FIX (#167): torch_spyre/_inductor/choices.py recovers the true (M,N) and row-pads the 1D
# flatten to m_val = M·ceil(N/64), so the op writes the physical stick count the tensor actually
# occupies. HW-VERIFIED max_diff=0.0 (add/mul/sub, neg/relu, fused single-input chains, and
# scalar-affine) × ragged M>1 {3×200,7×91,4×100,4×40,3×130,2×65} × eager AND compiled.
#   M==1: logical == physical (one row) → never had a drop → HW-passes. GREEN.
#   M>1 : row-padded flatten → HW-passes. GREEN.
#
# ⚠️ CACHE NOTE (#168): a choices.py source edit does NOT invalidate the Inductor FxGraphCache /
# Triton disk cache (keyed on FX graph + input meta, not choices source). A stale pre-fix kernel
# briefly made this look "unfixed" (max_diff=11 on a box with an old cached kernel) — a false
# "refuted" that was NOT a real geometry bug. The conftest fixture in tests/conftest.py
# (_clear_compile_caches_for_ragged) clears those caches before these tests so HW observes the
# current codegen. "eager" here is torch.compile(dynamic=False) under the hood (ops/eager.py),
# so both entry points share the choices.py path and the same cache discipline.
#
# Operands: _small_pos/_signed_small use arange%8 — over a flat 1D stream every element is
# distinct-per-position, so a dropped/duplicated tail element shows as a nonzero max_diff.
_RAGGED_N_SHAPES = [
    ("sub_stick_1x40",   1,   40),    # N<eps: single partial stick (24 tail), decode
    ("sub_stick_4x40",   4,   40),    # N<eps: multi-row sub-stick — M>1 (row-padded, #167 fixed)
    ("ragged_1x100",     1,   100),   # N>eps: 1 full + 36 tail, decode
    ("ragged_4x100",     4,   100),   # N>eps: multi-row ragged — M>1 (row-padded, #167 fixed)
    ("ragged_3x200",     3,   200),   # N>eps: 3 full + 8 tail — M>1 (row-padded, #167 fixed)
    ("ragged_7x91",      7,   91),    # N>eps: 1 full + 27 tail, prime dims — M>1 (#167 fixed)
    ("tiny_1x5",         1,   5),     # N<<eps: 5 valid, 59 tail (smallest real case)
]


@requires_hw
@pytest.mark.parametrize("shape_id,M,N", _RAGGED_N_SHAPES, ids=[s[0] for s in _RAGGED_N_SHAPES])
@pytest.mark.parametrize("op_id,fn", _UNARY_OPS, ids=[o[0] for o in _UNARY_OPS])
def test_pointwise_unary_ragged_n(op_id, fn, shape_id, M, N):
    """Unary pointwise op at ragged N (N % 64 != 0) must be exact — the tail must not
    corrupt the logical output. Regression-lock for the sub-stick support path (#167:
    M>1 row-padded flatten, HW-verified max_diff=0)."""
    if op_id not in _SUPPORTED_OPS:
        pytest.xfail(f"{op_id}: no HW-verified SFP binary yet (Part A fail-loud).")
    x = _signed_small((M, N)) if op_id == "relu" else _small_pos((M, N))
    out_dev = fn(x.to("spyre")).cpu()
    out_ref = fn(x)
    assert out_dev.shape == out_ref.shape, (
        f"{op_id} {shape_id}: device shape {tuple(out_dev.shape)} != logical "
        f"{tuple(out_ref.shape)} — stick-padding tail leaked into the logical shape."
    )
    _check(op_id, out_dev, out_ref)


@requires_hw
@pytest.mark.parametrize("shape_id,M,N", _RAGGED_N_SHAPES, ids=[s[0] for s in _RAGGED_N_SHAPES])
@pytest.mark.parametrize("op_id,fn", _BINARY_OPS, ids=[o[0] for o in _BINARY_OPS])
def test_pointwise_binary_ragged_n(op_id, fn, shape_id, M, N):
    """Binary pointwise op at ragged N (N % 64 != 0) must be exact, M==1 and M>1 alike.
    #167 (row-padded 1D flatten in choices.py) fixed the M>1 tail-element drop; HW-verified
    max_diff=0 for add/mul/sub across the ragged M>1 shapes. Regression-lock."""
    if op_id not in _SUPPORTED_OPS:
        pytest.xfail(f"{op_id}: no HW-verified SFP binary yet (Part A fail-loud).")
    a = _small_pos((M, N))
    b = (_small_pos((M, N)) % 4 + 1)  # 1..4, keeps sums/products <= 1024
    out_dev = fn(a.to("spyre"), b.to("spyre")).cpu()
    out_ref = fn(a, b)
    assert out_dev.shape == out_ref.shape, (
        f"{op_id} {shape_id}: device shape {tuple(out_dev.shape)} != logical "
        f"{tuple(out_ref.shape)} — stick-padding tail leaked into the logical shape."
    )
    _check(op_id, out_dev, out_ref)


def test_ragged_n_shape_model():
    """Hardware-independent: ragged N ceil-rounds to whole sticks (the sub-stick contract
    the codegen ceil-fix must honor). Guards the arithmetic without a board."""
    assert math.ceil(40 / _EPS) == 1     # N<eps → 1 partial stick
    assert math.ceil(100 / _EPS) == 2    # N>eps → 1 full + 1 tail stick
    assert math.ceil(200 / _EPS) == 4
    assert math.ceil(91 / _EPS) == 2
    assert math.ceil(5 / _EPS) == 1


# ---------------------------------------------------------------------------
# 3-dispatch-path coverage (mirrors tests/test_op_embedding.py _ENTRY_POINTS).
# A pointwise op can reach the device three ways, and they DO differ (embedding
# proved eager vs compiled dispatch route differently). Test all three:
#   aten     — torch.ops.aten.<op> directly (the registered device kernel)
#   eager    — the torch operator with no compile (a+b / a*b → same aten kernel)
#   compiled — torch.compile (Inductor graph capture → Triton TTIR → codegen)
# ---------------------------------------------------------------------------
_ENTRY_POINTS = ("aten", "eager", "compiled")

# Real-LLM shapes HW-verified THIS SESSION (2026-07-11/12): add/mul at multi-stick
# hidden and chunked-prefill up to the LX ceiling. (id, M, N, model/hidden note)
_LLM_BINARY_SHAPES = [
    ("decode_d4096",     1,    4096, "Granite-8B/Mistral-7B residual/SwiGLU, decode"),
    ("prefill_512x4096", 512,  4096, "d4096 chunked-prefill chunk (Fix A, unrolled)"),
    ("prefill_2048x2048", 2048, 2048, "Granite-2B/TinyLlama chunked-prefill (Fix A)"),
    ("prefill_2048x4096", 2048, 4096, "d4096 chunked-prefill (Fix B, looped L3, 128 chunks)"),
]
_ATEN_BINARY = {"add": torch.ops.aten.add, "mul": torch.ops.aten.mul}


def _run_binary_via(entry, op_id, fn, a_dev, b_dev):
    """Dispatch a binary pointwise op to the device via one of the 3 entry points."""
    if entry == "aten":
        return _ATEN_BINARY[op_id](a_dev, b_dev).cpu()
    if entry == "eager":
        return fn(a_dev, b_dev).cpu()          # a+b / a*b, no compile
    if entry == "compiled":
        return torch.compile(fn)(a_dev, b_dev).cpu()
    raise ValueError(entry)


@requires_hw
@pytest.mark.parametrize("entry", _ENTRY_POINTS)
@pytest.mark.parametrize("shape_id,M,N,note", _LLM_BINARY_SHAPES, ids=[s[0] for s in _LLM_BINARY_SHAPES])
@pytest.mark.parametrize("op_id", ["add", "mul"])
def test_pointwise_binary_dispatch_paths(op_id, shape_id, M, N, note, entry):
    """add/mul on real LLM shapes must be correct via ALL THREE dispatch paths
    (aten op / eager / torch.compile), matching the coverage embedding.md's test has.

    Shapes are the residual-add / SwiGLU-mul shapes HW-verified this session, incl.
    the Fix A (unrolled) and Fix B (looped L3) chunked-prefill regimes. All must be
    max_diff==0 with DL16-exact operands, regardless of how the op is dispatched.
    """
    fn = {"add": lambda a, b: a + b, "mul": lambda a, b: a * b}[op_id]
    a = _small_pos((M, N))
    b = (_small_pos((M, N)) % 4 + 1)     # 1..4 → sums/products <= 1024, DL16-exact
    out_dev = _run_binary_via(entry, op_id, fn, a.to("spyre"), b.to("spyre"))
    out_ref = fn(a, b)
    md = (out_dev - out_ref).abs().max().item()
    assert md == 0.0, (
        f"{op_id} {shape_id} via {entry}: max_diff={md} (expected 0.0). "
        f"[{note}]"
    )


def test_pointwise_shape_model():
    """Hardware-independent: the [M,N] → stick tiling the codegen path assumes.
    N tiles into ceil(N/64) sticks; M rows spread across cores. Guards the shape
    contract so it's validated in CI without a board.
    """
    assert math.ceil(64 / _EPS) == 1
    assert math.ceil(768 / _EPS) == 12
    assert math.ceil(4096 / _EPS) == 64


# ===========================================================================
# COMPILED entry-point counterparts for the standard unary / binary sweeps.
#
# WHY: the eager tests above run ONLY the bare op (`fn(x.to("spyre"))`). A real
# model reaches these ops through `torch.compile` (Inductor graph capture → Triton
# TTIR → codegen), which can SCHEDULE / FUSE differently than the per-op eager wrap
# — embedding proved eager vs compiled route differently, and add/mul are the only
# ops with explicit compiled coverage today (test_pointwise_binary_dispatch_paths).
# These mirror the eager sweeps exactly (same shapes, same gating, same DL16-exact
# known-answer) but drive `torch.compile(fn, dynamic=False)` so sub/neg/relu and the
# unary set get the compiled path pinned too. The user's hard requirement: every
# scenario in BOTH eager and compiled.
#
# torch.compile is the PRIMARY HW-verified path for these ops (pointwise.md: add/mul
# via all 3 dispatch paths; neg/relu/sub verified via torch.compile), so verified
# ops/shapes are expected-pass under the same _SUPPORTED_OPS / _MULTISTICK_VERIFIED_OPS
# / boundary gating as their eager siblings.
# ===========================================================================
@requires_hw
@pytest.mark.parametrize("shape_id,M,N,boundary", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("op_id,fn", _UNARY_OPS, ids=[o[0] for o in _UNARY_OPS])
def test_pointwise_unary_compiled(op_id, fn, shape_id, M, N, boundary):
    """Unary pointwise op via torch.compile(fn, dynamic=False) == CPU across real
    transformer shapes. Compiled counterpart of test_pointwise_unary (which is eager-
    only). Same gating; compiled is the primary HW-verified path for neg/relu."""
    if op_id not in _SUPPORTED_OPS:
        pytest.xfail(
            f"{op_id}: no HW-verified SFP binary yet — codegen fails loud "
            "(UnsupportedPointwiseOpError, Part A). Build+verify its binary to enable."
        )
    if shape_id not in _SINGLE_STICK_SHAPES and op_id not in _MULTISTICK_VERIFIED_OPS:
        pytest.xfail(
            f"{op_id} {shape_id}: multi-stick path not yet HW-verified for this unary op."
        )
    if boundary:
        pytest.xfail(
            f"{shape_id}: [{M},{N}] single-tile exceeds LX scratchpad (LBR0 pin) — "
            "needs tiling / chunk-large-tensors (tracked codegen follow-on)."
        )
    x = _signed_small((M, N)) if op_id == "relu" else _small_pos((M, N))
    compiled = torch.compile(fn, dynamic=False)
    out_dev = compiled(x.to("spyre")).cpu()
    out_ref = fn(x)
    assert out_dev.shape == out_ref.shape, (
        f"{op_id} {shape_id}: device shape {tuple(out_dev.shape)} != {tuple(out_ref.shape)}"
    )
    _check(op_id, out_dev, out_ref)


@requires_hw
@pytest.mark.parametrize("shape_id,M,N,boundary", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("op_id,fn", _BINARY_OPS, ids=[o[0] for o in _BINARY_OPS])
def test_pointwise_binary_compiled(op_id, fn, shape_id, M, N, boundary):
    """Binary pointwise op via torch.compile(fn, dynamic=False) == CPU. Compiled
    counterpart of test_pointwise_binary (eager-only). Adds explicit compiled coverage
    for sub (the dispatch-paths test covers only add/mul); maximum stays fail-loud."""
    if op_id not in _SUPPORTED_OPS:
        pytest.xfail(
            f"{op_id}: no HW-verified SFP binary yet — codegen fails loud "
            "(UnsupportedPointwiseOpError, Part A). maximum tracked as a follow-on."
        )
    if shape_id not in _SINGLE_STICK_SHAPES and op_id not in _MULTISTICK_VERIFIED_OPS:
        pytest.xfail(
            f"{op_id} {shape_id}: multi-stick path not yet HW-verified for this op."
        )
    if boundary:
        pytest.xfail(
            f"{shape_id}: [{M},{N}] single-tile exceeds LX scratchpad (LBR0 pin) — "
            "needs tiling / chunk-large-tensors (tracked codegen follow-on)."
        )
    a = _small_pos((M, N))
    b = (_small_pos((M, N)) % 4 + 1)  # 1..4, keeps sums/products <= 1024
    compiled = torch.compile(fn, dynamic=False)
    out_dev = compiled(a.to("spyre"), b.to("spyre")).cpu()
    out_ref = fn(a, b)
    assert out_dev.shape == out_ref.shape, (
        f"{op_id} {shape_id}: device shape {tuple(out_dev.shape)} != {tuple(out_ref.shape)}"
    )
    _check(op_id, out_dev, out_ref)


# ===========================================================================
# FUSED single-input linear elementwise CHAINS (the generic-fused-elementwise work).
#
# A chain of scalar-const arith over ONE tensor input — one tt.load, a straight
# chain of scalar-const arith nodes, one tt.store — is lowered by the codegen
# `pointwise_fused` path (backend/ttir_analyzer.py::_match_fused_chain) to a single
# SFP instruction chain latched through one scratch LRF. This CLOSED the
# [[generic-fused-elementwise-gap]] for the single-input linear case (task #85).
#
# ENTRY-POINT DIVERGENCE (the reason both entries matter here):
#   compiled — Inductor FUSES `2*x+3` into ONE kernel → the pointwise_fused SFP-chain
#              path (HW-verified 2026-07-16, see below).
#   eager    — bare `2*x+3` is TWO separate aten ops, each wrapped in its own per-op
#              torch.compile (register_torch_compile_kernel) → two SEQUENTIAL
#              scalar-affine ops (mul_c then add_c), NOT the fused path. Both
#              scalar-affine subforms are HW-verified multi-stick (scalar-affine.md).
#   So the SAME source expression reaches the device by two DIFFERENT mechanisms —
#   exactly the class of divergence that hid coverage gaps before.
#
# HW-VERIFIED ENVELOPE (codegen/docs/op-milestones/fused-elementwise.md, 2026-07-16,
# tmhoangt-spyre-dev-bob-quick, torch.compile, fp16, de-periodized operand max|err|=0):
#   chains: 2*x+3, 5*x-2, (x-5)*2+1 (fma-family) AND relu(x-5) (fminmax-in-chain)
#   shapes: [1,64], [128,768], [128,4096], [512,4096]
# Everything OUTSIDE that envelope (other chains the classifier accepts but that were
# not on the HW probe; other shapes incl. decode 1×4096, wider, ragged) is xfail-strict:
# not-yet-HW-verified, NOT assumed-pass ([[feedback-tests-before-hw-proof]]). >2-node
# and multi-CORE fused coverage is explicitly listed as pending in the milestone doc.
# ===========================================================================
# (id, fn, needs_signed_input, hw_verified_chain). fn is single-input.
_FUSED_SINGLE_INPUT_CHAINS = [
    ("fma_2x_plus_3",       lambda x: 2.0 * x + 3.0,        False, True),   # [mul, add] HW-verified
    ("fma_5x_minus_2",      lambda x: 5.0 * x - 2.0,        False, True),   # [mul, sub_r] HW-verified
    ("chain_xm5_x2_plus1",  lambda x: (x - 5.0) * 2.0 + 1.0, False, True),  # 3-node; was the src2=ONE hang
    ("relu_xm5",            lambda x: torch.relu(x - 5.0),  False, True),   # fminmax-in-chain HW-verified
    # div-by-const→mul(1/c), sub_l, and neg folds — HW-verified 2026-07-18 (max_diff=0 at all
    # _FUSED_HW_VERIFIED_SHAPES incl. decode_1x4096, eager + compiled, cache-cleared).
    ("div_x_by_2",          lambda x: x / 2.0,              False, True),   # divf-by-const → mul 0.5 fold
    ("chain_3_minus_x_x2",  lambda x: (3.0 - x) * 2.0,      False, True),   # sub_l then mul
    ("chain_neg_plus_3",    lambda x: torch.neg(x) + 3.0,   False, True),   # neg then add
]

# The chains + shapes silicon-verified max|err|=0. Base set 2026-07-16 (fused-elementwise.md);
# div/sub_l/neg folds added 2026-07-18 (all max_diff=0 at the 4 base shapes, cache-cleared).
# NOTE: fused_decode_1x4096 is deliberately NOT here — (x-5)*2+1 compiled is WRONG there
# (max_diff=8, isolated/cache-cleared 2026-07-18, a real 3-node×decode-wide bug, #170). The
# verified set is a (chain × shape) product, so a shape only enters once ALL verified chains
# pass at it. decode_1x4096 stays unverified until that bug is fixed.
_FUSED_HW_VERIFIED_CHAINS = {
    "fma_2x_plus_3", "fma_5x_minus_2", "chain_xm5_x2_plus1", "relu_xm5",
    "div_x_by_2", "chain_3_minus_x_x2", "chain_neg_plus_3",
}
_FUSED_HW_VERIFIED_SHAPES = {"fused_1x64", "fused_128x768", "fused_128x4096", "fused_512x4096"}

# (id, M, N). The four HW-verified shapes PLUS out-of-envelope shapes (decode-wide,
# ragged, sub-stick) that must be honestly xfail until probed.
_FUSED_SHAPES = [
    ("fused_1x64",       1,    64),     # HW-verified (single stick)
    ("fused_128x768",    128,  768),    # HW-verified (GPT-2 hidden, multi-stick)
    ("fused_128x4096",   128,  4096),   # HW-verified (Llama/Mistral hidden)
    ("fused_512x4096",   512,  4096),   # HW-verified (512-tok prefill chunk)
    ("fused_decode_1x4096", 1, 4096),   # NOT in HW set: decode wide (UR-band history, §11)
    ("fused_ragged_3x200", 3,  200),    # NOT in HW set: ragged N × M>1 (fused × ragged blind spot)
    ("fused_ragged_5x130", 5,  130),    # NOT in HW set: ragged N × M>1
]


def _run_unary_via(entry, fn, x_dev):
    """Dispatch a single-input expression to the device via eager or compiled."""
    if entry == "eager":
        return fn(x_dev).cpu()
    if entry == "compiled":
        return torch.compile(fn, dynamic=False)(x_dev).cpu()
    raise ValueError(entry)


@requires_hw
@pytest.mark.parametrize("entry", ("eager", "compiled"))
@pytest.mark.parametrize("shape_id,M,N", _FUSED_SHAPES, ids=[s[0] for s in _FUSED_SHAPES])
@pytest.mark.parametrize(
    "chain_id,fn,needs_signed,hw_verified",
    _FUSED_SINGLE_INPUT_CHAINS,
    ids=[c[0] for c in _FUSED_SINGLE_INPUT_CHAINS],
)
def test_pointwise_fused_single_input_chain(
    chain_id, fn, needs_signed, hw_verified, shape_id, M, N, entry, request
):
    """Fused single-input linear arith chain (`2*x+3`, `relu(x-5)`, …) == CPU, DL16-exact.

    compiled → the codegen pointwise_fused SFP-chain path; eager → sequential
    scalar-affine ops. Only the four chains × four shapes silicon-verified 2026-07-16 are
    expected-pass; every other (chain, shape) is xfail-strict so it auto-flips (XPASS →
    strict failure) once the HW sweep extends the verified envelope."""
    # Ragged N × M>1 for a fused single-input chain is HW-VERIFIED (#167): the chain produces a
    # binary and rides the choices.py row-padded flatten, so the tail no longer drops. Sweep
    # 2026-07-17 (caches cleared): 2*x+3 / 5*x-2 / relu(x-5) × {3×200,4×100,7×91} × eager+compiled
    # all max_diff=0.0. So the ragged×M>1 case is EXPECTED-PASS and takes NO xfail marker below.
    verified = (hw_verified and shape_id in _FUSED_HW_VERIFIED_SHAPES) or (
        N % _EPS != 0 and M > 1
    )
    if not verified:
        # decode_1x4096 (M=1, N=4096, decode-wide) is NOT exhaustively swept and has a CONFIRMED
        # real bug — (x-5)*2+1 compiled = max_diff=8 there (#170). Some chains pass, some don't, so
        # it is neither a verified shape nor a hard-fail: use a NON-strict xfail so a passing chain
        # (XPASS) does not turn the suite red and the (x-5)*2+1 failure is tolerated pending #170.
        strict = shape_id != "fused_decode_1x4096"
        if chain_id not in _FUSED_HW_VERIFIED_CHAINS:
            reason = (
                f"{chain_id}: classifier accepts this fold but it was NOT on the "
                "2026-07-16 fused-elementwise HW probe (fused-elementwise.md §6). "
                "Flip when a HW known-answer passes (max_diff==0)."
            )
        elif shape_id == "fused_decode_1x4096":
            reason = (
                f"{chain_id} {shape_id}: decode-wide (M=1,N=4096) is not exhaustively "
                "HW-swept; (x-5)*2+1 compiled is a known real bug here (#170). Non-strict "
                "xfail until #170 is fixed and the shape is swept for all chains."
            )
        else:
            reason = (
                f"{chain_id} {shape_id}: fused chain HW-verified only at "
                "[1,64]/[128,768]/[128,4096]/[512,4096] single-core (2026-07-16) plus "
                "ragged M>1 (#167, 2026-07-17); this shape (aligned multi-core) "
                "is out of the verified envelope. >2-node/multi-core fused are pending "
                "(fused-elementwise.md §6)."
            )
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=strict))
    x = _signed_small((M, N)) if needs_signed else _small_pos((M, N))
    out_dev = _run_unary_via(entry, fn, x.to("spyre"))
    out_ref = fn(x)
    assert out_dev.shape == out_ref.shape, (
        f"{chain_id} {shape_id} via {entry}: device shape {tuple(out_dev.shape)} != "
        f"{tuple(out_ref.shape)}"
    )
    md = (out_dev - out_ref).abs().max().item()
    assert md == 0.0, f"{chain_id} {shape_id} via {entry}: max_diff={md} (expected 0.0, DL16-exact)"


# ===========================================================================
# MULTI-TENSOR fused DAGs — the OPEN [[generic-fused-elementwise-gap]] boundary (#99).
#
# `(a*b)+c`, `a+b+c`, `(a-b)*c`, `a*b+scalar` have MORE THAN ONE tensor input. Under
# torch.compile Inductor fuses them into ONE kernel with >1 tt.load; the codegen fused
# classifier (_match_fused_chain restriction 1: "exactly ONE tt.load feeds the chain")
# rejects them → pointwise_unsupported → fail-loud. A generic multi-input / tree-shaped
# arith-DAG → SFP-chain lowering (multi-tensor segment packing) is the NEXT architectural
# step (#99, fused-elementwise.md §6 / generic-fused-elementwise-gap memory), NOT built.
#
# This is a SEPARATE gap from #167 (ragged) — failing loud here is CORRECT behaviour for an
# unbuilt path, not a stick-count bug. So these are PLAIN xfail (non-strict): they are the
# living contract for the multi-tensor lowering, and they must NOT be expected-pass (the
# op is unbuilt) NOR xfail-strict (a plain xfail tolerates both the current fail-loud AND a
# future correct result — when #99 lands and the DAG lowers, the case XPASSes and we
# promote it to a hard assertion by removing the mark). Both entry points are covered:
#   compiled — Inductor fuses → >1 tt.load → pointwise_unsupported fail-loud.
#   eager    — bare `(a*b)+c` is separate aten ops, each a verified binary op, so it may
#              compute correctly OR fail depending on how the per-op wrap composes — plain
#              xfail tolerates either until the composition is HW-probed.
# The body runs a DL16-exact known-answer so that IF the path ever produces a result, its
# correctness is checked (an XPASS is only meaningful if the numbers are also right).
# ===========================================================================
# (id, fn, arity) — arity = number of DEVICE tensor inputs.
_MULTI_TENSOR_FUSED_DAGS = [
    ("mul_add_abc",      lambda a, b, c: (a * b) + c,   3),   # (a*b)+c
    ("add3_abc",         lambda a, b, c: a + b + c,     3),   # a+b+c
    ("sub_mul_abc",      lambda a, b, c: (a - b) * c,   3),   # (a-b)*c
    ("mul_add_scalar",   lambda a, b: a * b + 7.0,      2),   # a*b + const (still 2 tensor loads)
]
_MULTI_TENSOR_99_REASON = (
    "#99 multi-tensor fused DAG (>1 tt.load) — generic multi-input arith-DAG → SFP-chain "
    "lowering (segment packing) NOT built; the codegen classifier rejects it "
    "(_match_fused_chain restriction 1) → pointwise_unsupported fail-loud BY DESIGN. "
    "Separate from #167 (ragged): failing loud on an unbuilt path is correct, not a bug. "
    "Plain (non-strict) xfail — promote to a hard known-answer assertion when #99 lands."
)


def _fused_dag_operands(arity, M, N):
    """DL16-exact operands whose fused-DAG result stays <= 1024.
    a,b,c small (1..4) so (a*b)+c <= 16+4, a+b+c <= 12, (a-b)*c in [-12,12], a*b+7 <= 23."""
    a = (_small_pos((M, N)) % 4 + 1)
    b = (_small_pos((M, N)) % 3 + 1)
    c = (_small_pos((M, N)) % 4 + 1)
    return [a, b, c][:arity]


@requires_hw
@pytest.mark.xfail(reason=_MULTI_TENSOR_99_REASON, strict=False)
@pytest.mark.parametrize("entry", ("eager", "compiled"))
@pytest.mark.parametrize("shape_id,M,N", [("dag_1x64", 1, 64), ("dag_128x768", 128, 768)],
                         ids=["dag_1x64", "dag_128x768"])
@pytest.mark.parametrize("dag_id,fn,arity", _MULTI_TENSOR_FUSED_DAGS, ids=[d[0] for d in _MULTI_TENSOR_FUSED_DAGS])
def test_pointwise_fused_multi_tensor(dag_id, fn, arity, shape_id, M, N, entry):
    """Multi-tensor fused DAG (`(a*b)+c`, `a+b+c`, …) via eager AND compiled. PLAIN xfail
    on #99 (multi-tensor segment-packing NOT built — fail-loud by design, distinct from the
    #167 ragged bug). DL16-exact known-answer body so an eventual XPASS is a correct result,
    not just a non-raise. Promote to a hard assertion when #99 lands + HW-verifies."""
    operands_ref = _fused_dag_operands(arity, M, N)
    operands_dev = [t.to("spyre") for t in operands_ref]
    if entry == "eager":
        out_dev = fn(*operands_dev).cpu()
    else:
        out_dev = torch.compile(fn, dynamic=False)(*operands_dev).cpu()
    out_ref = fn(*operands_ref)
    assert out_dev.shape == out_ref.shape
    md = (out_dev - out_ref).abs().max().item()
    assert md == 0.0, f"{dag_id} {shape_id} via {entry}: max_diff={md} (expected 0.0, DL16-exact)"


# ===========================================================================
# RAGGED N × M>1 — the exact blind spot that let the ragged-N M>1 bug hide.
#
# The existing test_pointwise_{unary,binary}_ragged_n cover ragged N (incl. M>1: 4×40,
# 4×100, 3×200, 7×91) but ONLY via the eager `fn(x.to("spyre"))` entry point. This block
# adds the MISSING axes for ragged × M>1:
#   (1) NEW ragged×M>1 shapes not covered anywhere yet: 5×130, 33×100, 2×65.
#   (2) the explicit torch.compile entry point (existing ragged tests are eager-only).
#
# HW-PROVEN FAILING (this session, #167/#166 — see the physical>logical stick-drop
# mechanism at the _RAGGED_N_SHAPES comment): ragged N × M>1 drops the trailing physical
# stick(s) → reads 0. The binary probe measured add=11 / mul=28 / sub=7 (operand maxima),
# reproduced BYTE-IDENTICAL pre-SP1/SP2 (codegen@7718bdc) AND with "Fix A" (Inductor
# choices.py) present, so PRE-EXISTING — the earlier committed-green ragged M>1 rows were a
# periodicity-masked / unverified-batch false-green. Fingerprint (direct mechanism proof):
# for 3×200 the wrong elements are cols 192-199 = the 200%64=8 partial tail of rows 1,2 ONLY
# (row 0 aligns by coincidence) — exactly the "physical row-pad sticks unwritten" prediction.
# The bug is in the SHARED .to(spyre) device-layout / stick-write layer (eager, which never
# routes through choices.py, fails identically at 11), NOT the Inductor tiling hook — so
# Fix A was REFUTED and #167 is reopened for a real fix at the shared layer.
# The drop is OP-AGNOSTIC (physical row-pad sticks never written, independent of the compute
# op), so unary neg/relu M>1 is PREDICTED to drop identically — but is NOT yet HW-measured,
# so it is tracked under #166 (op-independent tail-drop), distinct from #167 (measured for
# binary add/mul/sub). Both should ride the same real shared-layer fix and auto-flip together;
# the distinct labels surface a unary-specific need if unary does NOT auto-flip with binary.
# Every case here is xfail-STRICT so it auto-flips (XPASS → strict failure) once a real fix
# HW-verifies. (No green flip is pending — Fix A did not fix it.)
# ===========================================================================
_RAGGED_M_GT1_SHAPES = [
    ("ragged_5x130",   5,   130),   # N>eps ragged (2 full + 2 tail) × M=5
    ("ragged_33x100",  33,  100),   # N>eps ragged (1 full + 36 tail) × M=33 (multi-core)
    ("subsz_2x65",     2,   65),    # N just over one stick (1 full + 1 tail) × M=2
]
# The 5 ops HW-verified for ALIGNED shapes; ragged×M>1 is the #167/#166 residual for these.
_RAGGED_M_GT1_UNARY = [("neg", lambda x: torch.neg(x)), ("relu", lambda x: torch.relu(x))]
_RAGGED_M_GT1_BINARY = [
    ("add", lambda a, b: a + b),
    ("mul", lambda a, b: a * b),
    ("sub", lambda a, b: a - b),
]
# Ragged M>1 is HW-VERIFIED FIXED (#167, choices.py row-padded 1D flatten): sweep 2026-07-17
# (caches cleared) gave max_diff=0.0 for binary add/mul/sub AND unary neg/relu × {3×200,7×91,
# 4×100,4×40,3×130,2×65} × eager+compiled. Both dedicated tests below are now EXPECTED-PASS
# (no xfail markers) and serve as the regression-lock. The de-periodized operands (period 20/7,
# coprime-to-64) additionally guard against a per-stick/per-core feed regression.


@requires_hw
@pytest.mark.parametrize("entry", ("eager", "compiled"))
@pytest.mark.parametrize("shape_id,M,N", _RAGGED_M_GT1_SHAPES, ids=[s[0] for s in _RAGGED_M_GT1_SHAPES])
@pytest.mark.parametrize("op_id,fn", _RAGGED_M_GT1_UNARY, ids=[o[0] for o in _RAGGED_M_GT1_UNARY])
def test_pointwise_unary_ragged_m_gt1(op_id, fn, shape_id, M, N, entry):
    """Unary neg/relu at ragged N × M>1 via eager AND compiled — the coverage gap that let
    the ragged M>1 bug hide, now HW-verified fixed (#167 row-padded flatten, max_diff=0)."""
    x = _signed_small((M, N)) if op_id == "relu" else _small_pos((M, N))
    out_dev = _run_unary_via(entry, fn, x.to("spyre"))
    out_ref = fn(x)
    assert out_dev.shape == out_ref.shape, (
        f"{op_id} {shape_id} via {entry}: device shape {tuple(out_dev.shape)} != "
        f"{tuple(out_ref.shape)} — ragged tail leaked into logical shape."
    )
    md = (out_dev - out_ref).abs().max().item()
    assert md == 0.0, f"{op_id} {shape_id} via {entry}: max_diff={md} (expected 0.0)."


@requires_hw
@pytest.mark.parametrize("entry", ("eager", "compiled"))
@pytest.mark.parametrize("shape_id,M,N", _RAGGED_M_GT1_SHAPES, ids=[s[0] for s in _RAGGED_M_GT1_SHAPES])
@pytest.mark.parametrize("op_id,fn", _RAGGED_M_GT1_BINARY, ids=[o[0] for o in _RAGGED_M_GT1_BINARY])
def test_pointwise_binary_ragged_m_gt1(op_id, fn, shape_id, M, N, entry):
    """Binary add/mul/sub at ragged N × M>1 via eager AND compiled — the ragged M>1 case that
    Fix A #167 (choices.py row-padded flatten) fixed; HW-verified max_diff=0. Distinct-per-stick
    operands (period 20 / 7, both coprime-to-64) so a per-stick/per-core feed regression is
    CAUGHT.

    sub (non-commutative) is the sensitive canary: an operand swap/misfeed that add/mul
    absorb changes a-b. Result stays DL16-exact: mul<=140, sub in [-6,19], add in [2,27]."""
    a = _distinct_per_stick((M, N), period=20)
    b = _distinct_per_stick((M, N), period=7)
    if entry == "eager":
        out_dev = fn(a.to("spyre"), b.to("spyre")).cpu()
    else:
        out_dev = torch.compile(fn, dynamic=False)(a.to("spyre"), b.to("spyre")).cpu()
    out_ref = fn(a, b)
    assert out_dev.shape == out_ref.shape, (
        f"{op_id} {shape_id} via {entry}: device shape {tuple(out_dev.shape)} != "
        f"{tuple(out_ref.shape)} — ragged tail leaked into logical shape."
    )
    md = (out_dev - out_ref).abs().max().item()
    assert md == 0.0, (
        f"{op_id} {shape_id} via {entry} distinct-per-stick: max_diff={md} (expected 0.0). "
        "A nonzero diff means the ragged M>1 1D-flatten stick-count (#167) is still wrong "
        "or a per-stick feed is frozen."
    )


def test_ragged_m_gt1_shape_model():
    """Hardware-independent: the NEW ragged×M>1 shapes ceil to whole sticks and are
    genuinely ragged (N % 64 != 0) with M>1. Guards the blind-spot table without a board."""
    for _id, M, N in _RAGGED_M_GT1_SHAPES:
        assert M > 1, f"{_id}: ragged×M>1 table must have M>1"
        assert N % _EPS != 0, f"{_id}: N={N} must be ragged (not a stick multiple)"
    assert math.ceil(130 / _EPS) == 3      # 5x130: 2 full + tail
    assert math.ceil(100 / _EPS) == 2      # 33x100: 1 full + tail
    assert math.ceil(65 / _EPS) == 2       # 2x65: 1 full + 1-elem tail


def test_fused_hw_verified_envelope_is_honest():
    """Contract guard (HW-independent): the fused expected-pass envelope must match what
    silicon proved — the 4 chains + 4 shapes verified 2026-07-16, PLUS the div/sub_l/neg
    folds verified 2026-07-18 (all max_diff=0 at the 4 base shapes, cache-cleared).
    fused_decode_1x4096 is deliberately EXCLUDED: (x-5)*2+1 is wrong there (#170).
    Prevents silently widening expected-pass beyond what silicon proved."""
    assert _FUSED_HW_VERIFIED_CHAINS == {
        "fma_2x_plus_3", "fma_5x_minus_2", "chain_xm5_x2_plus1", "relu_xm5",
        "div_x_by_2", "chain_3_minus_x_x2", "chain_neg_plus_3",
    }
    assert _FUSED_HW_VERIFIED_SHAPES == {
        "fused_1x64", "fused_128x768", "fused_128x4096", "fused_512x4096",
    }
    # Every HW-verified chain id must exist in the chain table with hw_verified=True.
    verified_in_table = {c[0] for c in _FUSED_SINGLE_INPUT_CHAINS if c[3]}
    assert verified_in_table == _FUSED_HW_VERIFIED_CHAINS
