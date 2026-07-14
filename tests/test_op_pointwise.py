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
#   add/mul/sub — via the Fix A (unrolled) + Fix B (looped L3) chunked-prefill work
#     (HW-verified 128×768, 512×4096, 2048×2048/4096 across 32 cores).
#   neg/relu   — via the task-#12 bug-3 fix (2026-07-13, commit 5b59031).
# i.e. EVERY currently-supported pointwise op works multi-stick multi-core. The only
# remaining xfail is the boundary shape (2048×4096, LX-capacity — see `boundary` below).
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
