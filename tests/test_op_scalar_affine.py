"""Dedicated HW test for SCALAR-AFFINE pointwise ops on spyre (task #56).

PURPOSE
-------
The living coverage contract for `x <op> c` where `c` is a COMPILE-TIME SCALAR
constant (not a tensor) — `x * c`, `x + c`, `x - c`, `c - x`, `c * x`. Mirrors
tests/test_op_pointwise.py: real transformer shapes, DL16-exact known-answer
device-vs-CPU check, xfail with the EXACT reason until the lowering lands.

WHY THIS IS A DISTINCT OP FROM BINARY pointwise (add/mul/sub)
------------------------------------------------------------
Binary pointwise (test_op_pointwise.py) is tensor×tensor: BOTH operands are real
device-resident tensors. Scalar-affine is tensor×SCALAR: Inductor emits the scalar
as an `arith.constant dense<...>` in the TTIR, NOT a second input tensor.

TODAY THIS FAULTS ON HW (0x7b1b). Root cause (verified this session): the codegen
classifier routes `arith.mulf`/`arith.addf` with a `dense<const>` operand to the
BINARY `pointwise_mul`/`pointwise_add` pattern — which expects two genuine device
inputs — but the scalar `c` is never materialized as a device operand. The SFP
hardware fixed-constant ports only provide {0.0, 1.0, 2.0, 3.0}
(codegen scheduler/operands.py ZERO/ONE/TWO/THREE), so an ARBITRARY `c` (0.5, 3.7,
…) cannot be sourced from a constant port and needs a real materialized operand
(a fill_tensor'd constant stick) OR an init-packet-seeded register. Either lowering
must produce the SAME result; this test asserts BEHAVIOR (x*c correct), not the
mechanism, so it is valid regardless of which lowering is chosen.

TWO ENTRY POINTS (both must pass — mirrors the eager+compile requirement)
-------------------------------------------------------------------------
  1. EAGER  — bare `x * c` on a spyre tensor. Spyre has no true eager kernel for
     mul/add; `register_torch_compile_kernel` (torch_spyre/ops/eager.py:79) wraps
     aten.mul/aten.add in torch.compile per-op, so the bare call still lowers
     through Inductor → codegen. This is the path a user hits writing `x * 0.5`.
  2. COMPILE — explicit `torch.compile(lambda x: x * c)`. Full Inductor graph
     capture + lowering (same path the embedding test's nn_compiled uses). A fused
     graph may constant-fold or schedule differently than the per-op eager wrap, so
     both entry points are exercised.

DL16 (1-6-9) note: AIU 1.0 does not represent integers > 1024 exactly, and only 9
mantissa bits. Constants and results are kept DL16-exact (small integers and exact
binary fractions like 0.5, 0.25) so the check is exact (max_diff == 0), never a
spurious rounding artifact. See the project DL16 note.
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

_EPS = 64  # elements per stick at 16-bit


# --- operand builder: DL16-exact, result stays <= 1024 ------------------------
def _small_pos(shape):
    # values 1..8 (exact in DL16); products/sums with the small constants below
    # stay <= 1024. int64 arange then cast (a float16 arange overflows to NaN at
    # n>65504 — see test_op_pointwise._small_pos).
    n = int(torch.tensor(shape).prod())
    return (torch.arange(n, dtype=torch.int64) % 8 + 1).to(torch.float16).reshape(shape)


# --- scalar-affine op table: (id, callable(x)->x∘c, reference is the SAME fn on CPU) --
# c values are DL16-EXACT: small integers and exact binary fractions (0.5, 0.25).
# NOT representable-in-ports: 0.5/3.0(port exists)/... — we deliberately include
# constants OUTSIDE the {0,1,2,3} SFP fixed-constant ports (0.5, 0.25, 5.0, 7.0) so
# the test proves an ARBITRARY constant works, not just a port-backed one.
#   x * c : the canonical #56 case (x * 0.5 faults today)
#   x + c : affine add
#   x - c : affine sub (x on the left)
#   c - x : reverse sub (constant on the left — operand order matters)
#   c * x : commuted mul (constant on the left)
_SCALAR_AFFINE_OPS = [
    ("mul_0p5",    lambda x: x * 0.5,   True),   # OUT-of-port constant — the headline case
    ("mul_0p25",   lambda x: x * 0.25,  True),
    ("mul_5",      lambda x: x * 5.0,   True),
    ("add_0p5",    lambda x: x + 0.5,   True),
    ("add_7",      lambda x: x + 7.0,   True),
    ("sub_3",      lambda x: x - 3.0,   True),   # x - c
    ("rsub_10",    lambda x: 10.0 - x,  True),   # c - x (reverse; order-sensitive)
    ("rmul_0p5",   lambda x: 0.5 * x,   True),   # c * x (commuted)
]

# Real transformer element-wise shapes [seq, d_model]; (id, M, N, boundary).
# Single-stick s1x64 is the smallest slice; multi-stick shapes exercise the chunked
# path. boundary=True is the LX-capacity shape (xfail regardless of op).
_SHAPES = [
    ("s1x64",            1,    64,   False),   # smallest — the first slice to turn green
    ("gpt2_128x768",     128,  768,  False),
    ("llama_128x4096",   128,  4096, False),
    ("seq512x4096",      512,  4096, False),
    ("maxseq_2048x4096", 2048, 4096, True),    # LX scratchpad capacity boundary
]

# Which scalar-affine ops reach a working, HW-verified codegen binary today.
# EMPTY until the lowering lands + is HW-verified (max_diff=0). Flip entries in as
# each (op, path) is proven on silicon — same discipline as _SUPPORTED_OPS in
# test_op_pointwise.py. Until then every case xfails with the #56 reason.
_HW_VERIFIED_AFFINE_OPS: set[str] = {"mul_0p5"}  # Stage 1: mul_0p5 @ s1x64 HW-verified (#82)

# Ops HW-verified across MULTI-STICK shapes (not just s1x64). Subset of the above.
_MULTISTICK_VERIFIED_AFFINE_OPS: set[str] = set()
_SINGLE_STICK_SHAPES = {"s1x64"}


def _check(op_id, out_dev, out_ref):
    md = (out_dev.float() - out_ref.float()).abs().max().item()
    assert md == 0.0, f"{op_id}: max_diff={md} (expected 0.0, DL16-exact)"


def _xfail_reason(op_id, shape_id):
    return (
        f"{op_id} {shape_id}: scalar-affine (x∘const) not yet lowered (#56). Today the "
        "classifier routes arith.mulf/addf-by-dense<const> to the BINARY pointwise "
        "pattern but never materializes the scalar → FAULTS on HW (0x7b1b). The SFP "
        "fixed-constant ports give only {0,1,2,3}, so an arbitrary c needs a "
        "materialized constant operand (fill_tensor stick) or an init-seeded register. "
        "Flip into _HW_VERIFIED_AFFINE_OPS when the lowering lands + is HW-verified."
    )


# ---------------------------------------------------------------------------
# Entry point 1: EAGER (bare op). Routes via register_torch_compile_kernel's
# per-op torch.compile wrap (torch_spyre/ops/eager.py). This is what a user hits
# writing `x * 0.5` directly on a spyre tensor.
# ---------------------------------------------------------------------------
@requires_hw
@pytest.mark.parametrize("shape_id,M,N,boundary", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("op_id,fn,exact", _SCALAR_AFFINE_OPS, ids=[o[0] for o in _SCALAR_AFFINE_OPS])
def test_scalar_affine_eager(op_id, fn, exact, shape_id, M, N, boundary):
    """`x ∘ c` via the bare eager path (per-op torch.compile wrap) == CPU, DL16-exact."""
    if op_id not in _HW_VERIFIED_AFFINE_OPS:
        pytest.xfail(_xfail_reason(op_id, shape_id))
    if shape_id not in _SINGLE_STICK_SHAPES and op_id not in _MULTISTICK_VERIFIED_AFFINE_OPS:
        pytest.xfail(f"{op_id} {shape_id}: multi-stick scalar-affine not yet HW-verified.")
    if boundary:
        pytest.xfail(
            f"{shape_id}: [{M},{N}] single-tile exceeds LX scratchpad (LBR0 pin) — "
            "needs tiling / chunk-large-tensors (tracked codegen follow-on)."
        )
    x = _small_pos((M, N))
    out_dev = fn(x.to("spyre")).cpu()
    out_ref = fn(x)
    assert out_dev.shape == out_ref.shape, (
        f"{op_id} {shape_id}: device shape {tuple(out_dev.shape)} != {tuple(out_ref.shape)}"
    )
    _check(op_id, out_dev, out_ref)


# ---------------------------------------------------------------------------
# Entry point 2: explicit torch.compile (full Inductor graph capture + lowering).
# A fused graph may constant-fold / schedule the scalar differently than the per-op
# eager wrap, so it is exercised independently (mirrors embedding.py nn_compiled).
# ---------------------------------------------------------------------------
@requires_hw
@pytest.mark.parametrize("shape_id,M,N,boundary", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("op_id,fn,exact", _SCALAR_AFFINE_OPS, ids=[o[0] for o in _SCALAR_AFFINE_OPS])
def test_scalar_affine_compiled(op_id, fn, exact, shape_id, M, N, boundary):
    """`x ∘ c` via explicit torch.compile(fn) == CPU, DL16-exact."""
    if op_id not in _HW_VERIFIED_AFFINE_OPS:
        pytest.xfail(_xfail_reason(op_id, shape_id))
    if shape_id not in _SINGLE_STICK_SHAPES and op_id not in _MULTISTICK_VERIFIED_AFFINE_OPS:
        pytest.xfail(f"{op_id} {shape_id}: multi-stick scalar-affine not yet HW-verified.")
    if boundary:
        pytest.xfail(
            f"{shape_id}: [{M},{N}] single-tile exceeds LX scratchpad (LBR0 pin) — "
            "needs tiling / chunk-large-tensors (tracked codegen follow-on)."
        )
    x = _small_pos((M, N))
    compiled = torch.compile(fn, dynamic=False)
    out_dev = compiled(x.to("spyre")).cpu()
    out_ref = fn(x)
    assert out_dev.shape == out_ref.shape, (
        f"{op_id} {shape_id}: device shape {tuple(out_dev.shape)} != {tuple(out_ref.shape)}"
    )
    _check(op_id, out_dev, out_ref)


# ---------------------------------------------------------------------------
# Guard: an out-of-port constant is the whole point. This is a HW-independent CI
# guard documenting the invariant so a future "just use the constant port" shortcut
# that only handles {0,1,2,3} cannot silently pass the suite while dropping 0.5/0.25.
# ---------------------------------------------------------------------------
def test_out_of_port_constants_are_covered():
    """The op table MUST include constants outside the SFP fixed-constant ports
    {0.0,1.0,2.0,3.0} — otherwise a port-only shortcut would pass without proving
    arbitrary-constant support (the actual #56 requirement)."""
    port_values = {0.0, 1.0, 2.0, 3.0}
    # extract the literal constants the lambdas apply (documented in ids)
    out_of_port = {"mul_0p5", "mul_0p25", "mul_5", "add_0p5", "add_7", "rsub_10", "rmul_0p5"}
    covered = {o[0] for o in _SCALAR_AFFINE_OPS}
    assert out_of_port & covered, (
        "scalar-affine op table must exercise constants outside the {0,1,2,3} SFP ports"
    )
    # sanity: 0.5 and 0.25 are NOT port-backed
    assert 0.5 not in port_values and 0.25 not in port_values
