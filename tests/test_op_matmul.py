"""Living coverage CONTRACT for matmul (aten.mm / torch.matmul / ``a @ b``) on spyre.

STATUS (2026-07-18): SP5 frontend routing is LIVE and the K=N=64 envelope is
HW-VERIFIED via torch.compile AND eager.
================================================================================
A real ``a @ b`` in the seg-relative envelope (K<=64, N<=64) now routes end-to-end
to the HW-verified MATMUL_SEG_RELATIVE emitter: resolve_matmul_structure claims
matmul_strategy + num_cores, the _compute_tiling resolver registry dispatches it,
and SENCORES (default 32) is the choose_cores core-budget ceiling. All five
in-envelope cases pass their DL16-exact known-answers with max_diff==0 (D1 identity,
D3 ones) at the choose_cores-ROUTED C — M=2->C=1, M=4->C=2, M=8->C=4. Out-of-envelope
LLM shapes (K>64 SP3 / N>64 SP4) remain unbuilt and stay xfail.

The file's value as a LIVING CONTRACT (mirrors tests/test_op_pointwise.py and
tests/test_op_scalar_affine.py):
  (a) it DOCUMENTS the matmul coverage — the SP1 tile, the multi-core SP2 M-tiling
      envelope, and the real LLM projection shapes — as an executable table, and
  (b) it AUTO-FLIPS a case to a real known-answer test the moment its id is added to
      _HW_VERIFIED_MATMUL (SP1 tier) / _MULTICORE_HW_VERIFIED_MATMUL (SP2 tier) after
      a max_diff==0 HW pass — no test rewrite. The remaining xfail (LLM) flips when
      SP3/SP4 land + HW-verify.

BLOCKER HISTORY (both cleared for the K=N=64 envelope — see codegen
docs/op-milestones/matmul.md, the SP1→SP5 roadmap):
  1. SP5 FRONTEND ROUTING — CLEARED. Previously TilingConfig.matmul_strategy defaulted
     to SINGLE_TILE and was never assigned MATMUL_SEG_RELATIVE from a production
     compile, so ``a @ b`` hit the SINGLE_TILE guard in scheduler/engine.py
     (UnsupportedMatmulShapeError) and ``choose_cores`` had zero callers. SP5 wired
     resolve_matmul_structure + the _compute_tiling resolver registry + the SENCORES
     core-budget bridge, so an in-envelope ``a @ b`` now routes to the HW-verified
     seg-relative emitter at the choose_cores-routed C.
  2. #164 MULTI-CORE HW FAULT — RESOLVED. The first SP2 silicon probe FAULTED
     (RAS 0x7b1b, box wedged); root cause was a stale init_packet.py on the box, NOT
     the codegen. Multi-core (routed C=2/4) is now HW-verified (max_diff==0).
  (Real LLM shapes carry a THIRD blocker that is STILL OPEN: K>64 needs SP3 (within-core
   K-accumulation) and N>64 needs SP4 (output-column N-tiling), both unbuilt.)

WHEN DO CASES FLIP GREEN?
  SP1 / SP2 K=N=64 tile -> DONE (2026-07-18): in _HW_VERIFIED_MATMUL /
                           _MULTICORE_HW_VERIFIED_MATMUL after torch.compile + eager
                           HW known-answers passed max_diff==0 at the routed C.
  real LLM shapes       -> STILL require SP3/SP4 (K>64 / N>64 tiling) built + HW-verify;
                           add the id to _HW_VERIFIED_MATMUL then.

KNOWN ANSWERS are the D1/D3 references from the SP1/SP2 HW probe (DL16-exact):
  D1 (identity):  B = I  ->  C = A @ I == A         (exact for ANY K, values <= 1024)
  D3 (ones):      A=1, B=1, K sticks -> C == K       (exact only while K <= 1024)
AIU 1.0 is DL16 (1-6-9): integers > 1024 are not represented exactly, so operands
and results are kept <= 1024 and the check is max_diff == 0 (see the project DL16
note). 1p0 dtype is torch.float16 (DL16); never bf16 (that is 1p5).

NOTE ON C (num_cores): C is NOT a torch-level knob — a user writes ``a @ b`` and the
SP5 tiling path chooses cores internally via choose_cores(M,K,N,SENCORES-budget). The
declared C in the case rows below is DOCUMENTATION of the intended spc regime
(spc = M·ceil(N/64) / C); the ROUTED C (what actually runs) is choose_cores' pick at
the default budget 32: M=2->C=1, M=4->C=2, M=8->C=4. Where a row's declared C differs
from the routed C (e.g. sp2_2x64x64_c2 declares C=2 but M=2 routes to C=1), the routed
C is authoritative and is what the HW known-answer verified.

Cross-ref: codegen/docs/op-milestones/matmul.md (SP1 HW-verified 2026-07-16, commit
4755ab7; SP2–SP5 roadmap), matmul-1core-recipe / matmul-sp2-multicore-recipe memories.
"""
import math

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

_EPS = 64  # elements per stick at 16-bit (64 fp16 elements / 128-byte stick)


# --- entry points (mirror test_op_scalar_affine.py: EAGER + explicit torch.compile) --
# fn is the matmul under test; both entry points must ultimately reach the same codegen
# lowering once SP5 lands. Eager `a @ b` on a spyre tensor lowers through the per-op
# torch.compile wrap; explicit torch.compile(fn, dynamic=False) is the full Inductor
# graph-capture path a compiled model uses.
def _matmul(a, b):
    return torch.matmul(a, b)   # a[M,K] @ b[K,N] -> c[M,N]


# --- DL16-exact known-answer builders (D1 identity, D3 ones) ------------------------
def _small_pos_2d(M, K):
    """DL16-exact activation, values 1..8 (arange%8+1), so A @ I == A stays exact.

    int64 arange then cast (a float16 arange overflows to NaN at n>65504 — see
    test_op_pointwise._small_pos). Values <= 8 keep the identity passthrough and any
    later real accumulation comfortably DL16-exact.
    """
    n = M * K
    return ((torch.arange(n, dtype=torch.int64) % 8 + 1).to(torch.float16).reshape(M, K))


def _known_answer(mode, M, K, N):
    """Return (a, b, expected) for a DL16-exact matmul known-answer.

    mode == "identity" (D1): b = I[K,N] (requires K == N); expected = A (since N == K).
                             Exact for ANY K because only one product term is nonzero.
    mode == "ones"     (D3): a = ones[M,K], b = ones[K,N]; expected[m,n] = K.
                             Exact only while K <= 1024 (DL16 integer ceiling).
    """
    if mode == "identity":
        assert K == N, "identity known-answer requires a square K==N tile"
        a = _small_pos_2d(M, K)
        b = torch.eye(K, dtype=torch.float16)          # [K, N] with N == K
        expected = a.clone()                            # A @ I == A
        return a, b, expected
    if mode == "ones":
        assert K <= 1024, "ones known-answer overflows DL16 for K > 1024"
        a = torch.ones(M, K, dtype=torch.float16)
        b = torch.ones(K, N, dtype=torch.float16)
        expected = torch.full((M, N), float(K), dtype=torch.float16)
        return a, b, expected
    raise ValueError(f"unknown known-answer mode: {mode!r}")


# --- case table: (id, M, K, N, C, tier, ka_modes) -----------------------------------
# tier ∈ {sp1_single_core, sp2_multi_core, llm_real}. C is DOCUMENTATION ONLY (not a
# torch knob) — it records which core split / spc regime the SP2 emitter targets.
# ka_modes lists the DL16-EXACT known-answers valid for that shape (identity is exact
# for any K; ones only for K<=1024). spc = M·ceil(N/64) / C.
_SP1 = "sp1_single_core"
_SP2 = "sp2_multi_core"
_LLM = "llm_real"

_MATMUL_CASES = [
    # --- Single-core SP1 envelope (M<=4, K=N=64, C=1). HW-verified via BINARY
    #     INJECTION only; torch-level they still xfail on SP5 routing. ---
    ("sp1_2x64x64",      2,   64,   64,   1, _SP1, ("identity", "ones")),
    ("sp1_4x64x64",      4,   64,   64,   1, _SP1, ("identity", "ones")),
    # --- Multi-core SP2 envelope (C>1). Routed C via choose_cores: M=2->C=1,
    #     M=4->C=2, M=8->C=4, M=16->C=8 (all spc=2 except M=2->spc=2 at routed C=1).
    #     The SP2 emitter is spc-invariant per core; only core_mask + the C-1 PatchInit
    #     fold move with C, so C=8 rides the UNCHANGED emitter (byte-matched the C=8
    #     deeptools goldens offline; HW-verified max_diff=0 via BOTH binary-injection AND
    #     the torch path — eager + torch.compile, first 8-core matmul on silicon). ---
    ("sp2_2x64x64_c2",   2,   64,   64,   2, _SP2, ("identity", "ones")),   # spc=1
    ("sp2_4x64x64_c2",   4,   64,   64,   2, _SP2, ("identity", "ones")),   # spc=2
    ("sp2_8x64x64_c4",   8,   64,   64,   4, _SP2, ("identity", "ones")),   # spc=2
    ("sp2_16x64x64_c8",  16,  64,   64,   8, _SP2, ("identity", "ones")),   # spc=2, routes C=8
    # --- Real LLM projection (Granite/Llama-class hidden 4096). K>64 (SP3) and N>64
    #     (SP4) both unbuilt, on top of SP5 routing. ones would overflow DL16
    #     (K=4096>1024) so only the identity known-answer is DL16-exact here. ---
    ("llm_128x4096x4096", 128, 4096, 4096, 1, _LLM, ("identity",)),
]


# --- HW-verified sets: a case flips to a REAL known-answer test the moment its id is
#     added here (mirrors _SUPPORTED_OPS / _HW_VERIFIED_AFFINE_OPS discipline). --------
# SP5 frontend routing is LIVE (resolve_matmul_structure + resolver registry + SENCORES
# bridge): `a @ b` in the K=N=64 envelope routes to MATMUL_SEG_RELATIVE at the
# choose_cores-routed C. All five in-envelope cases are HW-VERIFIED (2026-07-18,
# max_diff==0, D1 identity + D3 ones, EAGER and torch.compile) at their ROUTED C:
#   sp1_2x64x64 (M=2)->C=1, sp1_4x64x64 (M=4)->C=2, sp2_2x64x64_c2 (M=2)->C=1,
#   sp2_4x64x64_c2 (M=4)->C=2, sp2_8x64x64_c4 (M=8)->C=4 (first C=4 silicon coverage).
# Tier (not routed C) picks the set: _SP1 ids -> _HW_VERIFIED_MATMUL, _SP2 ids ->
# _MULTICORE_HW_VERIFIED_MATMUL (so sp1_4x64x64 lives here despite routing to C=2).
_HW_VERIFIED_MATMUL: set[str] = {"sp1_2x64x64", "sp1_4x64x64"}
_MULTICORE_HW_VERIFIED_MATMUL: set[str] = {
    "sp2_2x64x64_c2", "sp2_4x64x64_c2", "sp2_8x64x64_c4", "sp2_16x64x64_c8"}


def _sp5_reason(case_id, M, K, N):
    return (
        f"{case_id} [{M}x{K}x{N}]: SP5 frontend routing is LIVE (resolve_matmul_structure "
        "+ resolver registry route `a @ b` in the K<=64,N<=64 envelope to "
        "MATMUL_SEG_RELATIVE), and the K=N=64 shapes are HW-verified. This SP1-tier id "
        "is in the table but NOT yet in _HW_VERIFIED_MATMUL — its torch.compile "
        "known-answer has not been confirmed max_diff==0 on silicon. Flip it in once a "
        "HW probe passes (max_diff==0, D1 identity + D3 ones, eager + compiled)."
    )


def _multicore_164_reason(case_id, M, K, N, C):
    return (
        f"{case_id} [{M}x{K}x{N}, routed C via choose_cores]: SP5 routing is LIVE and the "
        f"SP2 multi-core seg-relative emitter is HW-VERIFIED (K=N=64 routed C=1/2/4, "
        f"max_diff==0; #164 the 0x7b1b stale-file fault is RESOLVED). This SP2-tier id is "
        f"in the table but NOT yet in _MULTICORE_HW_VERIFIED_MATMUL — its torch.compile "
        f"known-answer has not been confirmed max_diff==0 at the choose_cores-routed C. "
        f"Flip it in once a HW probe passes."
    )


def _llm_reason(case_id, M, K, N):
    return (
        f"{case_id} [{M}x{K}x{N}]: BLOCKED on SP5 routing AND on SP3/SP4. Real LLM shapes "
        f"need SP3 (K>64 within-core K-accumulation) and SP4 (N>64 output-column tiling), "
        f"both unbuilt — the SINGLE_TILE guard raises UnsupportedMatmulShapeError for "
        f"tile_k<K or tile_n<N. Flip in only after SP5 + SP3 + SP4 land + HW-verify."
    )


def _xfail(case_id, M, K, N, C, tier):
    """Raise the tier-accurate xfail (all cases xfail today). Never returns."""
    if tier == _SP1:
        if case_id not in _HW_VERIFIED_MATMUL:
            pytest.xfail(_sp5_reason(case_id, M, K, N))
    elif tier == _SP2:
        if case_id not in _MULTICORE_HW_VERIFIED_MATMUL:
            pytest.xfail(_multicore_164_reason(case_id, M, K, N, C))
    elif tier == _LLM:
        if case_id not in _HW_VERIFIED_MATMUL:
            pytest.xfail(_llm_reason(case_id, M, K, N))
    else:
        raise ValueError(f"unknown tier: {tier!r}")


def _check_known_answers(case_id, entry, M, K, N, ka_modes, run):
    """Run each DL16-exact known-answer for this shape via `run(a, b)` and assert
    max_diff == 0. Executed ONLY once the case is HW-verified (past the _xfail gate) —
    documents the intended assertion until then."""
    for mode in ka_modes:
        a, b, expected = _known_answer(mode, M, K, N)
        out_dev = run(a.to("spyre"), b.to("spyre")).cpu()
        assert out_dev.shape == expected.shape, (
            f"{case_id}/{entry}/{mode}: device shape {tuple(out_dev.shape)} != "
            f"{tuple(expected.shape)}"
        )
        md = (out_dev.float() - expected.float()).abs().max().item()
        assert md == 0.0, f"{case_id}/{entry}/{mode}: max_diff={md} (expected 0.0, DL16-exact)"


# ---------------------------------------------------------------------------
# Entry point 1: EAGER (bare `a @ b`). Lowers via the per-op torch.compile wrap.
# ---------------------------------------------------------------------------
@requires_hw
@pytest.mark.parametrize(
    "case_id,M,K,N,C,tier,ka_modes", _MATMUL_CASES, ids=[c[0] for c in _MATMUL_CASES]
)
def test_matmul_eager(case_id, M, K, N, C, tier, ka_modes):
    """`a @ b` via the bare eager path == CPU, DL16-exact (D1 identity, D3 ones).

    ALL cases xfail today — matmul torch routing is unbuilt (SP5), and C>1 additionally
    faults on silicon (#164). See the module docstring / matmul.md for the flip criteria.
    """
    _xfail(case_id, M, K, N, C, tier)
    _check_known_answers(case_id, "eager", M, K, N, ka_modes, _matmul)


# ---------------------------------------------------------------------------
# Entry point 2: explicit torch.compile (full Inductor graph capture + lowering).
# ---------------------------------------------------------------------------
@requires_hw
@pytest.mark.parametrize(
    "case_id,M,K,N,C,tier,ka_modes", _MATMUL_CASES, ids=[c[0] for c in _MATMUL_CASES]
)
def test_matmul_compiled(case_id, M, K, N, C, tier, ka_modes):
    """`a @ b` via explicit torch.compile(fn, dynamic=False) == CPU, DL16-exact.

    ALL cases xfail today (SP5 routing unbuilt; C>1 also #164). Auto-flips when the
    matching HW-verified set is populated.
    """
    _xfail(case_id, M, K, N, C, tier)
    compiled = torch.compile(_matmul, dynamic=False)
    _check_known_answers(case_id, "compiled", M, K, N, ka_modes, compiled)


# ---------------------------------------------------------------------------
# HW-independent guards (run everywhere, incl. CI without a board) — document the
# contract's structural invariants so a future shortcut can't silently subvert it.
# ---------------------------------------------------------------------------
def test_matmul_verified_membership_locked():
    """Contract lock: SP5 routing has LANDED and the K=N=64 envelope is HW-verified
    (2026-07-18, max_diff==0 eager + torch.compile at the choose_cores-routed C). This
    guard pins EXACTLY which ids are verified so a future edit can't silently widen the
    verified set past what has actually passed on silicon (e.g. add an SP3/SP4 shape).
    Every _MATMUL_CASES id not listed here MUST still be xfail. When a new SP builds and
    a shape HW-verifies, add its id to the matching set AND here in the same change."""
    assert _HW_VERIFIED_MATMUL == {"sp1_2x64x64", "sp1_4x64x64"}, (
        "Single-core-tier verified set changed. Only the K=N=64 SP1-tier ids are "
        "HW-verified via torch.compile (SP5). Confirm max_diff==0 on silicon before "
        "widening, then update this lock."
    )
    assert _MULTICORE_HW_VERIFIED_MATMUL == {
        "sp2_2x64x64_c2", "sp2_4x64x64_c2", "sp2_8x64x64_c4", "sp2_16x64x64_c8"}, (
        "Multi-core-tier verified set changed. Only the K=N=64 SP2-tier ids (routed "
        "C=1/2/4/8) are HW-verified. Confirm max_diff==0 on silicon before widening, then "
        "update this lock."
    )


def test_multicore_cases_carry_the_164_blocker():
    """Every C>1 case must be tagged sp2_multi_core (so it xfails on BOTH SP5 routing
    AND #164), never mislabeled single-core (which would drop the HW-fault blocker)."""
    for case_id, M, K, N, C, tier, ka_modes in _MATMUL_CASES:
        if C > 1:
            assert tier == _SP2, (
                f"{case_id}: C={C}>1 but tier={tier!r}; multi-core must be {_SP2!r} so "
                "its xfail names #164 (the 0x7b1b silicon fault), not just SP5 routing."
            )


def test_matmul_shape_model():
    """HW-independent: the ks/ns tiling + SP2 divisor-clamp spc the codegen path
    assumes. ks=ceil(K/64), ns=ceil(N/64); spc = M·ns / C (C must divide M·ns). Guards
    the shape contract in CI without a board (mirrors test_pointwise_shape_model)."""
    for case_id, M, K, N, C, tier, ka_modes in _MATMUL_CASES:
        ks = math.ceil(K / _EPS)
        ns = math.ceil(N / _EPS)
        output_sticks = M * ns
        assert output_sticks % C == 0, (
            f"{case_id}: C={C} must divide output_sticks=M·ns={output_sticks} "
            "(SP2 divisor-clamp)."
        )
        spc = output_sticks // C
        assert spc >= 1 and ks >= 1 and ns >= 1
    # SP1 envelope sanity: 64x64 tile is exactly one stick each way.
    assert math.ceil(64 / _EPS) == 1
    # LLM projection: 4096 hidden = 64 sticks each way (needs SP3 K-loop + SP4 N-tile).
    assert math.ceil(4096 / _EPS) == 64


# Routed core count: choose_cores picks C from M at the default SENCORES budget (32).
# The test-case C column DOCUMENTS the intended spc regime; the ROUTED C is what runs.
_ROUTED_CORES = {2: 1, 4: 2, 8: 4, 16: 8}  # M -> choose_cores(M,64,64,32)


def test_matmul_routed_cores_contract():
    """SP5 routes num_cores via choose_cores (budget=SENCORES=32 default), NOT the
    declared C column. Documents/locks the M->C mapping the HW probe verifies."""
    from sentient_codegen.scheduler.matmul_tiling_policy import choose_cores
    for M, expected_c in _ROUTED_CORES.items():
        assert choose_cores(M, 64, 64, 32) == expected_c, (
            f"M={M}: choose_cores routed C={choose_cores(M,64,64,32)}, expected {expected_c}")
