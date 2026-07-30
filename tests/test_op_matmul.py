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


_ELEMS_PER_STICK = 64          # 128-byte stick / 16-bit DL16 element
_DL16_MAX_EXACT_INT = 1024     # 10 significand bits => ints exact to 2^10; 1025 is not


def _kat_kdense(mode, M, K, N):
    """DL16-exact known-answer with a DENSE activation (nonzero at every k).

    WHY: the "identity" operand has a large BLIND SUBSPACE. Because a[m,k] = 0 for
    every k > 0, ANY permutation of activation elements confined to k > 0 leaves
    every output BIT-IDENTICAL -- all K-1 non-zero-column k-columns are mutually
    interchangeable with zero observable effect. It can DETECT that something is
    wrong but cannot DISCRIMINATE between candidate activation-addressing fixes.
    Here a is dense, so a mis-addressed activation changes a value.

    WHY IT IS STILL EXACT AT LARGE K: b is ONE-HOT per column (b[j,j] = 1), so the
    product multiset for each output is one exact value plus K-1 exact zeros, hence
    every PARTIAL sum is either 0 or that value and no accumulation step rounds --
    in any order. This matters because on AIU 1.0 the PT FP accumulator IS DL16
    (no wider accumulator; 1p5 gets FP24), so a K-reduction rounds to 10 significand
    bits at EVERY step. That is also why "ones" needs K <= 1024 and this does not.

    The values are positional codes, so a residual names the element actually read:
      kdense_ktile (P1)  a[m,k] = 1 + (m % 16) + 16 * (k // 64)   k-tile in the HIGH
                         digit, so a one-tile error shows |delta| >= 16. Degenerate
                         at ks == 1 (single k-tile), hence gated to K > 64.
      kdense_lane  (P2)  a[m,k] = 1 + (m % 16) + 16 * (k % 64)    resolves the element
                         WITHIN a stick; max is 1024 for any shape.
      kdense_row   (P3)  a[m,k] = 1 + m + 128 * (k_t % 8)         resolves m EXACTLY,
                         so it sees ROW/CORE displacement that P1/P2 cannot.
    P1/P2 attain exactly 1024 (ZERO margin); P3's max depends on M and ks -- hence the
    asserts below rather than a fixed envelope.

    WHICH PATTERN SEES WHAT -- they are NOT interchangeable, and picking the wrong one
    silently proves nothing (verified offline on the R1/SP4K band, C=4 ks=4 rows=16):
                                      D1    P1    P2    P3
      contiguous-read (no ADDEARIMM)  64    48     0   336
      k-tiles 1<->3 exchanged          0    32     0   256
    P2 is BLIND to both because it is k-tile-invariant by construction. P1's row digit
    is (m % 16), which ALIASES with rows=16, so P1 sees these only via its k_t digit.
    P3 resolves m exactly and is the most sensitive to the `M - rows` inter-tile skip,
    which is a ROW-dimension law. Rule of thumb: THE PATTERN MUST RESOLVE THE DIMENSION
    THE DEFECT PERTURBS -- k-tile errors need P1, row/core errors need P3, intra-stick
    errors need P2.

    Canonical implementation and the full proof:
    spyre-knowledgebase codegen/scripts/matmul_kat_operands.py (kept in step with
    this by test_kdense_matches_the_canonical_kat_module).
    """
    # A one-hot selector needs a distinct k per output column.
    assert N <= K, (
        f"kdense requires N <= K (one-hot selector needs a distinct k per column); "
        f"got K={K}, N={N}. Large-N bands (N>K) cannot use this operand."
    )
    m = torch.arange(M, dtype=torch.int64).unsqueeze(1)
    k = torch.arange(K, dtype=torch.int64).unsqueeze(0)
    if mode == "kdense_ktile":
        assert K > _ELEMS_PER_STICK, (
            f"kdense_ktile is degenerate at ks == 1 (K={K}): k//64 is always 0, so "
            f"the pattern carries no k information. Use kdense_lane."
        )
        values = 1 + (m % 16) + 16 * (k // _ELEMS_PER_STICK)
    elif mode == "kdense_lane":
        values = 1 + (m % 16) + 16 * (k % _ELEMS_PER_STICK)
    elif mode == "kdense_row":
        # m resolved EXACTLY (not m % 16), so row/core displacement is visible.
        values = 1 + m + 128 * ((k // _ELEMS_PER_STICK) % 8)
    else:
        raise ValueError(f"unknown kdense mode: {mode!r}")

    # ZERO-MARGIN GUARDS. Refuse rather than emit a spurious diff: a value above
    # 1024 is not DL16-exact, so the resulting max_diff would look like a codegen
    # bug rather than an out-of-envelope operand.
    vmax = int(values.max().item())
    assert vmax <= _DL16_MAX_EXACT_INT, (
        f"{mode} at M={M}, K={K} reaches {vmax} > {_DL16_MAX_EXACT_INT}: not "
        f"DL16-exact (1025 is the first inexact integer), so every partial sum is "
        f"no longer provably exact."
    )
    assert int(values.min().item()) >= 1, "activation must be nonzero at every (m,k)"

    a = values.to(torch.float16)
    assert torch.equal(a, torch.round(a)), "activation must stay integer-valued"
    b = torch.zeros(K, N, dtype=torch.float16)
    cols = torch.arange(N, dtype=torch.int64)
    b[cols, cols] = 1.0
    # One-hot-per-column is THE precondition of the exactness argument, so check the
    # built tensor rather than trusting the indexing.
    assert torch.equal(b.sum(dim=0), torch.ones(N, dtype=torch.float16)), \
        "selector b must be one-hot per column"
    # Analytic reference: a pure gather of a, no arithmetic -> zero reference error.
    expected = a[:, :N].clone()
    return a, b, expected


def _kdense_modes(M, K, N):
    """Which kdense modes are valid for this shape.

    DERIVED rather than listed per case in _MATMUL_CASES, because unlike "identity"
    (exact for M <= 1024) and "ones" (exact for K <= 1024) the eligibility here is
    mechanical -- N <= K for the one-hot selector, K > 64 for a non-degenerate
    k-tile digit -- so deriving it keeps new cases covered automatically instead of
    relying on someone remembering to add the mode.
    """
    if N > K:
        return ()
    modes = ["kdense_lane"]
    if K > _ELEMS_PER_STICK:
        modes.append("kdense_ktile")
        # P3 resolves m exactly, so its ceiling depends on BOTH M and ks and it does
        # not fit every shape. Only offer it where it is provably DL16-exact.
        ks = K // _ELEMS_PER_STICK
        if 1 + (M - 1) + 128 * min(7, ks - 1) <= _DL16_MAX_EXACT_INT:
            modes.append("kdense_row")
    return tuple(modes)


def _known_answer(mode, M, K, N):
    """Return (a, b, expected) for a DL16-exact matmul known-answer.

    mode == "identity" (D1): identity-BLOCK weight B[k,k]=1 for k<min(K,N), 0 else.
                             With A's only nonzero column = column 0 (ramp A[i,0]=i+1),
                             C[i,j] = A[i,0]·B[0,j] -> C[:,0] = ramp, C[:,j>0] = 0. This
                             DISTINGUISHES per-K-tile / per-N-tile routing (a mis-routed
                             tile shows a nonzero outside column 0, or column 0 goes
                             wrong/zero) and works for ANY (K,N) — NOT just K==N (the SP1
                             K=N=64 case reduces to A @ I == A since A has only col 0
                             nonzero within the M<=... ramp). Exact for M<=1024 (ramp
                             value i+1 <= M), and it is the SAME D1 the HW probe uses.
    mode == "ones"     (D3): a = ones[M,K], b = ones[K,N]; expected[m,n] = K.
                             Exact only while K <= 1024 (DL16 integer ceiling).
    mode == "kdense_ktile" / "kdense_lane" (P1/P2): a is DENSE -- nonzero at EVERY
                             (m,k) -- so activation MIS-ADDRESSING is observable,
                             which "identity" structurally cannot see (see
                             _kat_kdense). b stays one-hot, which is what keeps it
                             DL16-exact at large K.
    """
    if mode.startswith("kdense_"):
        return _kat_kdense(mode, M, K, N)
    if mode == "identity":
        # ramp in column 0 only (generalizes the old K==N eye() form to any K,N).
        a = torch.zeros(M, K, dtype=torch.float16)
        for i in range(M):
            a[i, 0] = float(i + 1)
        b = torch.zeros(K, N, dtype=torch.float16)
        for k in range(min(K, N)):
            b[k, k] = 1.0
        expected = torch.zeros(M, N, dtype=torch.float16)
        expected[:, 0] = a[:, 0]
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
_SP3K = "sp3_clean_k"   # K>64 clean-K (K%64==0) within-core K-accumulation, C=1
_SP4 = "sp4_n_tiling"   # N>64 clean-N (N%64==0) output-column N-tiling (ns>1), multi-core
_SP4K = "sp4k_ktiling_ntiling"  # K>64 AND N>64 (ks>1 ∧ ns>1) composition, multi-core (R1)
_FLAT = "flat_single_pass"  # 127 < ks*ns <= ~192 single-pass large-N, C=1 (no ns time-loop)
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
    # --- SP3 rows>32 single-core (M>32, K=N=64). Phase 0 (#198) routes these to
    #     C=1 (spc=M<=200); the R3.1-R3.4 OUTER-band builders emit them, byte-matched
    #     offline + HW-verified where probed (M=193/194/195 D1/D3 max_diff=0). Exercised
    #     end-to-end here (eager + torch.compile). _SP1 tier (single-core). ---
    ("sp3_96x64x64",  96,  64, 64, 1, _SP1, ("identity", "ones")),
    ("sp3_160x64x64", 160, 64, 64, 1, _SP1, ("identity", "ones")),
    ("sp3_192x64x64", 192, 64, 64, 1, _SP1, ("identity", "ones")),
    # --- SP3 rows>32 MULTI-core: M=512 routes to C=16 (spc=32, the SP2 HW-verified
    #     ladder — spc=M=512>OUTER_SPC_MAX=200 excludes C=1). Exercises the SP2 emitter
    #     end-to-end for a rows>32 shape. _SP2 tier (multi-core). ---
    ("sp3_512x64x64_c16", 512, 64, 64, 16, _SP2, ("identity", "ones")),
    # --- Phase-1 multi-core rows>32 (2026-07-24): clean-OUTER C>=2 (spc in {64,96,192})
    #     and SPLIT-band C>=2 (spc in {128,256,512}) route via choose_cores Tier 3/4 and
    #     are HW-verified (D1/D3 max_diff=0, sp2 probe + band sweep). Declared C is the
    #     routed C. _SP2 tier. ---
    ("p1a_384x64x64_c2",  384,  64, 64, 2,  _SP2, ("identity", "ones")),   # spc=192 clean-OUTER
    ("p1a_288x64x64_c3",  288,  64, 64, 3,  _SP2, ("identity", "ones")),   # spc=96  clean-OUTER
    ("p1c_256x64x64_c2",  256,  64, 64, 2,  _SP2, ("identity", "ones")),   # spc=128 SPLIT-MC
    ("p1c_1024x64x64_c2", 1024, 64, 64, 2,  _SP2, ("identity", "ones")),   # spc=512 SPLIT-MC
    # --- SP3 CLEAN-K K-tiling (K>64, K%64==0), single-core (C=1). Routes via
    #     resolve_matmul_structure -> SEG_RELATIVE C=1 (within-core PE K-accumulation).
    #     HW-verified 2026-07-24 (D1 identity-block + D3 ones, max_diff=0) at ks<=6 for
    #     M=8 (spp=8) and M=32 (spp=32). ks=8 (K=512) fail-loud (TILE_SIZE overflow #311)
    #     -> NOT a case here. ones exact (K<=384<=1024). _SP3K tier. ---
    ("sp3k_8x128x64",   8,  128, 64, 1, _SP3K, ("identity", "ones")),   # ks=2
    ("sp3k_8x256x64",   8,  256, 64, 1, _SP3K, ("identity", "ones")),   # ks=4
    ("sp3k_8x384x64",   8,  384, 64, 1, _SP3K, ("identity", "ones")),   # ks=6
    ("sp3k_32x128x64", 32,  128, 64, 1, _SP3K, ("identity", "ones")),   # ks=2 spp=32
    ("sp3k_32x256x64", 32,  256, 64, 1, _SP3K, ("identity", "ones")),   # ks=4 spp=32
    # #SP3-MGEN (2026-07-26): clean-OUTER general-rows {12,16,20} — the M-gen widen
    # (LXLU restickify tail generalized from {8,32} to all clean-OUTER rows). Byte-match
    # verified M12/16/20 x ks2/4; xfail until the torch-path HW probe flips them green.
    ("sp3k_12x128x64", 12,  128, 64, 1, _SP3K, ("identity", "ones")),   # ks=2 spp=12
    ("sp3k_12x256x64", 12,  256, 64, 1, _SP3K, ("identity", "ones")),   # ks=4 spp=12
    ("sp3k_16x128x64", 16,  128, 64, 1, _SP3K, ("identity", "ones")),   # ks=2 spp=16
    ("sp3k_16x256x64", 16,  256, 64, 1, _SP3K, ("identity", "ones")),   # ks=4 spp=16
    ("sp3k_20x128x64", 20,  128, 64, 1, _SP3K, ("identity", "ones")),   # ks=2 spp=20
    ("sp3k_20x256x64", 20,  256, 64, 1, _SP3K, ("identity", "ones")),   # ks=4 spp=20
    # --- SP4 N-tiling (N>64, K=64 clean-N), multi-core. Routes via
    #     resolve_matmul_structure -> SEG_RELATIVE at the choose_cores-routed C
    #     (effC-clamped). The ns-FOLD PE-peel (pe_outer_imm*trip>=27) is now BUILT
    #     (d2ba51c) — unblocks the large-N caller class. K=64<=1024 so BOTH identity
    #     and ones are DL16-exact. C is the ROUTED count (not fixed here); shapes below
    #     exercise the peel band. xfail until the torch-path HW probe flips them green.
    #     _SP4 tier. ---
    ("sp4_128x64x2048", 128, 64, 2048, 8,  _SP4, ("identity", "ones")),  # routed C8  rows16 prod64 PEEL
    ("sp4_256x64x1024", 256, 64, 1024, 8,  _SP4, ("identity", "ones")),  # routed C8  rows32 prod64 PEEL
    ("sp4_256x64x2048", 256, 64, 2048, 16, _SP4, ("identity", "ones")),  # routed C16 rows16 prod64 PEEL
    ("sp4_512x64x1024", 512, 64, 1024, 16, _SP4, ("identity", "ones")),  # routed C16 rows32 prod64 PEEL
    ("sp4_128x64x512",  128, 64, 512,  16, _SP4, ("identity", "ones")),  # routed C16 rows8 prod8 loop (control)
    ("sp4_256x64x512",  256, 64, 512,  32, _SP4, ("identity", "ones")),  # routed C32 rows8 prod8 loop (control)
    # --- SP4K: K>64 AND N>64 composition (ks>1 ∧ ns>1), multi-core (R1). Routes via
    #     resolve_matmul_structure -> SEG_RELATIVE at the choose_cores-routed C. The
    #     multi-core K-tiling x N-tiling was CODEGEN-HW-verified 2026-07-27
    #     (r1_multicore_ktiling_ntiling_hw_probe: 64x256x256 @C4/@C6, D1+D3 = 4/4
    #     max_diff==0 via emit + launch_kernel_from_bytes). K,N=256<=1024 so identity
    #     and ones are DL16-exact (ones -> out=K=256). Fit bound ks*ns<=127 (Llama
    #     K=4096 unreachable single-kernel -> R2/R5). NOTE the routed C: at default
    #     SENCORES=32, 64x256x256 routes C16->rows4 (KLOOP, unbuilt) -> stays xfail
    #     until the rows=4 KLOOP rung (#13) OR a per-case SENCORES pin lands; the
    #     HW-verified path is C4 (SENCORES<=12). _SP4K tier. ---
    ("sp4k_64x256x256",  64, 256, 256, 4,  _SP4K, ("identity", "ones")),  # ks4 ns4, HW-verified @C4
    ("sp4k_64x512x512",  64, 512, 512, 4,  _SP4K, ("identity", "ones")),  # ks8 ns8 (ns==ks)
    ("sp4k_128x256x256", 128, 256, 256, 8,  _SP4K, ("identity", "ones")),  # ks4 ns4 rows16
    # --- FLAT single-pass large-N (127 < ks*ns <= ~192), C=1. The whole N extent is
    #     RESIDENT and swept in ONE pass — there is NO ns time-loop, so this is a
    #     structurally different emitter from SP4/SP4K, with its own derived PT drain
    #     (flat_pt_row_programs) and its own init-packet trailing pad (BODY 4-FLIT
    #     ALIGNMENT, body = total_flits-1 ≡ 0 mod 4).
    #     OFFLINE STATUS: rows 8 and 16 are FULL-BINARY RAW-IDENTICAL to the shipped
    #     deeptools goldens (codegen 7f4c1ca) — the ceiling of what can be proven
    #     offline. That is NOT correctness: the emulator stubs the compute units
    #     (sfp/pe/pt/l0 pass-through) and models no DMA/multicast/XLAT, so
    #     BusFence/deadlock is invisible to us. THIS TIER IS THE ONLY ORACLE.
    #     rows=32 (flat_32x256x2048) additionally exercises the UNCOLLAPSED LXLU
    #     activation ladder, which is intentionally NOT raw-byte-match-gated (deeptools'
    #     LDU-burst collapse is an encoding-only cost artifact we don't replicate);
    #     silicon max_diff=0 is its ONLY acceptance criterion. It currently fail-louds
    #     at serialization on flat_rows_ge32_hwkat_pending — run rows 8/16 FIRST, then
    #     clear that flag and run rows=32.
    #     K=256 <= 1024 so identity AND ones are both DL16-exact (ones -> out=K=256).
    #     xfail until the board returns max_diff==0. _FLAT tier. ---
    ("flat_8x256x2048",   8, 256, 2048, 1, _FLAT, ("identity", "ones")),  # rows8  ks4 ns32 RAW-matched
    ("flat_16x256x2048", 16, 256, 2048, 1, _FLAT, ("identity", "ones")),  # rows16 ks4 ns32 RAW-matched
    ("flat_32x256x2048", 32, 256, 2048, 1, _FLAT, ("identity", "ones")),  # rows32 uncollapsed ladder
    # --- Real LLM projection (Granite/Llama-class hidden 4096). K>64 clean-K IS now
    #     built (SP3 clean-K), but K=4096 = ks=64 (>=8) fail-loud on the TILE_SIZE
    #     overflow (#311), AND N=4096 needs SP4 (unbuilt). ones overflows DL16
    #     (K=4096>1024) so only identity is DL16-exact. _LLM tier. ---
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
_HW_VERIFIED_MATMUL: set[str] = {
    "sp1_2x64x64", "sp1_4x64x64",
    # SP3 rows>32 single-core (Phase 0 #198 routes M=96/160/192 -> C=1). HW-verified
    # 2026-07-23: eager + torch.compile, D1 identity + D3 ones, max_diff==0 on silicon.
    "sp3_96x64x64", "sp3_160x64x64", "sp3_192x64x64"}
_MULTICORE_HW_VERIFIED_MATMUL: set[str] = {
    "sp2_2x64x64_c2", "sp2_4x64x64_c2", "sp2_8x64x64_c4", "sp2_16x64x64_c8",
    # SP3 rows>32 multi-core (Phase 0 #198 routes M=512 -> C=16 ladder). HW-verified
    # 2026-07-23: eager + torch.compile, D1 identity + D3 ones, max_diff==0 on silicon.
    "sp3_512x64x64_c16"}
    # NOTE: the Phase-1 multi-core rows>32 cases (p1a_*, p1c_*) are HW-verified at the
    # BINARY-INJECTION level (sp2_matmul_hw_probe.py + band sweep, max_diff=0) but NOT
    # yet through the torch eager/compile path — they stay xfail here until a torch-level
    # HW run confirms max_diff==0, then their ids are added above (auto-flip).
# SP3 clean-K (K>64, K%64==0) single-core. HW-verified 2026-07-24 END-TO-END through
# the torch eager AND torch.compile paths (max_diff==0, D1 identity-block + D3 ones):
# bare torch.matmul K>64 now classifies "matmul" (the K-accumulate addf is no longer
# misread as a bias — ttir_analyzer._addf_feeds_loop_yield) and routes via
# resolve_matmul_structure -> SEG_RELATIVE C=1 (the _matmul_resolver clean-K envelope).
# ks<=6; ks=8 (K=512) stays fail-loud (TILE_SIZE overflow #311).
_SP3K_HW_VERIFIED_MATMUL: set[str] = {
    "sp3k_8x128x64", "sp3k_8x256x64", "sp3k_8x384x64",
    "sp3k_32x128x64", "sp3k_32x256x64",
    # #SP3-MGEN clean-OUTER general-rows — HW-verified 2026-07-26 (max_diff==0 on
    # silicon, D1 identity + D3 ones; codegen sweep ks2..8, torch path ks2/4).
    "sp3k_12x128x64", "sp3k_12x256x64",
    "sp3k_16x128x64", "sp3k_16x256x64",
    "sp3k_20x128x64", "sp3k_20x256x64"}

# SP4 N-tiling (N>64, K=64). The ns-FOLD PE-peel (pe_outer_imm*trip>=27) is BUILT +
# CODEGEN-HW-VERIFIED 2026-07-27 (large_n_peel_hw_probe: 8 shapes x D1/D3 = 16/16
# max_diff==0 at routed C, via emit_sp2_matmul + launch_kernel_from_bytes). BUT the
# HW-VERIFIED 2026-07-27: codegen large_n_peel_hw_probe 16/16 max_diff==0 at routed C,
# AND the SP5 frontend gate widened to dispatch N>64 to resolve_matmul_structure ->
# SEG_RELATIVE (ttir_to_schedule _matmul_resolver _clean_n). Real torch eager +
# torch.compile known-answer tests now.
_SP4_HW_VERIFIED_MATMUL: set[str] = {
    "sp4_128x64x2048", "sp4_256x64x1024", "sp4_256x64x2048",
    "sp4_512x64x1024", "sp4_128x64x512", "sp4_256x64x512"}

# SP4K K-tiling x N-tiling composition (ks>1 ∧ ns>1), multi-core (R1). CODEGEN-HW-
# verified 2026-07-27 (r1_multicore_ktiling_ntiling_hw_probe: 64x256x256 @C4/@C6,
# max_diff==0 via emit + launch_kernel_from_bytes) AND full-binary RAW byte-match. The
# torch eager/compile path routes ks>1∧ns>1 (resolve_matmul_structure now PROBES the
# emitter, no more "unwitnessed triple" raise). EMPTY until a torch-path HW run confirms
# max_diff==0 at the choose_cores-routed C: at default SENCORES=32 these route to
# C16->rows4 (KLOOP band, still unbuilt, #13) or unverified C16/C32 -> the torch-path
# known-answer isn't green yet. Flip an id in once its torch-path HW probe passes.
_SP4K_HW_VERIFIED_MATMUL: set[str] = set()

# FLAT single-pass large-N (ks*ns>127, C=1, no ns time-loop). Offline is EXHAUSTED:
# rows 8/16 are FULL-BINARY RAW-IDENTICAL to the shipped deeptools goldens (codegen
# 7f4c1ca) — both the derived PT drain and the derived init-packet body-alignment pad.
# EMPTY, and it must stay empty until the board speaks: raw-identity to a golden is
# strong evidence but NOT correctness, because the emulator stubs the compute units and
# models no DMA/multicast/XLAT, so a BusFence/deadlock would be invisible offline.
# CACHE HYGIENE IS LOAD-BEARING before running these: clear ~/.sentient_cache AND
# RESTART Python. The Triton in-memory cache survives torch._dynamo.reset() and
# TRITON_ALWAYS_COMPILE=1 only bypasses the on-disk cache — a stale binary can make a
# real fault look fixed, or a working fix look broken.
# Flip an id in ONLY after max_diff==0 with no BusFence/RbTimeOut on silicon.
# HW-VERIFIED 2026-07-28 on tmhoangt-spyre-dev-bob-quick (AIU 1.0 RCUDD1A): rows8/16
# eager AND compiled, max_diff==0, no BusFence/RbTimeOut. rows=32 (flat_32x256x2048)
# stays OUT — it is still router-refused (flat_rows_ge32_hwkat_pending; uncollapsed
# LXLU ladder), so its equivalence is unproven on silicon.
_FLAT_HW_VERIFIED_MATMUL: set[str] = {"flat_8x256x2048", "flat_16x256x2048"}


def _sp4_reason(case_id, M, K, N):
    return (
        f"{case_id} [{M}x{K}x{N}]: SP4 N-tiling (N>64) is BUILT — the ns-FOLD PE-peel "
        "(pe_outer_imm*trip>=27) landed (codegen d2ba51c) and byte-matches the crossing "
        "goldens. This id is in the table but NOT yet in _SP4_HW_VERIFIED_MATMUL — its "
        "torch eager/compile known-answer has not been confirmed max_diff==0 on silicon. "
        "Flip it in once the torch-path HW probe passes (D1 identity + D3 ones, K=64 so "
        "both DL16-exact). Real LLM-scale N (ns>=64 at large K) still needs the "
        "chunked-K x N-tiling composition (LX-capacity-bound)."
    )


def _sp4k_reason(case_id, M, K, N):
    return (
        f"{case_id} [{M}x{K}x{N}]: SP4K K-tiling x N-tiling (ks>1 AND ns>1) composition "
        "is BUILT and CODEGEN-HW-verified (r1_multicore_ktiling_ntiling_hw_probe: "
        "64x256x256 @C4/@C6 max_diff==0) + full-binary RAW byte-match. The torch path "
        "routes it (resolve_matmul_structure PROBES emit_sp2_matmul), but this id is NOT "
        "yet in _SP4K_HW_VERIFIED_MATMUL — at default SENCORES=32 it routes to a C whose "
        "torch-path known-answer is not confirmed max_diff==0 on silicon (64x256x256 -> "
        "C16->rows4 KLOOP, unbuilt #13; the HW-verified path is C4 at SENCORES<=12). Flip "
        "it in once a torch-path HW probe passes at the routed C (D1 identity + D3 ones, "
        "K,N<=1024 both DL16-exact). Llama-scale ks*ns>127 needs R2/R5."
    )


def _flat_reason(case_id, M, K, N):
    rows32 = case_id == "flat_32x256x2048"
    return (
        f"{case_id} [{M}x{K}x{N} @C1]: FLAT single-pass large-N (ks*ns>127) is BUILT — "
        "the whole N extent is resident and swept in ONE pass (no ns time-loop), with a "
        "DERIVED PT drain (flat_pt_row_programs: FILL carries the systolic skew, DRAIN = "
        "(rows//4)*(ns//2) granules) and a DERIVED init-packet trailing pad (BODY 4-FLIT "
        "ALIGNMENT: body = total_flits-1 is a multiple of 4). rows 8/16 are FULL-BINARY "
        "RAW-IDENTICAL to the shipped deeptools goldens (codegen 7f4c1ca) — the CEILING "
        "of offline evidence, and NOT correctness: the emulator stubs the compute units "
        "and models no DMA/multicast/XLAT, so BusFence/deadlock is invisible offline. "
        + ("This rows=32 case additionally exercises the UNCOLLAPSED LXLU activation "
           "ladder, which is intentionally NOT raw-byte-match-gated (deeptools' "
           "LDU-burst collapse is an encoding-only cost artifact we do not replicate; "
           "both forms address the same |A|=rows*ks sticks per trip). Silicon "
           "max_diff==0 is its ONLY acceptance criterion, and it currently fail-louds "
           "at serialization on flat_rows_ge32_hwkat_pending — run rows 8/16 first, then "
           "clear that flag. " if rows32 else "")
        + "Before running: clear ~/.sentient_cache AND restart Python (the Triton "
        "in-memory cache survives torch._dynamo.reset()). Flip this id into "
        "_FLAT_HW_VERIFIED_MATMUL once the board returns max_diff==0 with no "
        "BusFence/RbTimeOut (D1 identity + D3 ones; K=256<=1024 so both are DL16-exact)."
    )


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


def _sp3k_reason(case_id, M, K, N):
    return (
        f"{case_id} [{M}x{K}x{N}]: SP3 clean-K (K%64==0, K>64) within-core K-accumulation "
        f"is BUILT and HW-verified via BINARY INJECTION (max_diff==0, ks<=6). The torch "
        f"eager/compile path routes it (resolve_matmul_structure -> SEG_RELATIVE C=1), but "
        f"its torch-level HW known-answer has not yet been confirmed max_diff==0 on silicon. "
        f"Flip it into _SP3K_HW_VERIFIED_MATMUL once a torch-path HW probe passes (D1/D3)."
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
    elif tier == _SP3K:
        if case_id not in _SP3K_HW_VERIFIED_MATMUL:
            pytest.xfail(_sp3k_reason(case_id, M, K, N))
    elif tier == _SP4:
        if case_id not in _SP4_HW_VERIFIED_MATMUL:
            pytest.xfail(_sp4_reason(case_id, M, K, N))
    elif tier == _SP4K:
        if case_id not in _SP4K_HW_VERIFIED_MATMUL:
            pytest.xfail(_sp4k_reason(case_id, M, K, N))
    elif tier == _FLAT:
        if case_id not in _FLAT_HW_VERIFIED_MATMUL:
            pytest.xfail(_flat_reason(case_id, M, K, N))
    elif tier == _LLM:
        if case_id not in _HW_VERIFIED_MATMUL:
            pytest.xfail(_llm_reason(case_id, M, K, N))
    else:
        raise ValueError(f"unknown tier: {tier!r}")


def _check_known_answers(case_id, entry, M, K, N, ka_modes, run):
    """Run each DL16-exact known-answer for this shape via `run(a, b)` and assert
    max_diff == 0. Executed ONLY once the case is HW-verified (past the _xfail gate) —
    documents the intended assertion until then.

    The kdense modes are appended mechanically (see _kdense_modes): they are the only
    known-answers whose activation is dense in k, so they are the only ones that can
    observe an activation-addressing regression. On an already-passing shape they must
    give max_diff == 0 — that is the no-false-positive check for the instrument."""
    ka_modes = tuple(ka_modes) + _kdense_modes(M, K, N)
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
    assert _HW_VERIFIED_MATMUL == {
        "sp1_2x64x64", "sp1_4x64x64",
        # SP3 rows>32 single-core (Phase 0 #198, routed C=1) — HW-verified 2026-07-23,
        # max_diff==0 eager + torch.compile, D1 identity + D3 ones.
        "sp3_96x64x64", "sp3_160x64x64", "sp3_192x64x64"}, (
        "Single-core-tier verified set changed. Only the K=N=64 SP1-tier ids + the "
        "HW-verified SP3 rows>32 shapes (M=96/160/192, C=1) are verified. Confirm "
        "max_diff==0 on silicon before widening, then update this lock."
    )
    assert _MULTICORE_HW_VERIFIED_MATMUL == {
        "sp2_2x64x64_c2", "sp2_4x64x64_c2", "sp2_8x64x64_c4", "sp2_16x64x64_c8",
        # SP3 rows>32 multi-core (Phase 0 #198, M=512 routed C=16) — HW-verified
        # 2026-07-23, max_diff==0 eager + torch.compile, D1 identity + D3 ones.
        "sp3_512x64x64_c16"}, (
        "Multi-core-tier verified set changed. Only the K=N=64 SP2-tier ids (routed "
        "C=1/2/4/8) + the HW-verified SP3 M=512->C=16 shape are verified. Confirm "
        "max_diff==0 on silicon before widening, then update this lock."
    )
    assert _SP3K_HW_VERIFIED_MATMUL == {
        # SP3 clean-K (K>64, K%64==0) single-core — HW-verified 2026-07-24 END-TO-END
        # (eager + torch.compile, D1 identity-block + D3 ones, max_diff==0) after the
        # ttir classifier fix (K-accumulate no longer misread as bias) + the clean-K
        # dispatcher envelope. ks<=6 (K<=384); ks=8 (K=512) stays fail-loud (#311).
        "sp3k_8x128x64", "sp3k_8x256x64", "sp3k_8x384x64",
        "sp3k_32x128x64", "sp3k_32x256x64",
        # #SP3-MGEN clean-OUTER general-rows {12,16,20} — HW-verified 2026-07-26
        # (max_diff==0, D1+D3; codegen sweep ks2..8, torch path ks2/4).
        "sp3k_12x128x64", "sp3k_12x256x64",
        "sp3k_16x128x64", "sp3k_16x256x64",
        "sp3k_20x128x64", "sp3k_20x256x64"}, (
        "SP3 clean-K verified set changed. K in {128,256,384} at M in {8,32} + the "
        "#SP3-MGEN general-rows {12,16,20} at K in {128,256} (ks<=6, C=1) are torch-path "
        "HW-verified. ks>=8 (K>=512) is #311. "
        "Confirm max_diff==0 on silicon before widening, then update this lock."
    )


def test_multicore_cases_carry_the_164_blocker():
    """Every C>1 K=N=64 SP2 case must be tagged sp2_multi_core (so it xfails on BOTH
    SP5 routing AND #164), never mislabeled single-core (which would drop the HW-fault
    blocker). SP4 (N>64) is its own multi-core tier: it routes to the same seg-relative
    multi-core path but is HW-verified past the #164 fault (large_n_peel_hw_probe), so
    it carries the _SP4 blocker-reason, not _SP2 — exclude it here."""
    for case_id, M, K, N, C, tier, ka_modes in _MATMUL_CASES:
        if C > 1 and tier not in (_SP4, _SP4K):
            assert tier == _SP2, (
                f"{case_id}: C={C}>1 but tier={tier!r}; K=N=64 multi-core must be "
                f"{_SP2!r} so its xfail names #164 (the 0x7b1b silicon fault). "
                f"(N>64 multi-core is {_SP4!r}; K>64∧N>64 multi-core is {_SP4K!r} — "
                f"both HW-verified past #164, allowed.)"
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


# --- kdense known-answer: offline properties (no device needed) ----------------------

@pytest.mark.parametrize("mode", ("kdense_lane", "kdense_ktile", "kdense_row"))
@pytest.mark.parametrize("M,K,N", [(128, 4096, 2752), (64, 256, 256), (8, 128, 64)])
def test_kdense_operand_is_dense_and_dl16_exact(mode, M, K, N):
    """The two properties the operand exists for: DENSE in k, and DL16-exact.

    Exactness is checked the only way that proves it -- the analytic gather must equal
    a float64 matmul, so the reference is validated independently of the algebra.
    """
    a, b, expected = _known_answer(mode, M, K, N)
    assert a.shape == (M, K) and b.shape == (K, N) and expected.shape == (M, N)
    # DENSE: every (m,k) nonzero. This is exactly what "identity" lacks.
    assert int((a == 0).sum()) == 0, f"{mode}: activation must be nonzero at every (m,k)"
    # DL16-exact: integer-valued and inside the 10-significand-bit ceiling.
    assert torch.equal(a, torch.round(a))
    assert int(a.max().item()) <= _DL16_MAX_EXACT_INT
    # b one-hot per column -> every partial sum is 0 or a single exact value.
    assert torch.equal(b.sum(dim=0), torch.ones(N, dtype=torch.float16))
    # The reference is a gather; prove it equals a real matmul computed exactly.
    ref = (a.to(torch.float64) @ b.to(torch.float64)).to(torch.float16)
    assert torch.equal(ref, expected), f"{mode}: analytic gather != float64 matmul"


def test_kdense_sees_what_identity_is_blind_to():
    """The reason this mode exists, as an executable claim.

    Under "identity" a[m,k] = 0 for every k > 0, so exchanging two k-tiles with
    k >= 1 is the IDENTITY MAP ON THE DATA and the output is bit-identical -- the
    blind subspace. Under kdense_ktile the same exchange moves the k-tile digit, so
    the residual is exactly the digit weight, 16.
    """
    M, K, N = 128, 4096, 2752
    es = _ELEMS_PER_STICK

    def swap_k_tiles(t, t1, t2):
        s, lo1, lo2 = t.clone(), t1 * es, t2 * es
        s[:, lo1:lo1 + es] = t[:, lo2:lo2 + es]
        s[:, lo2:lo2 + es] = t[:, lo1:lo1 + es]
        return s

    a_id, _, exp_id = _known_answer("identity", M, K, N)
    a_kd, _, exp_kd = _known_answer("kdense_ktile", M, K, N)
    # Both source blocks are identically zero under identity, so the swap is a no-op.
    assert int((a_id[:, es:3 * es] != 0).sum()) == 0
    blind = (swap_k_tiles(a_id, 1, 2)[:, :N].float() - exp_id.float()).abs().max().item()
    seen = (swap_k_tiles(a_kd, 1, 2)[:, :N].float() - exp_kd.float()).abs().max().item()
    assert blind == 0.0, "identity must be BLIND to a k>0 k-tile permutation"
    assert seen == 16.0, f"kdense_ktile must see it as exactly the digit weight 16, got {seen}"
    # A swap TOUCHING k-tile 0 is NOT in the blind subspace -- so "both tiles >= 1"
    # above is load-bearing, not incidental.
    touches_zero = (swap_k_tiles(a_id, 0, 1)[:, :N].float()
                    - exp_id.float()).abs().max().item()
    assert touches_zero > 0.0


def test_kdense_matches_the_canonical_kat_module():
    """Keep this in step with the canonical builder instead of letting them drift.

    The canonical implementation and the full DL16 proof live in spyre-knowledgebase
    (codegen/scripts/matmul_kat_operands.py). It is a script, not an installed package,
    so import it by path and skip when the sibling checkout is absent.
    """
    import importlib.util
    from pathlib import Path
    here = Path(__file__).resolve()
    # Two layouts in use: the dev checkout (.../spyre-knowledgebase/codegen/...) and the
    # box, which rsyncs codegen/ straight into AIU/ (.../AIU/codegen/...).
    rel = ("codegen", "scripts", "matmul_kat_operands.py")
    cand = [p.joinpath("spyre-knowledgebase", *rel) for p in here.parents]
    cand += [p.joinpath(*rel) for p in here.parents]
    src = next((c for c in cand if c.is_file()), None)
    if src is None:
        pytest.skip("matmul_kat_operands.py not found in any known checkout layout")
    spec = importlib.util.spec_from_file_location("matmul_kat_operands", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    M, K, N = 128, 4096, 2752
    for mine, theirs in (("kdense_ktile", "k_tile"), ("kdense_lane", "lane"),
                         ("kdense_row", "row")):
        a1, b1, e1 = _known_answer(mine, M, K, N)
        a2, b2, e2 = mod.kat_operands(M, K, N, theirs)
        assert torch.equal(a1, a2), f"{mine}: activation diverged from canonical {theirs}"
        assert torch.equal(b1, b2), f"{mine}: selector diverged from canonical {theirs}"
        assert torch.equal(e1, e2), f"{mine}: reference diverged from canonical {theirs}"
