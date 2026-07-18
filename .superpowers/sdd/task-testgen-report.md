# Test-Gen Report — pointwise/reduction/scalar-affine/embedding coverage expansion

**Repo:** `torch-spyre-thoangtrvn` (branch `new-runtime`, not switched)
**Scope:** TESTS ONLY. No codegen / torch_spyre source touched. codegen/tests/ untouched
(owned by a parallel agent). HW box NOT touched.

## New case counts (net, offline `--collect-only`)

| file | baseline | after | NEW |
|---|---|---|---|
| test_op_pointwise.py | 152 | 348 | **+196** |
| test_op_scalar_affine.py | 81 | 178 | **+97** |
| test_op_reduction.py | 48 | 75 | **+27** |
| test_op_embedding.py | 34 | 58 | **+24** |
| test_op_matmul.py | 15 | 15 | 0 (already a complete living contract; all cases xfail on SP5 routing — more same-reason xfails = noise, not unique coverage) |
| **TOTAL** | | | **+344** |

Collect-clean: all 4 touched files collect with 0 errors. Full non-HW run:
**8 passed, 651 skipped** (device tests skip without a board), 0 collection errors.
All 5 files py_compile-clean.

## Axes covered (the matrix)

### test_op_pointwise.py (+196)
- **Compiled entry-point counterparts** for the standard unary + binary sweeps
  (`test_pointwise_unary_compiled`, `test_pointwise_binary_compiled`) — the existing
  sweeps were EAGER-ONLY; now every op × shape runs via `torch.compile(fn, dynamic=False)`
  too. Same gating as eager siblings (expected-pass for the 5 HW-verified ops at verified
  shapes; xfail for maximum/transcendentals + boundary).
- **Fused single-input linear chains** (`test_pointwise_fused_single_input_chain`,
  eager+compiled): `2*x+3`, `5*x-2`, `(x-5)*2+1`, `relu(x-5)` (fma-family + fminmax-in-chain)
  and classifier-accepted-but-unprobed folds (`x/2`, `(3-x)*2`, `neg(x)+3`).
- **Multi-tensor fused DAGs** (`test_pointwise_fused_multi_tensor`, eager+compiled):
  `(a*b)+c`, `a+b+c`, `(a-b)*c`, `a*b+7`.
- **Ragged × M>1** (`test_pointwise_{unary,binary}_ragged_m_gt1`, eager+compiled): NEW
  shapes 5×130, 33×100, 2×65 for neg/relu (unary) + add/mul/sub (binary, distinct-per-stick).
- **Flipped 4 existing ragged M>1 binary cases** in `test_pointwise_binary_ragged_n`
  (sub_stick_4x40, ragged_4x100, ragged_3x200, ragged_7x91) from expected-pass → xfail-strict
  (#167), per documented this-session HW evidence (see conflict resolution below). M==1 rows
  kept GREEN. Unary ragged test left untouched (no unary HW evidence).
- **Corrected the false 294-301 comment**: the "ragged works today, tail lands in row
  padding" claim is replaced with the HW-proven physical>logical stick-drop mechanism.
- 3 new HW-independent guard tests (shape-model + fused-envelope-honesty).

### test_op_scalar_affine.py (+97)
- **Ragged-N block** (`test_scalar_affine_ragged_n`, eager+compiled, all 8 subforms):
  M==1 ragged (1×100, 1×40, 1×5) expected-pass for verified subforms; M>1 ragged
  (4×100, 3×130, 2×65) xfail-strict on #166 (op-agnostic tail-drop, scalar-affine rides the
  shared unary datapath → same 1D-flatten drop). Closes the ragged gap (aligned-only before).
- 1 HW-independent shape-model guard.

### test_op_reduction.py (+27)
- **Entry-point axis** (`test_reduction_dim0_entry_points`, aten/eager/compiled) at real-LLM
  dim=0 shapes (1×4096 decode, 8×768, 8×4096) — the file was bare-op-only; pins the
  aten/eager/compiled convergence the docstring asserts but never tested. sum/max/mean
  expected-pass (dim=0 HW-verified); op-specific unverified reasons xfail as elsewhere.

### test_op_embedding.py (+24)
- **Token-count axis** (`test_embedding_multichunk_token_sweep`, 3 entry points × tokens
  {1,8,16,32} × C=2 d_models 4096/3584) — regression-lock for the resolved 32-token L3SU
  register-index-wrap bug (was max_diff=511; now HW-verified 0). The main sweep varies
  d_model but pins n_tokens, so the wrap-boundary token axis was uncovered.

## Expected-pass vs xfail taxonomy

**Expected-pass** (only HW-verified per the milestone docs):
- pointwise add/mul/sub/neg/relu at aligned shapes, eager AND compiled (pointwise.md).
- fused single-input chains `2*x+3`/`5*x-2`/`(x-5)*2+1`/`relu(x-5)` at the FOUR silicon-
  verified shapes [1,64]/[128,768]/[128,4096]/[512,4096] (fused-elementwise.md 2026-07-16).
- scalar-affine all 8 subforms at aligned + M==1 ragged (scalar-affine.md multi-stick verified).
- reduction sum/max/mean dim=0 at aligned real-LLM shapes, all 3 entry points (#193 fixed).
- embedding C=1 and C>1 across the token sweep (embedding.md multi-chunk HW-verified).

**xfail-strict** (predicted/proven-failing, auto-flips XPASS→fail when a REAL fix HW-verifies):
- `#167` ragged M>1 (binary add/mul/sub): HW-MEASURED add=11/mul=28/sub=7 (add this session,
  mul/sub earlier probes). Bug is in the SHARED .to(spyre) stick-write layer (see refutation).
- `#166` ragged M>1 (unary neg/relu, scalar-affine): op-agnostic tail-drop, PREDICTED (NOT yet
  HW-measured for these ops), should ride the same shared-layer fix → auto-flip with #167.
- fused single-input chain out-of-envelope: ragged M>1 → shared-layer #167 tail-drop (predicted,
  fused not separately measured); decode-wide/other → "not on the 2026-07-16 HW probe" reason.

NOTE: NO green flip is pending. "Fix A" (Inductor choices.py tiling hook) was HW-REFUTED
2026-07-17 (see refutation below); the xfail-strict markers are the correct honest terminal
state until a real shared-layer fix lands + HW-verifies.

**xfail plain (non-strict)** — fail-loud-by-design, tolerates a future correct result:
- `#99` multi-tensor fused DAGs (>1 tt.load): generic multi-input segment-packing NOT built;
  classifier rejects → pointwise_unsupported. Distinct from #167; not a stick-count bug.

**xfail (pre-existing reasons, unchanged):** maximum (no binary), transcendentals (FEST),
LX-capacity boundary (2048×4096 LBR pin), reduction Regime-B / dim=-1 / ragged (#180/#183).

## Fused-ops support status (grounded in codegen source, supersedes 4-day-old memory)

Verified against `backend/ttir_analyzer.py::_match_fused_chain` (HEAD) + fused-elementwise.md
(dated 2026-07-16 — the memory `generic-fused-elementwise-gap` is stale/overturned):
- **Single-input linear chains ARE lowered + HW-verified single-core** (`pointwise_fused`,
  task #85): one `tt.load`, straight chain of scalar-const arith → one SFP chain latched
  through one scratch LRF. Folds: mul/add/sub_l/sub_r/neg/max/min/div-by-const. `relu(x-c)`
  is now SUPPORTED (contradicts the stale pointwise.md §9 "relu(x-4) NOT handled" caveat).
- **Multi-tensor DAGs FAIL LOUD by design**: `_match_fused_chain` restriction 1 requires
  exactly ONE `tt.load`; `(a*b)+c` etc → pointwise_unsupported. Generic multi-input DAG→SFP
  lowering (#99) NOT built.
- Pending (→ xfail): multi-CORE fused, >2-node chains (fused-elementwise.md §6).

## Conflict resolution (recorded for honesty)

The task premise ("ragged M>1 currently FAIL, #167") CONFLICTED with committed-green ragged
tests + pointwise.md §13 (commit 5f5a6e0/be22754 assert max_diff=0 at the same M>1 shapes).
I refused to flip committed green on an unverifiable claim, then the coordinator supplied
DOCUMENTED this-session HW evidence: (1) probe at the exact committed operands gave
add=11/mul=28/sub=7 = the operand maxima (failing element reads ~0 where correct value was
maximal); (2) reproduced BYTE-IDENTICAL under codegen@7718bdc (pre-SP1/SP2) → PRE-EXISTING,
so be22754's green was a periodicity-masked false-green (same class as pointwise.md §7/§11/§12);
(3) offline-confirmed 8/8 mechanism: physical row-pad sticks (M·ceil(N/64)) > logical
(ceil(M·N/64)) when N%64≠0 ∧ M>1 → trailing sticks unwritten → read 0. That cleared the bar to
flip the 4 M>1 binary cases + correct the comment. M==1 stays green (logical==physical, no drop).

## Concern

The one status I could not fully self-adjudicate offline: whether unary neg/relu (and
scalar-affine) ragged M>1 drop IDENTICALLY to binary. The drop mechanism is documented
op-AGNOSTIC (choices.py 1D-flatten physical>logical, independent of SFP compute) and
offline-confirmed 8/8, but only add/mul/sub were HW-MEASURED. I marked unary/affine M>1 as
xfail-strict #166 (predicted, not-yet-measured) — the honest never-green default; if the
imminent HW sweep shows they were actually green, the strict marks XPASS→fail and force an
evidence-based un-mark. This is the intended self-correcting transition.

### Fix A refutation (2026-07-17 HW probe) — recorded for the durable record
After I committed the xfail-strict version, the coordinator's HW probe (choices.py "Fix A"
synced to the box, confirmed present, fresh process + cache clear per shape) measured
add-ragged_3x200 = **max_diff=11 on BOTH eager and compiled, BYTE-IDENTICAL to pre-fix**.
Fix A had ZERO effect. Because EAGER never routes through Inductor/choices.py yet is broken
identically, the controlling stick-count lives in the layer SHARED by both entries — the
`.to("spyre")` device-layout / kernel stick-write — NOT the Inductor tiling hook Fix A edited.
The coordinator's earlier "eager recovers (3,200)" was a false offline verification (helper run
in isolation, not real dispatch). **Impact on this test work: NONE except phrasing** — the
tests genuinely fail on HW, so xfail-strict is exactly right and STANDS; there is no follow-up
green flip. #167 is reopened for a real fix at the shared layer (a separate investigation owns
that). Reason strings were softened from "auto-flips when Fix A HW-verifies" → "auto-flips when
a REAL fix HW-verifies (Fix A refuted, shared .to(spyre) stick-write layer, #167 reopened)".
The wrong-element fingerprint (3×200: cols 192-199 = the 200%64=8 partial tail of rows 1,2 only,
row 0 correct) is the strongest in-file evidence of the physical-row-pad-unwritten mechanism.

## Commit

See git log — commit SHA reported separately. `git status` confirmed only tests/test_op_*.py
modified (untracked PLAN_*/build.sh/senlib/ etc. are pre-existing, not staged).
