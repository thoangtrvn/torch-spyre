# PLAN — attention compiled example (examples/attention.py; the owed M6 cell)

Lane `attention-compiled-example`, torch-spyre base `new-runtime` @ `ebc11369`,
codegen base `streaming-c2-staging` @ `90a7d1ff4` (pinned worktree — the KB
main checkout is volatile mid-window).

## The owed item

The silicon e-cells (2026-11-23, claim 15:14:51Z) ran EAGER
`F.scaled_dot_product_attention`. `torch.compile` through
`spyre__sdpa_overrideable` has never been executed. This lane executes it.

## The off-box finding (probed before any code change)

`torch.compile(fn)` with `fn = F.scaled_dot_product_attention` does NOT
reach `spyre__sdpa_overrideable`:

- Dynamo puts `aten::scaled_dot_product_attention` in the graph; aot has no
  table entry for it, so the C++ composite executes under FakeTensor and
  emits the MATH form: `prims.convert_element_type` fp32 upcast of q/k/v,
  `bmm`, `softmax`, `where` (captured with `TORCH_LOGS=aot_graphs`).
- The seam's decomposition is registered on
  `aten._scaled_dot_product_fused_attention_overrideable` ONLY — the op the
  EAGER C++ backend selection dispatches to on the device. Under dynamo the
  top-level `aten::scaled_dot_product_attention` is never lowered to it.
- Inductor then compiles the math-fp32 kernel and the Sentient codegen
  refuses typed: `F32PointerElementGrainError
  SENTIENT-POINTER-ELEMENT-GRAIN-COMPUTE-ONLY-POINTEE` (f32 pointee, no
  32-bit-grain route on rcudd1a). Same torch on the box (2.13.0, verified
  by ssh query) — this is not a local-only artifact.

Eager divergence explained: on the device, C++ backend selection picks the
OVERRIDEABLE backend for spyre tensors (silicon-witnessed), whose dispatch
hits our PrivateUse1 kernel. Under FakeTensor tracing the same C++ selection
falls to math. So compiled ≠ eager at this seam today — a real gap, not a
harness artifact.

## The fix (root-cause, in the decomposition table)

Register a decomposition for the TOP-LEVEL op dynamo actually traces:
`aten::scaled_dot_product_attention` → call
`aten._scaled_dot_product_fused_attention_overrideable` and project output
[0]. This mirrors exactly what the eager C++ selection does on the device —
no new refusal, no new admission; the overrideable decomposition (and its
delegate, the predicate) owns everything downstream. Standard
`register_spyre_decomposition` (table + PrivateUse1), the same convention as
every other op in the table.

## The routing mechanism (fix-rounds R1+R2, executed evidence — receipts/)

The PrivateUse1 kernel registration is what makes COMPILE work, not the
table entry. The stages, as executed (R2 corrected the R1 wording — the
dynamo graph does NOT inline the kernel):

- dynamo traces the user function and puts the TOP-LEVEL
  `aten::scaled_dot_product_attention` in the graph (the captured dynamo
  graph is exactly that one op);
- inside compile_fx, aot's FakeTensor tracing dispatches the top-level
  op THROUGH the PrivateUse1 kernel, expanding it to `spyre.attention` —
  the aot forward graph is the single `torch.ops.spyre.attention.default`
  node (raw `__aot_graphs` lines in
  `receipts/b396_compiled_path_mechanism.log`; NOTE that receipt's own
  machine summary line "aot forward graph contains spyre.attention:
  False" is a probe-script bug — its logging handler captured nothing
  because the artifact records reach stderr from a compile subprocess;
  the node is visible in the log body);
- the dispatcher is never consulted at RUNTIME (instrumented at
  b396ebbb: zero OPWrapper dispatch calls, 15 launches).

A TABLE-ONLY variant (decomposition table entry, no PrivateUse1 kernel —
the R1 reviewers' preferred branch, preserving eager dispatch) was built
and REFUTED BY EXECUTION: without the kernel, the C++ composite expands
to the math form under FakeTensor and the table entry never applies, so
the Sentient codegen refuses `F32PointerElementGrainError`
(`receipts/table_only_refutation.log`).
The eager reroute is therefore ACCEPTED, not chosen: registering the
kernel reroutes EAGER `F.scaled_dot_product_attention` through the Spyre
seam (the C++ composite and its fp16→fp32 prelude are bypassed). The
destination is the same overrideable backend the composite selects on
the device, but the route differs from the pre-change e-cell path —
the box cell must RE-WITNESS eager F.sdpa (see Box cell below).

## Off-box execution law (the codegen-only stand-ins, all documented)

The local env is codegen-only (no runtime extension, no device bytes). The
off-box ladder executes the REAL op body with three stand-ins, each the
established executor-execution pattern (`test_attention_executor_execution.py`):

1. `att._launch` → the phase simulator (dispatches on the plan's own role
   vocabulary, computes each phase's semantics on host copies).
2. `att.require_runtime` → no-op (there is no runtime locally; on the box
   this is the real guard).
3. `torch.Tensor.to` cpu→spyre routes through `copy_` so the codegen-only
   shadow channel carries values (the real backend's H2D; the fake
   backend's `_to_copy` leaves tensors channel-less — a codegen-only
   artifact, pinned by probe).

Eager F.sdpa off-box additionally hits the C++ prelude's fp16→fp32 `_to_copy`
which the codegen-only `copy_` refuses (dtype conversion) — on the box the
real backend performs it. The off-box eager baseline therefore runs the same
function through the seam one level down (the overrideable op), which is the
exact form the e-cell's C++ selection reaches; disclosed in the receipts.

## The example

`examples/attention.py` per the directory convention: seeded inputs at the
witnessed envelope (rows=8, head_dim=64, kv_len=128 — the HOST_SERIAL path;
DL16), eager + `torch.compile(dynamic=False)` runs, grades vs the fp64 flash
reference (band 0.25, the e2 0.0294-class observed), the refusal surface
(rows=33 → typed [row-budget]). The example PRINTS its grades and the
refusal; the load-bearing ASSERTIONS (anti-fallback pins, tolerances,
typed-refusal classes) live in the test, not the example — a user copies
the file, the suite gates it. Its docstring discloses: compiled path
off-box-verified/silicon-unwitnessed, the eager reroute, and
inference-only.

## Tests

`torch_spyre/tests/test_attention_compiled_execution.py` (fix-round R1 —
the original frame-counter pin was mutation-proven VACUOUS: with the
user compile removed, all four original tests passed, because the
counted frame was the seam OPWrapper's OWN inner torch.compile during
eager dispatch):
- compiled EXECUTES through dynamo AT THE USER LEVEL:
  `torch.compiler.is_compiling()` recorded at the user function's entry
  must be True (False on a silent eager fallback — including one through
  the OPWrapper inner compile, which compiles the decomposition, never
  the user function), values vs the fp64 reference at the simulator.
- the compiled path routes through spyre::attention: the traced dynamo
  graph contains the top-level SDPA op, and the EXECUTED launch roles
  are the attention plan's own phase vocabulary with the launch count
  derived from the plan's step count (not a hand-counted constant).
- mutation-proven teeth (R2 set, on the final code — the *_v2 logs;
  the R1-era logs are superseded): user compile removed -> the two
  compile tests FAIL on the is_compiling pin
  (`receipts/mutationA_v2.log`); BACKEND NO-OP (the recording backend
  returns gm.forward, traced-then-eager through the seam's own inner
  compile) -> they FAIL on the user-graph-compiled pin
  (`receipts/mutationC_v2.log` — only the USER-level compile hands
  compile_fx a graph containing the top-level SDPA op; the OPWrapper
  inner compile's graph contains spyre.attention only); OPWrapper inner
  compile neutralized -> they still pass, expected and disclosed — the
  inner compile is off the user-level pins (`receipts/mutationB_v2.log`).
- eager F.sdpa reroutes through the seam (the R1-required committed
  eager test: the top-level PrivateUse1 wrapper records an EAGER
  dispatch, values correct).
- negative dropout_p refuses typed (the ATen range the C++ composite
  validated before dispatch; the compiled path bypasses the composite).
- out-of-envelope refuses typed under compile (the predicate's own
  UnsupportedAttentionError, chain-traversed — not a bare Exception).

Dynamo note (R1 correction): only `_build_plan_offline` is
`@_dynamo_offline`; the admission mapping (`sdpa_attention_mapping`) IS
traced — it executes under dynamo tracing inside the inlined kernel.
The zero-graph-breaks fact is measured (counters), not implied by any
dynamo-disabled claim.

## Box cell (USER-gated; only if off-box is green)

Sync checksum-mode (replaces the box's untracked attention.py drop), claim
per hw-probe-coordination, control-first (a1p3 bdcde582), then the compiled
example AND an eager F.sdpa re-witness (the eager route changed with the
kernel registration — see The routing mechanism), L-190 LAUNCH-STATE pin
with launched binary hashes.

## Not in scope

Device-resident staging (the D2H/H2D per-call cost is the DECLARED v1
form), B>1/H>1 box witnessing, the f32 device route.
