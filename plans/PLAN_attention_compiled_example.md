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
(rows=33 → typed [row-budget]), graph-break count printed.

## Tests

`torch_spyre/tests/test_attention_compiled_execution.py`:
- compiled EXECUTES through dynamo (frames compiled > 0 — fails on a silent
  eager fallback), values vs the fp64 reference at the simulator.
- the compiled GRAPH routes through spyre::attention (no math-fp32 kernel).
- out-of-envelope refuses typed under compile.

## Box cell (USER-gated; only if off-box is green)

Sync checksum-mode (replaces the box's untracked attention.py drop), claim
per hw-probe-coordination, control-first (a1p3 bdcde582), then eager e2-form
- the compiled example, L-190 LAUNCH-STATE pin with launched binary hashes.

## Not in scope

Device-resident staging (the D2H/H2D per-call cost is the DECLARED v1
form), B>1/H>1 box witnessing, the f32 device route.
