# 2-D all_reduce layout-safe fix + reduce_scatter/subgroup un-park — Design

**Status:** Approved design (brainstorming complete); ready for implementation plan.
**Date:** 2026-08-05
**Branch:** `phase-1-overlap-communicate-compute` (same branch as the closed async Phase-1 work, per prior user decision to land both together).

## Goal

Fix the confirmed, HW-pinned correctness bug where a 2-D `[tokens, hidden]`
`all_reduce(SUM)` on the IBM Spyre c10d backend (`SpyreCCLBackend`) silently
under-reduces at ≥4 ranks (only `numel/ROWS` elements summed), by replacing the
linear-byte-slicing Ring reduce-scatter with a **layout-safe pairwise
reduce_scatter + all_gather** that never slices a single tiled buffer by device
byte offset. Un-park the coupled `reduce_scatter`/`exchange_uneven`/subgroup
work-in-progress that already implements the required primitive.

## Background: the confirmed bug (do not re-litigate)

HW-pinned 2026-08-04/05 via `diag_allreduce_split_coverage.sh` (v3, plain
torchrun, per-rank logs):

- 2-D `[8,4096]` fp16 at 4 ranks: `reduced=4096 / 32768` (exactly `1/ROWS`),
  `first_unreduced_idx=512`, on every rank.
- 1-D `[32768]` (same numel, same run): `reduced=32768 / 32768`, FULL.
- Split geometry is **correct** (`base_size=8192 × bases=4 = 32768`); all ranks
  emit `COMPUTE_ops=3`; SEND/RECV/COMPUTE/CONCAT tile the full 65536 bytes with
  no gaps (textbook ring). Add-kernel **exonerated** (1-D reduces fully).

**Root cause:** `allreduce` flattens the tensor to logical 1-D `{numel}`
(`context.cpp:729-730`) and hands LIBCOLL the raw device address; the Ring
reduce-scatter slices/reduces by **linear device byte offset**
(`envelope.cpp` `partByteOffset` / `CalcAddress`). Correct for a genuinely
contiguous 1-D buffer; **wrong** for a ≥2-D spyre tensor, which is stored in a
4-D **tiled/stickified** `SpyreTensorLayout` (`device_size` e.g. `[4,2,*,64]`,
128-byte stick innermost, dim-0 interleaved — `spyre_mem.cpp:123-135`). A linear
16384-byte chunk spans partial sticks of multiple logical rows, so the reduction
runs on the right *physical bytes* in the wrong *logical arrangement*; only the
`1/ROWS` stick-aligned fraction survives de-stickification. Same hazard class as
the HW-confirmed `alltoall_base` slicing bug.

**Why it matters (vLLM):** this is the TP activation `all_reduce(SUM)` on
`[num_tokens, hidden]` — fired twice per transformer layer, every forward pass,
the hottest TP collective. TP≥4 silently produces wrong logits with no crash;
TP=2 masks it. Production correctness blocker for any TP≥4 deployment.

## Why not the alternatives (settled during brainstorming)

A read-only code investigation (opus, 13 tool uses) established with file:line
evidence:

- **Approach A (copy-free logical-contiguous view) is impossible.** Every
  transport address type — `BufferDesc`, `CompositeAddress`, `Chunk`,
  `LogicalAddress` — is a flat `(region, offset, bytes)` with zero stride/tile
  capacity (`collective.cpp:462-553`, `alloc_address.hpp:20-96`,
  `spyre_comms_tensor.hpp:45-64`). SEND/RECV/COMPUTE cannot address logical
  elements of a tiled buffer (A1). ≥2-D spyre tensors are unconditionally tiled
  and no un-tile helper exists (A2).
- **Approach C (de-tile → reduce → re-tile copy)** works (it is how
  `alltoall_base` solved the sibling bug, `spyre_ccl.cpp:1067-1084`) but pays a
  full-tensor D2D copy per all_reduce on the per-layer hot path. Rejected as the
  primary; retained only as a conceptual fallback.
- **Approach B (chosen):** reduce **without ever byte-slicing a single tiled
  buffer** — a pairwise `sendrecv` shuffle over whole per-rank chunk *tensors*
  with on-device accumulation. Correct-by-construction for tiled 2-D, copy-free
  at the transport layer, and it **unifies with the parked
  `reduce_scatter`/`exchange_uneven` WIP** that already implements this exact
  primitive.

**Key correctness principle** (from `docs/design/allgatherv-reducescatterv-design.md`
lines 159-170): `SpyreTensorLayout` tiling is a *pure function of (shape, dtype)*.
A whole-tensor (or contiguous-prefix) byte-blob transfer between two
identically-shaped tensors reconstructs the correct tiling on the receiver — the
byte-blob transfer was never the hazard; **slicing at a logical-row offset inside
one already-tiled combined buffer** was. The pairwise shuffle never does that:
every chunk is its own right-sized, freshly-tiled tensor.

## Architecture

`all_reduce(SUM) = reduce_scatter(SUM) → all_gather`, both layout-safe:

```
SpyreCCLBackend::allreduce([tokens,hidden], SUM)
  ≥2-D → new compose path (below); 1-D / already-correct cases unchanged
    │
    ├─ Unit 2: split [tokens,hidden] into N per-rank chunk TENSORS along dim-0
    │          via torch-layer .narrow(0)+clone (own SpyreTensorLayout each) —
    │          NEVER a BufferDesc byte-range slice
    │
    ├─ Unit 3a: reduce_scatter — N-1 pairwise sendrecv rounds + on-device +=
    │           (parked WIP: reduce_scatter + exchange_uneven)
    │              → each rank holds its fully-reduced chunk
    │
    ├─ Unit 3b: all_gather — N-1 pairwise sendrecv rounds (parked WIP / existing)
    │              → every rank reassembles the full reduced [tokens,hidden]
    │
    └─ concat chunks back into the output (torch layer)
```

## Components

### Unit 1 — `allreduce` ≥2-D dispatch (`spyre_ccl.cpp::allreduce`)
- Keep existing correct paths: the committed splittability guard (`byte_count ≥
  world_size × 128`), the 1-D path, and any uniform fast path that is already
  correct.
- For a **≥2-D** input tensor, route to the compose path (Units 2+3) instead of
  `Context::allreduce` → LIBCOLL Ring. Gate strictly on rank ≥ 2 and dim ≥ 2 so
  the currently-correct 1-D/small cases are untouched (no regression).

### Unit 2 — dim-0 chunking at the torch layer (NEW; the one piece the WIP lacks)
- Split `[tokens, hidden]` into N chunks along **dim-0** using `.narrow(0, off,
  len)` on the **logical** tensor, each materialized into its own right-sized
  tensor via `.clone()` / `copy_from_d2d` (own `SpyreTensorLayout`) — the
  `alltoall_base` pattern (`spyre_ccl.cpp:1067-1084`). **Never** a `BufferDesc`
  byte-range slice; **never** rely on `prepare_buffer_desc` for a sub-range
  (it ignores `storage_offset`).
- **Uneven `tokens` (not divisible by N):** reuse the uneven decomposition the
  WIP's `exchange_uneven` already handles (balanced sendrecv + credit-bounded
  remainder).
- **Decode degenerate case (`tokens < N`, e.g. `tokens=1`):** chunking dim-0
  across N ranks is impossible when `tokens < N`, so the reduce-scatter split is
  undefined. **Default (chosen): all_gather-then-local-sum** — every rank
  all_gathers all N whole `[tokens,hidden]` contributions (whole-tensor
  transfers are layout-safe) and sums them on-device. This is unconditionally
  correct for any `tokens` (including `tokens=1`), reuses the layout-safe
  all_gather from Unit 3, and for tiny decode tensors the extra bytes are
  negligible. It also bounds the whole ≥2-D path: use reduce_scatter+all_gather
  when `tokens ≥ N`, else all_gather+local-sum. (Alternative considered and
  rejected as default: pad dim-0 to N and slice off the pad — adds a pad/copy
  and a stickified-layout pad-alignment concern for no benefit at decode sizes.)

### Unit 3 — un-park `reduce_scatter` + `exchange_uneven` + subgroup wireup
- The parked WIP (`spyre_ccl.{cpp,hpp}` ~429 lines + spyre-comms
  `create_context`/`getSubComm` + wireup) implements the pairwise sendrecv
  shuffle with on-device `+=` accumulation and the credit-safe `exchange_uneven`
  sub-leg decomposition. Un-park it as part of this cycle.
- Subgroup pieces are REQUIRED: vLLM TP groups are subgroups of the process (DP
  replicas each own a TP group), so the fix must be correct on subgroups, not
  just world.
- **Known integration risk** (from memory `spyre-subgroup-allgatherv-parked-wip`):
  the WIP changes live world-path comm_id key strings (`g0_` prefix); this needs
  a HW regression run of the existing world-path collectives before commit.

## Data flow (2-D allreduce, per rank)

```
[tokens,hidden] input
  → Unit 2: narrow(0)+clone into N logical chunk tensors           (layout-safe)
  → Unit 3a reduce_scatter: N-1 pairwise sendrecv rounds, on-device +=
       → this rank ends with its fully-reduced chunk
  → Unit 3b all_gather: N-1 pairwise sendrecv rounds
       → every rank reassembles the full reduced [tokens,hidden]
  → concat chunks into output                                      (torch layer)
```

No operation ever addresses a logical-row offset inside a combined tiled buffer;
every transfer is a whole, identically-shaped chunk tensor.

## Testing

1. **Correctness (the bug) — regression test.** Adapt
   `diag_allreduce_split_coverage.sh` into a committed test: 2-D `[8,4096]` at
   4 ranks must yield `reduced=numel, FULL=yes`. Add TP=2 and TP=8; a
   non-divisible case (e.g. `[6,4096]` at 4 ranks); and the decode case
   `[1,4096]` at 4 ranks.
2. **Equivalence.** async≡sync and new-2-D-path≡1-D-reference (reuse the Task-10
   equivalence harness: fresh `torch.ones()`, `torch.equal`).
3. **Subgroup.** Extend the multi-PG teardown test to exercise the new 2-D path
   on a TP subgroup — this validates the un-parked WIP's subgroup wireup. Run
   the world-path regression too (the `g0_` comm_id key change).
4. **Benchmark gate (data artifact, not pass/fail).** Measure new 2-D all_reduce
   latency (p50/p95) for representative **prefill** (large `num_tokens`) and
   **decode** (`num_tokens=1`) `[num_tokens, 4096]` at TP=4, vs. the current
   (broken-but-fast) ring. Quantifies the perf gap and sizes the follow-up.
5. **No regression.** Existing 1-D, uniform allgather, alltoall, P2P, and the
   whole Phase-1 async suite stay green.

**Harness gotcha (learned this cycle):** the diagnostic/regression probe MUST run
as plain `torchrun python …` (NOT under `pytest`) or `tests/conftest.py` + the
`spyre_inference` vLLM platform plugin auto-builds a full Qwen3 engine and times
out before the collective runs.

## Performance: explicit scope and tracked follow-up

The pairwise sendrecv shuffle moves comparable total bytes to a ring
(≈`S·(N−1)/N` per rank for reduce-scatter) but does **not** pipeline/overlap its
N−1 rounds and pays a per-round rendezvous cost (`address_exchange_.invalidate()`
+ wireup handshake per call). So its wall-clock latency is higher than a
bandwidth-optimal *pipelined* ring — worst in the **decode/small-message** case
where per-round fixed overhead dominates.

This is accepted deliberately:
- **Correctness beats speed:** TP≥4 currently produces wrong logits silently; a
  correct-but-not-optimal all_reduce is strictly better than a fast wrong one.
- **The optimal ring is exactly what's broken** (it slices a tiled buffer by
  linear offset); making it layout-safe needs new LIBCOLL split machinery.
- **The overlap a pipelined ring needs is blocked anyway** — single
  `comm_stream_` + per-call `address_exchange_.invalidate()` mean concurrent
  in-flight collectives on one context aren't safe today (the same runtime
  constraint the async Phase-1 work began addressing).

**Tracked follow-up (out of scope here):** a topology-aware / pipelined ring
reduce-scatter for NCCL-competitive latency, once the async-overlap runtime work
unblocks it. The benchmark gate (Testing #4) produces the data to prioritize it.

## Explicitly out of scope

- Bandwidth-optimal pipelined/topology-aware ring reduce-scatter (follow-up).
- Round-overlap/pipelining (blocked on the broader async-start redesign).
- `_reduce_scatter_base` / `_allgather_base` single-tensor c10d forms
  (separate follow-up per the allgatherv design doc).
- Non-SUM reductions (backend only supports SUM today).

## Risks

- **Subgroup wireup `g0_` comm_id change** (memory): HW-regress world-path
  collectives before commit.
- **Decode `tokens=1 < N`** degenerate path: must be explicitly correct, not an
  afterthought — it is the decode hot path.
- **Un-parking the WIP** re-activates ~429 lines + spyre-comms subgroup changes
  that have not been HW-run since parking; treat as new code under full review,
  not "known-good".
- **Perf regression vs. the broken-but-fast ring** is expected and accepted;
  the benchmark gate documents its size so it is not a silent surprise.
