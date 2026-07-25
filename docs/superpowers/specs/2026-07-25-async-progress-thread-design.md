# Async collective dispatch — c10d background progress thread (Phase 1)

**Date:** 2026-07-25
**Component:** `torch-spyre/torch_spyre/csrc/distributed/spyre_ccl.{cpp,hpp}`
**Status:** Design — approved architecture (Option A), pending spec review
**Scope:** Part 1 of the "async is not async" P0. Phase 2 (in-comms per-stream
DMA worker / device-event fences) is a separate later spec.

---

## 1. Problem

A c10d collective on `SpyreCCLBackend` is **synchronous to completion today**.
Every collective entry point (e.g. `allreduce`, `spyre_ccl.cpp:796`) runs this
sequence *on the caller thread*:

```
order_after_caller_stream(t)   // spyre_ccl.cpp:597-599 -> getCurrentStream().synchronize(): HOST BLOCKS
ws->SetStreamAffinity(comm_stream_)
ws->start()                    // work_schedule.cpp: runs the ENTIRE LaunchAll() dispatch inline,
                               //   with per-op blocking synchronize() fences, on THIS thread
return make_intrusive<SpyreCCLWork>(...)   // ...but the transfer has already fully drained
```

The returned `Work` is a formality: by the time the Python caller receives it,
the collective is done. `async_op=True` returns late; `wait()` is a no-op.
**Overlap is structurally impossible** regardless of what the caller does —
even a perfectly-pipelined vLLM TP layer gets zero benefit, because the
all-reduce fully drains before the next Python line executes.

This violates the c10d contract (async collectives must return before
completion) and forfeits the one performance lever the hardware does offer
(see §2).

---

## 2. Hardware analysis — why Option A is the only viable shape

This is the load-bearing rationale. Two senlib architecture documents were
reviewed to decide between a single-FIFO progress thread (**A**) and a
per-stream worker pool (**B**), and to decide how the caller→comm-stream
ordering handoff should work.

### 2.1 `senlib/docs/35_stream_support_analysis_1p5.md` — no stream concurrency

- **No CUDA-style streams.** AIU 1p5 exposes hardware *submission queues*
  (DMGR, WQM) with barrier semantics, not concurrent execution streams
  (doc §"Executive Summary", §"Why AIU 1p5 Is NOT CUDA Streams").
- **No concurrent kernel execution.** The compute queue (QGI) has **4 entries**
  and processes **one kernel at a time**; "Concurrent kernel limit: 1"
  (doc table, line 164).
- **No concurrent submission.** 16 AppID channels exist, but "firmware supports
  AppID 0 only" (doc line 883) — multiple host threads cannot even *post* work
  in parallel today.
- **Separate DMGR queues for DMA vs compute.** DMAI/DMAO have 32 entries each,
  distinct from QGI's 4 (doc lines 54-62, 1150-1159). This is the one genuine
  overlap axis: DMA-queue work can progress while compute-queue work runs,
  *provided the host thread is free to keep both fed*.
- **Firmware dispatch latency 5-20 µs; per-kernel 50-100 µs** (doc lines 561,
  682). Keeping the queues fed rather than host-round-tripping between every op
  is therefore a real launch-latency win.

### 2.2 `senlib/docs/36_device_side_synchronization_analysis.md` — no device-side cross-stream wait

Available sync primitives, and why none serves as a device-side caller→comm
handoff:

| Primitive | What it is | Usable for the handoff? |
|---|---|---|
| CB barrier (`SetBarrier`) | Ordering within one submitted CB series | No — intra-stream only; "no overlap, sequential" (doc line 106) |
| RDMA Wdone | DMAI engine blocks on remote-write completion | No — RDMA-context-specific (doc line 201) |
| Multi-AIU Signal/Wait | Cross-*device* sync via a **host-memory** coordination buffer (doc line 303) | No — cross-device collective building block, host-buffer-backed |
| Async events | Device→host error/telemetry notifications | No — one-way (doc line 424) |
| Mark bit | Per-CB completion flag the host polls | No — "completion tracking, not inter-CB signaling" (doc line 492) |

Explicit negatives (doc lines 496-593): **no hardware semaphores, no inter-CB
signaling, no `cudaStreamWaitEvent` equivalent, no memory fences.** The
document's own recommended producer-consumer pattern is **host-assisted**
(`submitNonBlocking` → host `future.wait()` → submit consumer, doc line 810) —
which is exactly Option A's worker-side host-sync handoff.

`flex::RuntimeEvent` (`flex/src/runtime_stream/runtime_event.cpp`) confirms this
at the runtime layer: `record()` calls `stream->synchronize()` (host-blocks,
line 24) and `wait(stream)` **ignores its stream argument** and host-blocks on a
`std::future` (lines 29-42). It is a host-thread completion fence, not a
device-side stream-ordering primitive — and per §2.2 there is nothing at the
1p5 ISA/firmware level for a device-side variant to lower to.

### 2.3 Verdict

Option **B** (per-stream worker pool) is rejected on four independent hardware
grounds: no concurrent kernel execution, no concurrent submission, no stream
events, and cross-stream/-device coordination is host-assisted by design. B
would add threads and a cross-rank ordering hazard while buying zero device
concurrency.

Option **A** is the only shape congruent with the hardware: a single serialized
submission path matches how the firmware actually consumes work, and the
host-sync handoff is the vendor's own blessed producer-consumer pattern — not a
fallback.

**Consequence for this design:** the "device-side event handoff" previously
considered as a near-term seam is *deleted*. It over-promises against 1p5
hardware. If a future chip adds a hardware stream scheduler (doc 35 "Long-Term"
section), that is a new design, not a hook pre-built now. The existing
`order_after_caller_stream` TODO (spyre_ccl.cpp:590-595) referencing a
device-event follow-up is downgraded accordingly.

### 2.4 What Phase 1 actually unlocks (honest accounting)

- **Host-thread unblocking (primary win).** The caller thread stops eating the
  `synchronize()` + full `LaunchAll()`. It returns and keeps submitting
  subsequent independent work, so the DMGR queues stay fed instead of
  host-round-tripping between ops.
- **Bounded device DMA/compute overlap.** The collective's DMA-queue traffic can
  progress while the caller's next compute-queue op runs — bounded by firmware
  dispatch and HBM bandwidth, *not* the free lunch CUDA gets. Not concurrency of
  two collectives.
- **Contract correctness.** `Work` stops lying: `async_op=True` genuinely
  returns early, `wait()` genuinely waits. Hardware-independent, always valid.
  Couples directly to the already-landed query()-lifecycle-gate fix (`202f152`).

Phase 1 unlocks **no new collective functionality** — same op set, same
SUM-only / balanced / uniform / single-chunk constraints.

---

## 3. Architecture (Option A)

One persistent FIFO **progress thread** per `SpyreCCLBackend`, owning all
`comm_stream_` launches. The caller thread does only cheap, thread-safe work and
returns a real (not-yet-complete) `Work` immediately.

```
Caller thread (Python/c10d entry, e.g. allreduce)          Progress thread (1 per backend)
--------------------------------------------------          --------------------------------
abort_guard(); validate; prepare_buffer_desc                 loop:
build ws = group_context_->allreduce(buf, op)                  job = queue.pop()            // blocks
capture caller SpyreStream handle (thread-local!)              caller_stream.synchronize()  // host fence (§2.2)
ws->SetStreamAffinity(comm_stream_)                            job.ws->start()               // LaunchAll on WORKER
enqueue job {ws*, caller_stream, work_state}                   job.ws->wait()                // drain on WORKER
seq_.fetch_add(1)                                              work_state -> DONE_SUCCESS / DONE_ERROR
return SpyreCCLWork(work_state, ws-ref, hold, result)          (on throw: report_and_abort + DONE_ERROR)
```

### 3.1 Invariants (must hold, in priority order)

1. **Single FIFO per rank.** Exactly one progress thread per backend, one
   in-order queue. This preserves cross-rank collective launch order *by
   construction* — the property the whole system's deadlock-freedom depends on
   (HDMA barrier / pipeline-submission / CBS-matcher deadlock history). This is
   the highest-priority invariant; do not add parallelism here.
2. **Caller-stream captured on the caller thread.** `spyre::getCurrentStream()`
   is thread-local; it MUST be read in the entry point and passed in the job.
   Reading it on the worker returns the wrong stream.
3. **Completion gates on the Work's own state, never `ws->query()` blindly.**
   The worker opens a window where the schedule is enqueued-but-not-started;
   `ws->query()` on an unstarted schedule can report "done" (the query()-lies
   class). `isCompleted()`/`wait()` must consult the job's own lifecycle state
   first.
4. **`ws` ordering on the comm stream is unchanged.** `SetStreamAffinity` +
   `start` + `wait` semantics inside spyre-comms are untouched; they simply run
   on the worker. Zero spyre-comms change in Phase 1.
5. **Tensor lifetime.** `SpyreCCLWork` continues to hold `hold_tensors_` until
   the schedule drains (existing dtor drain, spyre_ccl.cpp:1663-1682). The
   worker must not free anything the schedule still references.

### 3.2 Collective-scope for Phase 1

Async dispatch applies to the **single-`Work`-return, fully-correct** ops:
`allreduce` (SUM), `broadcast`, `gather`, uniform `allgather`, `send`, `recv`,
`barrier`.

**Explicitly excluded from Phase 1:**
- **Multi-leg inline-wait paths** (`alltoall` 848, `alltoall_base` 934,
  `reduce_scatter` 1328, uneven `allgather` 713-776). These call
  `start()+wait()` inline *between legs* to enforce leg-to-leg data
  dependencies. They stay synchronous in Phase 1; making them async is coupled
  to the Phase 2 decomposition rewrite. They already throw cleanly on their
  unsupported (asymmetric/variable) cases.
- **Host-staged reduce path.** `reduce` and any allreduce host-fallback run the
  D2H→host-sum→H2D path flagged for the use-after-free in the
  allreduce-scratch-compute-corruption investigation. Do **not** place the
  host-reduce path on the background worker until that UAF is confirmed fixed.
  If `allreduce`'s SUM path can route to host-reduce under any condition, that
  condition must remain synchronous or be proven UAF-free before inclusion.

---

## 4. Components

### 4.1 Job struct (backend-private)

```cpp
// Owned by the queue; one per enqueued collective.
struct ProgressJob {
  // The schedule to drive lives in WorkState::ws (single owner). The worker
  // reaches it via state->ws; the job carries no separate ws pointer, so there
  // is exactly one owner and no raw-pointer/owner race.
  spyre::SpyreStream caller_stream;       // captured on the caller thread (invariant 2)
  std::shared_ptr<WorkState> state;       // shared with the SpyreCCLWork (§4.2)
};
```

### 4.2 WorkState — the shared lifecycle (new)

The single source of truth for "where is this collective," shared (via
`shared_ptr`) between the `SpyreCCLWork` and the worker. Replaces reliance on
`ws->query()` for the not-yet-started window (invariant 3).

```cpp
enum class ProgressState { ENQUEUED, LAUNCHED, DONE_SUCCESS, DONE_ERROR };

struct WorkState {
  std::atomic<ProgressState> state{ProgressState::ENQUEUED};
  std::atomic<bool> cancelled{false};     // set by ~SpyreCCLWork on a still-ENQUEUED job (R3)
  std::string error_reason;               // set before state=DONE_ERROR; read after
  // ws ownership lives here so the worker and Work agree on its lifetime.
  std::unique_ptr<spyre_comms::WorkSchedule> ws;
};
```

State transitions (worker only, except construction):
`ENQUEUED --pop--> LAUNCHED (before start) --wait ok--> DONE_SUCCESS`
`(any exception in synchronize/start/wait) --> DONE_ERROR (+error_reason)`

### 4.3 Progress thread (new, `SpyreCCLBackend`)

- Started in the constructor (after `comm_stream_`/watchdog setup,
  spyre_ccl.cpp:172-181), joined in the destructor **before**
  `finalize_library()` (alongside the watchdog join, spyre_ccl.cpp:183-192).
- Members mirror the watchdog pattern: `std::thread progress_thread_`,
  `std::atomic<bool> progress_stop_`, a bounded MPSC queue + `mutex`/`condvar`,
  or the existing `external/concurrentqueue` (flex vendors it) if a lock-free
  MPSC is preferred. **Decision:** use a `std::mutex`+`std::condition_variable`
  bounded deque for Phase 1 — simplest to reason about, the enqueue rate is low
  (one per collective), and it makes shutdown-drain semantics explicit. Revisit
  only if profiling shows enqueue contention.
- Loop:
  ```
  while (!progress_stop_ || queue not empty):
    wait for job or stop
    if stop and queue empty: break
    job = pop()
    if job.state->cancelled: continue            // Work destroyed before launch (R3); nothing to drain
    if aborted_: state -> DONE_ERROR("aborted"); continue
    try:
      job.caller_stream.synchronize()      // §2.2 host fence, MOVED off caller thread
      job.state->state = LAUNCHED
      job.state->ws->start()               // synchronous LaunchAll, on the worker
      job.state->ws->wait()                // drain, on the worker
      job.state->state = DONE_SUCCESS
    catch (e):
      job.state->error_reason = e.what()
      job.state->state = DONE_ERROR
      report_and_abort("<op>: " + e.what())   // same fail-fast as today's catch blocks
  ```

### 4.4 `SpyreCCLWork` changes

- Holds `std::shared_ptr<WorkState>` instead of owning the
  `unique_ptr<WorkSchedule>` directly (the `ws` moves into `WorkState::ws`).
- `isCompleted()`:
  - if `completed_` → true
  - read `state->state`: `DONE_SUCCESS` → `finish_success()` (CAS-guarded);
    `DONE_ERROR` → `finish_error(reason)`; else → false.
  - **Never** calls `ws->query()` for the terminal decision (invariant 3). It
    may still call `ws->query()` *only after* observing `LAUNCHED`/`DONE_*` if a
    finer-grained in-flight check is ever needed — but the state machine is
    authoritative.
- `wait(timeout)`:
  - Poll `state->state` with the existing timeout/`needsShutdown()` logic
    (spyre_ccl.cpp:1728-1817), swapping the `while (!ws->query())` condition for
    `while (state->state == ENQUEUED || state->state == LAUNCHED)`.
  - `needsShutdown()` check retained (peer failure via watchdog).
  - On deadline: `finish_error` + throw, as today.
- `~SpyreCCLWork()`: drain semantics preserved. Because the worker owns the
  launch, the dtor must wait until `state` is terminal (or the schedule drained)
  before releasing `hold_tensors_`. Concretely:
  - if `state` is `ENQUEUED`: set `state->cancelled = true` so the worker skips
    it without launching; no drain needed (nothing touches the tensors yet).
    (R3 — avoids blocking on a drain that would never start.)
  - if `state` is `LAUNCHED`: block on `state->ws->synchronize()` as today
    (spyre_ccl.cpp:1674-1681) — the schedule is live on the worker and may be
    reading `hold_tensors_`.
  - if `state` is `DONE_*`: no drain needed.
  The ENQUEUED→LAUNCHED transition and the dtor's decision must not race (a
  naive read-then-cancel has a TOCTOU: dtor reads ENQUEUED, worker launches and
  starts, dtor frees live tensors). Resolve with a per-`WorkState` mutex held
  across (a) the worker's "check cancelled, then set LAUNCHED, then start()"
  and (b) the dtor's "read state, set cancelled-or-choose-drain" so the two are
  mutually exclusive. The mutex is held only around the cheap state decision,
  never across the actual `start()`/`synchronize()` work. Exact locking is
  finalized in the implementation plan.

### 4.5 Entry-point changes (per in-scope op)

Replace the inline `order_after_caller_stream(t); ws->SetStreamAffinity;
ws->start(); return Work(...)` tail with:

```cpp
ws->SetStreamAffinity(comm_stream_);
auto state = std::make_shared<WorkState>();
state->ws = std::move(ws);
auto caller_stream = spyre::getCurrentStream(t.device());   // invariant 2: caller thread
enqueue(ProgressJob{caller_stream, state});   // worker reaches the schedule via state->ws
seq_.fetch_add(1, std::memory_order_relaxed);
return c10::make_intrusive<SpyreCCLWork>(OpType::X, state, hold, result, op_timeout_);
```

`order_after_caller_stream` is no longer called on the caller thread; its host
fence moves into the worker (§4.3). The function itself is retained (the worker
calls the equivalent `caller_stream.synchronize()`), and the multi-leg excluded
ops still call it inline as today.

---

## 5. Cross-cutting concerns

### 5.1 Cross-rank ordering
Guaranteed by invariant 1 (single FIFO, one thread). The `seq_` counter
increments on enqueue in program order; the worker consumes in that order; every
rank does the same → matching collective order across ranks. No change to the
spyre-comms matcher.

### 5.2 Shutdown / abort
- `abort()`/`shutdown()` (spyre_ccl.cpp:1545-1569) set `aborted_`. The worker
  checks `aborted_` before launching each job and fails queued jobs to
  `DONE_ERROR` without launching (matches "no new collectives once aborted").
- `report_and_abort()` (1571-1609) is now *also* invoked from the worker's catch
  block — it is already idempotent (CAS on `aborted_`, 1573) and thread-safe, so
  concurrent calls from a collective's own path and the worker converge.
- Destructor order: set `progress_stop_`, notify, **join progress thread**, then
  join watchdog, then `finalize_library()`. The progress thread must exit before
  `finalize_library()` (it touches `comm_stream_`). Queued-but-unlaunched jobs at
  shutdown transition to `DONE_ERROR("backend shutting down")`.

### 5.3 Error propagation
Unchanged in surface: `Work::wait()` throws on `DONE_ERROR`, `isSuccess()`
reflects it, the Future is completed via `finish_error`. The only change is the
*source* of the terminal state (WorkState set by worker) vs. today (read from
`ws->getState()` inline). `getShutdownReason()` still consulted for the
peer-failure reason.

### 5.4 Interaction with the existing watchdog
Independent thread, unchanged. It sets `aborted_` + comm-stream shutdown on peer
failure; the worker observes `aborted_`/`needsShutdown()` and fails its current
and queued jobs promptly. No new coupling beyond both calling the idempotent
`report_and_abort`.

---

## 6. Risks / limitations

- **R1 — worker still host-blocks on `caller_stream.synchronize()`.** This is
  inherent to 1p5 (§2.2); it moves the block off the *caller* thread, which is
  the win. Not removable without new hardware.
- **R2 — bounded overlap only.** Per §2.4, the device-level overlap is DMA-queue
  vs compute-queue, bandwidth/dispatch-limited. If the target workload has
  little independent compute after its collectives, the win is small and the
  real latency lever is Phase 2 (chunked intra-collective overlap). Stated
  plainly so expectations match hardware.
- **R3 — destroy-before-launch.** A `SpyreCCLWork` destroyed while its job is
  still `ENQUEUED` (never launched) must not deadlock waiting for a drain that
  will never start. The dtor must distinguish `ENQUEUED` (cancel/skip: the
  worker will fail it, or it can be marked cancelled) from `LAUNCHED` (must
  drain). Handled via a `cancelled` flag on `WorkState` checked by the worker
  before launch.
- **R4 — single-FIFO serializes independent collectives.** By design (invariant
  1). On this hardware there is no concurrency to give up (§2.1). Accepted.
- **R5 — host-reduce UAF.** Excluded from scope (§3.2); must not regress. A
  guard/assertion that the host-reduce path is never enqueued is included.

---

## 7. Testing / verification

- **Unit (host-buildable where possible):** WorkState transition logic and the
  `isCompleted`/`wait` gating are pure enough to test without hardware — mirror
  the existing pure-function test pattern.
- **HW regression (torchrun, ≥2 ranks):** the existing collective gtests
  (`MultipleCalls`, `SendRecv`, `MultipleBarriers`, `UnevenSplitAllReduce`) must
  stay green — they exercise the ordering + reset/reuse paths. Reuse the
  installed-binary + device-count-guard script pattern already established
  (`run_query_state_gate_verify.sh`, `run_unevensplit_fp16_verify.sh`).
- **Async-specific:** a test that issues an in-scope collective with
  `async_op=True`, asserts the call returns before completion (observably: the
  Work is not yet complete immediately after return), then `wait()`s and
  verifies the numeric result — this is the behavioral proof the `Work` no
  longer lies.
- **Ordering stress:** back-to-back distinct collectives on multiple ranks to
  confirm the single-FIFO preserves cross-rank order (no matcher deadlock).
- **Abort/shutdown:** issue collectives, abort mid-flight, confirm queued jobs
  fail to `DONE_ERROR` and the destructor joins cleanly (no hang).

Build/deploy is done by the user; verification runs the installed binary via the
established remote pattern. Confirm the exact rebuild command before the first
torch-spyre C++ rebuild.

---

## 8. Out of scope (Phase 2, separate spec)

- In-comms per-stream DMA worker + device-event fences inside `LaunchAll`
  (chunked intra-collective compute/comm overlap).
- Async-enabling the multi-leg decomposition ops (alltoall*, reduce_scatter,
  uneven allgather) and completing their asymmetric/variable (MoE) forms.
- Host-reduce UAF fix (prerequisite for putting `reduce`/host-fallback on the
  worker).
- Any device-side cross-stream event primitive (absent on 1p5; a future-chip
  concern).
