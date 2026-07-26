# Async collective dispatch — c10d background progress thread (Phase 1)

**Date:** 2026-07-25
**Component:** `torch-spyre/torch_spyre/csrc/distributed/spyre_ccl.{cpp,hpp}`
**Status:** Design — v3. v1 REJECTed (caller-thread build unsafe); v2
APPROVE-WITH-CHANGES (worker-build model correct but wrongly scoped per-backend);
v3 scopes the worker process-global per the C1/C2 findings, confirmed by a perf
analysis + a PG-topology check. Companion: `2026-07-25-async-phase1-performance-
analysis.md`. Pending final review.
**Scope:** Part 1 of the "async is not async" P0. Phase 2 (in-comms per-stream
DMA worker / device-event fences) is a separate later spec.

---

## 0. Revision note (what changed from v1, and why)

v1 proposed: **build the `WorkSchedule` on the caller thread**, then enqueue it and
run only `start()`/`wait()` on a background worker. An independent review plus a
code investigation showed this is **unsafe by construction**:

- **BUILD is not a local operation.** For every in-scope op, building the schedule
  performs **blocking, FIFO-order-sensitive OOB address exchange** on the
  process-global `SocketOOB` channel (`context.cpp:740`→`Convert`→
  `preExchangeAddresses`, `context.cpp:758-770`→`address_exchange_.pre_exchange`,
  blocking `oob->send/recv` at `address_exchange_cache.cpp:144-161`). Raw
  send/recv/gather/barrier also do blocking OOB at build time.
- **START is its own order-sensitive cross-rank rendezvous** (HDMA send-credit +
  per-CBS completion barriers + CBS matching, `work_schedule.cpp:200-341`,
  `:629-670`).
- If the caller races into BUILD(N+1)'s OOB while the worker is still in START(N)'s
  HDMA rendezvous, **different ranks interleave the OOB and HDMA channels in
  different global orders**, desynchronizing the tag-less per-peer OOB FIFO
  (`address_exchange_cache.cpp:129-135`) → deadlock/mismatch — the exact class this
  system has repeatedly hit.

The investigation also confirmed the *good* news that shapes the fix:
`LaunchAll()` **never re-reads** `address_exchange_`/`wireup_cache_`/`oob_channel`;
it replays addresses baked into the schedule's operations at build time
(`work_schedule.cpp:122-135`, confirmed). So the caches are not the hazard — the
**cross-rank channel ordering** is.

**v2 fix:** move **both BUILD and START onto a single FIFO worker.** The caller
thread does only thread-local/cheap capture and enqueues a *request*. The worker
does BUILD(N)→START(N)→wait(N)→BUILD(N+1)… strictly in enqueue order. Because
enqueue order = program order = identical across SPMD ranks, the OOB(N),
HDMA(N), OOB(N+1)… traffic lines up in the same relative order on every rank →
cross-rank ordering preserved by construction. The worker becomes the **sole
driver** of the schedule, `comm_stream_`, the allocator, and the OOB channel,
which also removes the two-thread stream-sharing hazard and the v1 double-driver
bug.

**v3 fix (the C1/C2 findings — the load-bearing scoping correction):** the
resources the worker drives — `comm_stream_` (`get_comm_stream()`, a
process-global singleton, spyre_comms.cpp:394-400), the OOB channel, and the
allocator — are **process-global, shared by every PG in the process.** And a
single process instantiates **multiple** `SpyreCCLBackend`s: vLLM's
`initialize_model_parallel` unconditionally builds the world PG plus
TP/PP/DP/DCP/PCP(/EP) subgroups via `new_group` on the `spyre:spyreccl` backend
(parallel_state.py:1766-1876), and torch-spyre's `test_subgroup.py` runs world+TP+DP
backends concurrently, documenting the shared per-process `comm_stream_` singleton
(test_subgroup.py:284-292). Therefore **one worker per backend would race sibling
PGs on the shared stream/OOB** — reintroducing the exact cross-rank ordering hazard
across PGs (a regression: today the single caller thread serializes PG-A fully
before PG-B). v3 makes the progress thread + FIFO **process-global**: exactly one
per process, shared by all backends, refcounted to
`initialize_library`/`finalize_library`. On 1p5 this costs **zero** device
concurrency — all PGs already serialize on the one `comm_stream_` (no concurrent
kernel execution, senlib doc 35) — so the process-global FIFO only imposes program
order on already-serialized work. See the companion performance analysis. This also
requires the teardown-unblock to be **backend-local**, never the process-fatal
`report_and_abort`/`setShutdown` on the shared stream (§5.2).

---

## 1. Problem

A c10d collective on `SpyreCCLBackend` is **synchronous on the caller thread
today**. Each entry point (e.g. `allreduce`, `spyre_ccl.cpp:796`) runs this
sequence on the caller thread:

```
buf = prepare_buffer_desc(t)
ws  = group_context_->allreduce(buf, op)   // BUILD: blocking OOB address pre-exchange (context.cpp:740)
order_after_caller_stream(t)               // host fence: getCurrentStream().synchronize() (spyre_ccl.cpp:597-599)
ws->SetStreamAffinity(comm_stream_)
ws->start()                                // START: LaunchAll() -> HDMA/CBS cross-rank rendezvous; sets RUNNING
return make_intrusive<SpyreCCLWork>(...)    // Work returned; final drain deferred to Work::wait()
```

Precise cost accounting (correcting v1, which wrongly said `start()` fully drains
and `wait()` is a no-op):

- `start()` runs `LaunchAll()` and sets `RUNNING` (`work_schedule.cpp:45,78-79`); it
  does **not** host-drain. The final drain happens in `SpyreCCLWork::wait()` →
  `WorkSchedule::wait()` → `WaitAll()` → `stream->synchronize()`
  (`work_schedule.cpp:99,190-192`). So `wait()` today is real work.
- The **synchronous host cost on the caller thread** is therefore: the BUILD-time
  blocking OOB pre-exchange, plus `order_after_caller_stream`'s host fence, plus
  `start()`'s inline CBS/HDMA cross-rank barriers.

The returned `Work` is nonetheless effectively synchronous for overlap purposes:
the caller has already paid the OOB + submission cost inline, and there is no
window in which the caller's *subsequent* independent work can proceed while the
collective's cross-rank coordination happens. `async_op=True` does not free the
caller thread. This violates the c10d async contract and forfeits the host-side
pipelining the hardware does permit (§2.4).

---

## 2. Hardware analysis — why Option A is the only viable shape

Load-bearing rationale for a single-FIFO progress thread (**A**) over a per-stream
worker pool (**B**), from two senlib architecture docs.

### 2.1 `senlib/docs/35_stream_support_analysis_1p5.md` — no stream concurrency
- No CUDA-style streams; hardware exposes submission queues with barrier
  semantics, not concurrent execution streams.
- No concurrent kernel execution: QGI compute queue has **4 entries**, "Concurrent
  kernel limit: 1" (doc line 164).
- No concurrent submission: 16 AppID channels exist but "firmware supports AppID 0
  only" (doc line 883).
- Separate DMAI/DMAO (32 entries) vs QGI queues (doc lines 54-62): the one genuine
  overlap axis is DMA-queue vs compute-queue progress, *if the host thread is free
  to keep both fed*.
- Firmware dispatch 5-20 µs, per-kernel 50-100 µs (doc lines 561, 682): keeping
  queues fed beats host round-trips.

### 2.2 `senlib/docs/36_device_side_synchronization_analysis.md` — no device-side cross-stream wait
No device-side cross-stream event exists (no `cudaStreamWaitEvent` equivalent, no
hardware semaphores, no inter-CB signaling; doc lines 496-593). CB barriers are
intra-stream only; Multi-AIU Signal/Wait is cross-device and host-buffer-backed;
mark bits are completion-tracking only. The doc's own recommended
producer-consumer pattern is **host-assisted** (doc line 810), which is exactly
this design's worker host-sync handoff. `flex::RuntimeEvent`
(`runtime_event.cpp`) confirms it: `record()` host-blocks via `synchronize()`
(line 24); `wait(stream)` ignores its stream arg and host-blocks (lines 29-42).

### 2.3 Verdict
**B is rejected** on four independent hardware grounds (no concurrent kernels, no
concurrent submission, no stream events, host-assisted-only cross coordination)
plus the cross-rank ordering hazard a multi-worker pool would add. **A** — one
serialized FIFO — matches how the firmware consumes work and how the OOB/HDMA
channels must be ordered across ranks. The device-side event handoff considered in
early drafts is **deleted** as unimplementable on 1p5; if a future chip adds a
hardware stream scheduler that is a new design, not a pre-built hook. The
`order_after_caller_stream` TODO (spyre_ccl.cpp:590-595) is downgraded accordingly.

### 2.4 What Phase 1 actually unlocks (honest accounting)
- **Host-thread unblocking (primary win).** With BUILD+START both on the worker
  (§0), the caller thread pays *neither* the OOB pre-exchange nor the LaunchAll
  submission nor the drain. It returns after cheap capture and keeps submitting
  independent work, keeping the DMGR queues fed.
- **Bounded device DMA/compute overlap.** The collective's DMA-queue traffic can
  progress while the caller's next compute-queue op runs — bounded by firmware
  dispatch and HBM bandwidth, not CUDA-style free concurrency. Not concurrency of
  two collectives (single FIFO, §3.1).
- **Contract correctness.** `Work` stops lying: `async_op=True` returns early,
  `wait()` genuinely waits. Couples to the landed query()-gate fix (`202f152`).

Phase 1 unlocks **no new collective functionality** (same op set, same SUM-only /
balanced / uniform / single-chunk constraints).

---

## 3. Architecture (Option A, v3: one process-global worker builds and runs)

Exactly **one** persistent FIFO **progress thread per process** (not per backend),
shared by the world PG and every subgroup. It is the **sole driver** of schedule
build, the process-global `comm_stream_`, the allocator, and the OOB channel. The
caller thread does only thread-local capture and enqueues a request. Lifetime is
refcounted to `initialize_library`/`finalize_library` (started on first library
init, joined+destroyed on the last finalize), mirroring the existing refcounted
global lifecycle (`GlobalInfo::incInit/decInit`).

```
Caller thread (c10d entry, e.g. allreduce)              Process-global progress thread (1/process, sole driver)
------------------------------------------              ---------------------------------------------
abort_guard(); validate tensors                          loop:
buf = prepare_buffer_desc(t)      // local                 req = queue.pop()                 // blocks; FIFO
aux = prepare output-slot descs   // C4, caller thread      if req.state->cancelled: continue           // R3
caller_stream = getCurrentStream(t.device())  // TLS!       if req.aborted_flag: state -> DONE_ERROR("aborted"); continue
hold refs to input/output tensors                          try:
enqueue Request{op, ctx, buf, aux, params,                   ws = req.context->OP(req.buf, req.aux, ...)  // BUILD: OOB pre-exchange
       caller_stream, op_timeout, aborted_flag, state}       req.caller_stream.synchronize()    // producer->collective fence
seq_.fetch_add(1)                                            ws->SetStreamAffinity(get_comm_stream())
return SpyreCCLWork(state, hold, result, op_timeout_)        state->ws = move(ws); state = LAUNCHED
                                                             wait_interruptible(state->ws, req.op_timeout)  // start()+drain (§4.3 F5)
                                                             state = terminal_from(ws)          // §4.3 F1 mapping
                                                           catch (e):
                                                             state->error_reason = e.what(); state = DONE_ERROR
                                                             req.on_error(e)   // backend-local abort hook, NOT process-fatal (§5.2)
```

### 3.1 Invariants (priority order)

1. **One process-global FIFO preserves cross-rank order by construction.** Exactly
   one progress thread + one in-order queue **per process**, shared across all PGs
   (world + subgroups). Because BUILD *and* START both run on this thread in enqueue
   order, and enqueue order = program order = identical across SPMD ranks, the
   OOB(N)/HDMA(N)/OOB(N+1) cross-rank traffic lines up in the same relative order on
   every rank — *including* across different PGs, since they share the one FIFO and
   the one `comm_stream_`/OOB channel. This is the property the whole system's
   deadlock-freedom depends on. Do not add parallelism here (per-backend workers
   would race sibling PGs on the shared singletons — the C1 regression).
2. **The process issues collectives from one thread.** Enqueue order is well-defined
   and cross-rank-consistent only if the process drives its collectives (across all
   PGs) from a single thread — the standard c10d SPMD usage (the inference/training
   loop thread). Enqueue takes a mutex so the queue can never corrupt, but cross-rank
   ordering *correctness* rests on this single-issuer assumption. If a process ever
   issues collectives from multiple threads concurrently (across PGs or within one),
   enqueue order becomes nondeterministic and invariant 1 breaks. Documented as a
   precondition; revisit if multi-threaded issue is ever required.
3. **Caller-stream captured on the caller thread.** `spyre::getCurrentStream()`
   reads a `thread_local` (`spyre_stream.cpp:70-71`); it MUST be read in the entry
   point and passed in the request. The captured `SpyreStream` is a value handle
   (`c10::Stream(UNSAFE, device, id)`); `synchronize()` resolves the runtime stream
   via the global pool, so calling it on the worker is safe (confirmed in review).
4. **Completion gates on WorkState, never `ws->query()` blindly.** The worker opens
   windows where the request is enqueued-but-not-built and built-but-not-started;
   `ws->query()` is meaningless or misleading there (`query()` returns false for
   IDLE, `work_schedule.cpp:179-180`). `isCompleted()`/`wait()` consult the
   request's own `WorkState` (§4.2).
5. **The one process-global worker is the sole driver of every `ws`,
   `comm_stream_`, the allocator, and OOB — process-wide.** No other thread (no
   caller, no other backend) calls `start()/wait()/synchronize()` on any schedule or
   submits to `comm_stream_`. This is true only because the worker is process-global
   (v3); a per-backend worker would violate it across sibling PGs. `SpyreCCLWork`
   destructors wait on a condition variable, never on `ws` directly (§4.4). Removes
   the v1 double-driver race (F6), the two-thread stream-sharing hazard
   (investigation Q5a), and the C1 cross-PG race.
6. **Tensor lifetime.** `SpyreCCLWork` holds input/output tensors until the
   schedule reaches a terminal state (worker-signaled), because the schedule
   captures raw device pointers into that memory.

### 3.2 Collective scope for Phase 1

In scope (single-`Work`-return, fully-correct ops): `allreduce` (SUM),
`broadcast`, `gather`, uniform `allgather`, `send`, `recv`, `barrier`.

Excluded:
- **Multi-leg inline-wait paths** (`alltoall` 848, `alltoall_base` 934,
  `reduce_scatter` 1328, uneven `allgather` 713-776): they call `start()+wait()`
  inline *between legs* for leg-to-leg data dependencies. They stay synchronous;
  async-enabling them is coupled to the Phase 2 decomposition rewrite. They already
  throw cleanly on unsupported asymmetric/variable cases.
- **Host-staged reduce path** (`reduce`, and any allreduce host fallback): the
  D2H→host-sum→H2D path implicated in the allreduce-scratch-compute-corruption UAF.
  See §4.6 for the concrete guard (not just a scope note — F8).

---

## 4. Components

### 4.1 Request struct (backend-private)

```cpp
// One per enqueued collective. Carries everything the ONE process-global worker
// needs to BUILD+RUN against the RIGHT PG's context. No WorkSchedule here — the
// worker builds it (the point of v2).
struct ProgressRequest {
  OpType op;                                        // which collective (drives the build call)
  std::shared_ptr<spyre_comms::Context> context;    // the issuing PG's context (v3: worker serves all PGs)
  spyre_comms::BufferDesc buf;                       // prepared on the caller thread (local, cheap)
  std::vector<spyre_comms::BufferDesc> aux_bufs;     // output slots for gather/allgather (C4; caller-prepared)
  CollectiveParams params;                          // op-specific: root, reduceOp(SUM), tag, peer, sizes
  spyre::SpyreStream caller_stream;                  // captured on the caller thread (invariant 3)
  std::chrono::milliseconds op_timeout;              // the issuing backend's PG timeout (for wait_interruptible)
  std::function<bool()> is_aborted;                  // reads the issuing backend's aborted_ (checked before build)
  std::function<void(const std::string&)> on_error;  // issuing backend's report_and_abort (GENUINE errors only)
  std::shared_ptr<WorkState> state;                  // shared with the SpyreCCLWork (§4.2)
};
```

`is_aborted`/`on_error` bind to the **issuing** backend (not a global), so a build
or run failure fails-fast on the correct PG via its existing `report_and_abort`
(idempotent). This is for *genuine errors* only. Clean teardown does NOT go through
`on_error` — see §5.2 (C2): teardown uses a backend-local worker-unblock that never
touches the process-fatal shared-stream shutdown.

`CollectiveParams` is a small tagged struct holding the per-op scalars each
`context->OP(...)` build call needs (e.g. `root` for broadcast/gather, `peer`+`tag`
for send/recv). **C4:** ops whose build needs output-slot descriptors
(`gather`, uniform `allgather`) or a `root` (`broadcast`, `gather`) must have those
descriptors/scalars **captured on the caller thread** (they touch `at::Tensor`s)
and carried in `aux_bufs`/`params`. The worker never touches `at::Tensor` (TLS /
refcount hazards) — it operates only on the pre-extracted `BufferDesc`s. The build
call runs against `request.context` (the issuing PG), not a single backend's
`group_context_`, because the one worker serves every PG.

### 4.2 WorkState — shared lifecycle (new)

```cpp
enum class ProgressState { ENQUEUED, BUILDING, LAUNCHED, DONE_SUCCESS, DONE_ERROR };

struct WorkState {
  std::mutex m;                                  // guards the transition decision (R3/F7)
  std::condition_variable cv;                    // worker notifies terminal; dtor/wait may block on it
  ProgressState state{ProgressState::ENQUEUED};  // guarded by m
  bool cancelled{false};                         // set by ~SpyreCCLWork on a still-ENQUEUED request (R3)
  std::string error_reason;                      // set before state=DONE_ERROR; read after
  std::unique_ptr<spyre_comms::WorkSchedule> ws; // built and owned by the worker; sole driver = worker
};
```

Transitions (worker only, under `m` for the decision points):
`ENQUEUED → BUILDING → LAUNCHED → DONE_SUCCESS`, or `→ DONE_ERROR` on any throw or
error state. `cv.notify_all()` on every terminal transition.

### 4.3 Progress thread (new, process-global)

- **One per process, not per backend** (v3). Lives in process-global state
  alongside `get_comm_stream()` — either in spyre-comms global state or a
  backend-shared singleton owned by torch-spyre. Started on first
  `initialize_library` (or first backend construction), stopped+joined on the last
  `finalize_library` **before** the comm stream/library teardown (§5.2), refcounted
  exactly like the existing `GlobalInfo::incInit/decInit` lifecycle so the Nth
  backend reuses the running worker and only the count-0 finalize tears it down.
- Members (process-global): `std::thread progress_thread_`,
  `std::atomic<bool> progress_stop_`, a `std::mutex queue_mutex_` +
  `std::condition_variable queue_cv_` + `std::deque<ProgressRequest>`.
  Rationale for mutex+deque over the vendored lock-free `concurrentqueue`: enqueue
  rate is one per collective (low), and explicit locking makes the shutdown-drain
  and ordering semantics auditable. Revisit only if profiling shows contention.
- The worker builds/runs against `request.context` and uses `request.op_timeout`
  (both carried per-request, §4.1) — it holds no single backend's members. It reads
  the process-global `comm_stream_` via `get_comm_stream()` (the same singleton every
  backend uses).
- **F1-correct terminal mapping.** `WorkSchedule::wait()` does **not throw** on
  failure — on `needsShutdown()` it sets `DONE_ERROR` and returns normally
  (`work_schedule.cpp:91-106`). Therefore after the drain the worker MUST inspect
  state, exactly as the current synchronous code does (spyre_ccl.cpp:1708,1793):
  ```
  wait_interruptible(ws);   // §4.3 F5
  if (ws->getState() == DONE_ERROR || comm_stream_->needsShutdown()) {
     state->error_reason = ws->getShutdownReason();   // may be empty
     set_state(DONE_ERROR);
  } else {
     set_state(DONE_SUCCESS);
  }
  ```
  Never map blindly to `DONE_SUCCESS`.
- **F5 — interruptible worker wait.** The worker must NOT use the raw
  `ws->wait()` unguarded (no deadline; blocks forever on a dead peer,
  head-of-line-blocking the whole FIFO). `wait_interruptible` mirrors
  `SpyreCCLWork::wait()`'s existing loop (spyre_ccl.cpp:1762-1787): poll
  `ws->query()` with a `needsShutdown()` check, a `request.is_aborted()` check (which
  covers both a genuine `report_and_abort` and the backend-local `local_abort_`
  teardown interrupt, §5.2(a)), and the `request.op_timeout` deadline — so a peer
  failure, a clean subgroup teardown, or a timeout unblocks the worker promptly and
  the FIFO advances. On deadline with no
  completion, fall through to the F1 mapping (which will record `DONE_ERROR`); do
  not wedge the worker.

### 4.4 `SpyreCCLWork` changes

- Holds `std::shared_ptr<WorkState>` instead of owning the `WorkSchedule`.
- `isCompleted()`: if `completed_`, true. Else read `state->state` under `state->m`:
  `DONE_SUCCESS` → `finish_success()` (CAS-guarded); `DONE_ERROR` →
  `finish_error(reason)`; else false. **Never** consults `ws->query()` for the
  terminal decision (invariant 4).
- `wait(timeout)`: keep the existing timeout-precedence logic
  (spyre_ccl.cpp:1728-1817), but block on `state->cv` until `state->state` is
  terminal or the effective deadline elapses (instead of polling `ws->query()`).
  On deadline: `finish_error` + throw, as today. On `DONE_ERROR`: throw with
  `error_reason`. Preserves the c10d contract (throw on failure, false only on
  non-completing timeout).
- `~SpyreCCLWork()` (R3/F6/F7 — the destroy-before/without-drain hazard):
  ```
  std::unique_lock lk(state->m);
  switch (state->state) {
    case ENQUEUED:                     // not yet built; worker will skip it
      state->cancelled = true;         // worker re-checks under m before BUILDING
      return;                          // nothing touches tensors yet — safe, no drain
    case BUILDING: case LAUNCHED:      // worker owns a live schedule that may read tensors
      state->cv.wait(lk, [&]{ return terminal(state->state); });  // block until worker signals
      return;                          // worker was sole driver; no second synchronize()
    case DONE_SUCCESS: case DONE_ERROR:
      return;                          // already drained
  }
  ```
  The worker, symmetrically, under `state->m`: checks `cancelled` before advancing
  `ENQUEUED→BUILDING`; if cancelled, drops the request without building. Holding
  `m` across the cancelled-check + `BUILDING` transition (NOT across the actual
  build/start work) closes the TOCTOU: the dtor either wins (cancels before BUILD)
  or waits for terminal (BUILD already committed). The dtor **never** drives `ws`
  itself — single-driver invariant 5 — so there is no double-`synchronize()` race.
  If a launched schedule is stuck on a dead peer, the dtor's `cv.wait` is unblocked
  when the worker reaches terminal via its interruptible wait — driven by a genuine
  abort (peer failure/`report_and_abort`) or the backend-local teardown interrupt
  (§4.3 F5, §5.2).

### 4.5 Entry-point changes (per in-scope op)

Replace the inline build/order/start/return tail with capture+enqueue:

```cpp
abort_guard("allreduce");
check_vector_tensor(tensors, 1, 1);
if (opts.reduceOp != ReduceOp::SUM) TORCH_CHECK(false, "...SUM only...");   // unchanged
spyre_comms::BufferDesc buf = prepare_buffer_desc(tensors[0]);              // local, cheap
auto state = std::make_shared<WorkState>();
auto caller_stream = spyre::getCurrentStream(tensors[0].device());          // invariant 3: caller thread
ProgressRequest req{
    .op = OpType::ALLREDUCE,
    .context = group_context_,                    // this PG's context (worker builds against it)
    .buf = buf, .aux_bufs = {}, .params = {.reduceOp = SUM},
    .caller_stream = caller_stream,
    .op_timeout = op_timeout_,
    .is_aborted = [this]{ return aborted_.load(std::memory_order_acquire) ||
                                 local_abort_.load(std::memory_order_acquire); },  // §5.2(a) backend-local too
    .on_error   = [this](const std::string& m){ report_and_abort(m); },   // genuine-error path only
    .state = state};
spyre_global_progress_enqueue(std::move(req));    // the ONE process-global queue (§4.3)
seq_.fetch_add(1, std::memory_order_relaxed);                               // Q6: diagnostics only, unchanged
return c10::make_intrusive<SpyreCCLWork>(OpType::ALLREDUCE, state, tensors, tensors, op_timeout_);
```

The `group_context_->allreduce(buf, op)` build call moves into the worker
(§4.3), invoked as `req.context->allreduce(...)`. `order_after_caller_stream` is no
longer called on the caller thread; its host fence (`caller_stream.synchronize()`)
runs on the worker between build and start. The function may be retired or
repurposed as a worker helper; the excluded multi-leg ops keep calling it inline as
today. `prepare_buffer_desc` and the output-slot descriptor extraction (C4) are
confirmed local (no OOB) and stay on the caller thread; if any op's build inputs are
found to touch cross-rank state on the caller thread, that extraction moves fully to
the worker too (verify in implementation). The lambdas capture `this` (the issuing
backend); the `SpyreCCLWork` holds the shared `WorkState`, and the backend outlives
its in-flight Works by the destructor drain (§5.2), so the captured `this` stays
valid for the request's lifetime.

### 4.6 Host-reduce guard (F8, concrete)

The `allreduce` SUM entry already rejects host-only buffers
(`context.cpp:702` throws on `is_host_only`), so a host-only tensor cannot enter.
The residual risk is a *device* buffer whose LIBCOLL `AllReduce::Convert` emits a
`HostComputeOperation` (the D2H→host-sum→H2D UAF path). Guard concretely: **after
BUILD, before `start()`, the worker inspects the built schedule's operation list
and, if it contains any host-compute/host-reduce operation, transitions the
request to `DONE_ERROR` ("host-reduce path excluded in Phase 1 async") instead of
launching.** A located, enforced check in the worker's build→start seam, not a
scope note. (The exact predicate — the op-type enum identifying a host reduce in
`work_schedule`/`operations` — is pinned in the implementation plan.) Fails loud
and safe; never silently runs the UAF path on the worker.

---

## 5. Cross-cutting concerns

### 5.1 Cross-rank ordering
Guaranteed by invariants 1+2 (one **process-global** FIFO, single per-process
issuer thread, BUILD+START both on the worker in enqueue order). No change to the
spyre-comms matcher, OOB FIFO, or HDMA rendezvous — they always see the same
relative order on every rank, across all PGs.

### 5.2 Shutdown / abort (F4 + C2)
Two distinct teardown scenarios, deliberately separated (C2):

**(a) One backend destructs while others live (subgroup teardown, the common case).**
The process-global worker keeps running for the surviving PGs; it must NOT be
stopped, and the shared `comm_stream_` must NOT be shut down (that is process-fatal
across all groups — see [[spyre-subgroup-failure-isolation-boundary]], and the
current dtor at spyre_ccl.cpp:183-192 deliberately avoids it). The destructing
backend must instead:
  1. Mark itself aborted **locally** (`aborted_=true`) so no new requests enqueue.
  2. **Drain its own outstanding requests**: any `WorkState` it still has
     in-flight must reach terminal before the backend object (and its `is_aborted`/
     `on_error` lambdas capturing `this`) is freed. The `SpyreCCLWork` dtor already
     blocks on `state->cv` (§4.4); the backend dtor additionally waits until all its
     issued requests are terminal (tracked via a per-backend in-flight counter
     decremented by the worker on terminal transition). To unblock a request stuck
     on a dead peer **without** the process-fatal shared-stream shutdown, use a
     **backend-local interrupt**: a per-backend `std::atomic<bool> local_abort_`
     that `wait_interruptible` also polls (alongside `needsShutdown()` and the
     deadline). This bounds the drain without poisoning siblings.
  3. Join the per-backend **watchdog** (unchanged, spyre_ccl.cpp:183-192). Do NOT
     touch the process-global worker or `comm_stream_`.

**(b) Last backend / process finalize.** When the refcount reaches zero
(`finalize_library` count-0 path): (1) set the worker's `progress_stop_`,
`queue_cv_.notify_all()`; (2) since no backends remain to serve, it is now safe to
flag `comm_stream_->setShutdown(true)` to unblock any final in-flight wait; (3)
**join the process-global progress thread**; (4) then the existing count-0
`finalize_library` teardown proceeds (it already runs a collective OOB barrier and
tears down the stream). The worker must exit before that teardown (it touches
`comm_stream_`). Queued, never-built requests transition to
`DONE_ERROR("process shutting down")`.

- `abort()`/`shutdown()` (spyre_ccl.cpp:1545-1569) still set `aborted_` (+ now
  `local_abort_`); genuine error aborts still go through `report_and_abort()`
  (idempotent, 1571-1609), which remains the ONLY path that flips the shared
  `comm_stream_->setShutdown` — reserved for real faults, never clean teardown (C2).

### 5.3 Error propagation
Surface unchanged: `Work::wait()` throws on `DONE_ERROR`, `isSuccess()` reflects it,
the Future completes via `finish_error`. The source of the terminal state is now the
worker-set `WorkState` (mapped F1-correctly, §4.3), replacing the inline
`ws->getState()` read. `getShutdownReason()` still supplies the peer-failure reason.

### 5.4 Watchdog interaction
Unchanged per-backend independent thread. On peer failure it calls
`report_and_abort` → sets `aborted_` + `comm_stream_->setShutdown` (this IS a
genuine fault, so the process-fatal shared-stream shutdown is correct here — a peer
failure legitimately dooms the shared transport). The worker's interruptible wait
(§4.3 F5) and the `SpyreCCLWork` dtor's `cv.wait` (§4.4) both observe it and unblock
promptly. Contrast §5.2(a): *clean* subgroup teardown must NOT reach this path — it
uses the backend-local `local_abort_` interrupt instead.

---

## 6. Risks / limitations

- **R1 — worker still host-blocks** on the OOB pre-exchange, `caller_stream.
  synchronize()`, and the drain. That is inherent to 1p5 (§2). The win is that the
  *caller* thread pays none of it. Not removable without new hardware.
- **R2 — bounded overlap only** (DMA-queue vs compute-queue, bandwidth/dispatch
  limited; §2.4). If a workload has little independent compute after its
  collectives, the win is small and the real lever is Phase 2. Stated so
  expectations match hardware.
- **R3 — destroy-before/without-drain**: closed by the `WorkState` mutex + cv
  protocol (§4.4). Single-driver (invariant 5) prevents the double-`synchronize`.
- **R4 — one process-global FIFO serializes all collectives across all PGs**: by
  design (invariant 1). On 1p5 no device concurrency exists to give up — all PGs
  already share the single `comm_stream_` (§2.1, confirmed by the perf analysis and
  test_subgroup.py:284-292). Accepted; it is the correct scope, not a compromise.
- **R5 — per-process single-issuer assumption** (invariant 2): cross-rank ordering
  correctness rests on the process issuing collectives (across all PGs) from one
  thread — standard c10d SPMD. Enqueue is internally locked so the queue can't
  corrupt, but multi-threaded issue would break ordering and is out of scope for
  Phase 1.
- **R8 — backend lifetime vs process-global worker.** The worker outlives
  individual backends and holds requests whose `is_aborted`/`on_error` lambdas
  capture a backend `this`. A backend must not be freed while the worker still holds
  a live request of its own → the backend dtor drains its own in-flight requests to
  terminal first (§5.2(a), per-backend in-flight counter + backend-local interrupt).
  The process-global worker's own lifetime is refcounted to
  `initialize_library`/`finalize_library` (§4.3), so it outlives every backend and
  is torn down only at count-0.
- **R6 — host-reduce UAF**: excluded and *enforced* by the §4.6 worker guard, not
  merely asserted.
- **R7 — latency for a lone collective.** Moving BUILD to the worker adds a
  thread-handoff hop before the OOB exchange even starts. For a single collective
  with nothing to overlap, this is marginally *slower* than synchronous. Accepted:
  Phase 1 optimizes the pipelined/independent-compute case; a lone collective on
  the critical path is no worse than before once `wait()` is called (same total
  work), just with one queue hop of latency.

---

## 7. Testing / verification

- **Unit (host-buildable):** `WorkState` transitions and the `isCompleted`/`wait`
  gating logic are pure enough to test without hardware.
- **HW regression (torchrun, ≥2 ranks):** existing collective gtests
  (`MultipleCalls`, `SendRecv`, `MultipleBarriers`, `UnevenSplitAllReduce`) stay
  green (ordering + reset/reuse). Reuse the installed-binary + device-count-guard
  script pattern (`run_query_state_gate_verify.sh`).
- **Async behavioral proof:** issue an in-scope collective with `async_op=True`,
  assert the call returns before completion (Work not yet complete immediately after
  return), then `wait()` and verify the numeric result — proves the Work no longer
  lies.
- **Ordering stress:** back-to-back distinct collectives across ranks; confirm no
  matcher/OOB/HDMA deadlock (validates invariants 1+2 under the worker-build model).
- **Multi-PG coexistence (C1/C2 — the v3 regression guard):** extend the
  `test_subgroup.py` pattern (world + TP + DP backends in one process, ws=4). Issue
  interleaved async collectives on the world PG and a TP subgroup; confirm no
  matcher/OOB/HDMA deadlock (validates the process-global FIFO serializes cross-PG
  traffic correctly). Then **destroy the TP subgroup backend while the world PG is
  still live and issuing** — confirm the world PG keeps working and its collectives
  still succeed (validates C2: subgroup teardown does not shut down the shared
  `comm_stream_`). This is the single most important new test for v3.
- **Failure paths:** (a) abort mid-flight → queued requests fail to `DONE_ERROR`,
  destructor joins cleanly (no hang) — validates F4/§5.2(b); (b) inject a
  `DONE_ERROR` schedule state → Work throws, no false success — validates F1;
  (c) destroy a Work while its request is ENQUEUED and while LAUNCHED — validates
  R3/§4.4.

Build/deploy is done by the user; verification runs the installed binary via the
established remote pattern. Confirm the exact rebuild command before the first
torch-spyre C++ rebuild.

---

## 8. Out of scope (Phase 2, separate spec)

- In-comms per-stream DMA worker + device-event fences inside `LaunchAll`
  (chunked intra-collective compute/comm overlap).
- Async-enabling the multi-leg decomposition ops (alltoall*, reduce_scatter,
  uneven allgather) and completing their asymmetric/variable (MoE) forms.
- Host-reduce UAF fix (prerequisite for ever putting `reduce`/host-fallback on the
  worker; until then the §4.6 guard excludes it).
- Multi-threaded per-PG issue (invariant 2 / R5).
- Any device-side cross-stream event primitive (absent on 1p5).
