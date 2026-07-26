# Phase-1 Async Collective Dispatch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `SpyreCCLBackend` collectives genuinely asynchronous by moving both schedule *build* and *run* onto a single process-global FIFO progress thread, so a c10d collective returns a real (not-yet-complete) `Work` immediately instead of running the whole build+dispatch synchronously on the caller thread.

**Architecture:** One process-global FIFO worker (refcounted to `initialize_library`/`finalize_library`) is the sole driver of the shared `comm_stream_`, OOB channel, and allocator. Caller threads do only thread-local capture (`prepare_buffer_desc`, `getCurrentStream`, output-slot descriptors) and enqueue a `ProgressRequest`; the worker builds against the issuing PG's `Context`, runs `start()`, and drains via an interruptible wait that returns a discriminated `WaitOutcome`. Completion is tracked in a shared `WorkState` the `SpyreCCLWork` reads — never `ws->query()` blindly.

**Tech Stack:** C++20, torch-spyre (PrivateUse1 c10d backend), spyre-comms (`libspyre_comms.so`), flex runtime. GoogleTest. Build via `uv sync` (torch-spyre) / user-driven remote build for HW.

## Global Constraints

- **License header:** every source file carries the 14-line Apache 2.0 header (C++ `/* */` form). — copy from any existing file.
- **Style:** Google C++ Style; torch-spyre line length 88, spyre-comms 120 (Allman, 4-space). `import regex` never `import re` in any Python.
- **Commits:** `git commit -s` (DCO). **No `Co-Authored-By` trailer** in this workspace.
- **Build ownership:** the USER builds/installs (both `uv sync` for torch-spyre and the remote flex/spyre-comms build). This plan's "run" steps assume the user has built; agent verification runs the installed binary via the established remote script pattern. Confirm the exact rebuild command with the user before the first C++ rebuild of each repo.
- **Scope (in):** `allreduce` (SUM), `broadcast`, `gather`, uniform `allgather`, `send`, `recv`, `barrier`. **Scope (out):** `alltoall`, `alltoall_base`, `reduce_scatter`, uneven `allgather` (stay synchronous, Phase 2); `reduce`/host-reduce path (excluded + guarded).
- **Invariant (cross-rank ordering):** exactly ONE process-global FIFO + one worker thread; the process issues collectives across all PGs from a single thread (standard SPMD). Never add parallelism to the worker.
- **Design source of truth:** `docs/superpowers/specs/2026-07-25-async-progress-thread-design.md` (v3.1). Section refs below (§N) point to it.

---

## File Structure

**spyre-comms (`flex-opensource/spyre-comms/`) — one small additive change:**
- `include/spyre_comms.hpp` — add pure-virtual `bool containsHostReduceOp() const` to abstract `WorkSchedule` (§4.6 guard needs it; abstract base is what torch-spyre holds).
- `src/work_schedule.hpp` / `src/work_schedule.cpp` — override `containsHostReduceOp()` on `SpyreCommsWorkSchedule` (iterate `operations_`, match `OperationKind::{HOST_COMPUTE, HOST_REDUCE_SUM_ALL}`).
- `src/work_schedule_host_reduce_test.cpp` — new unit test for the predicate (host-buildable, no HW).

**torch-spyre (`torch-spyre/torch_spyre/csrc/distributed/`) — the bulk:**
- `progress_worker.hpp` / `progress_worker.cpp` — NEW. Process-global worker: `ProgressState`, `WaitOutcome`, `WorkState`, `ProgressRequest`, the FIFO queue+thread, refcounted lifecycle, `spyre_global_progress_enqueue()`, `spyre_global_progress_start()`/`_stop_if_last()`.
- `spyre_ccl.hpp` / `spyre_ccl.cpp` — MODIFY. `SpyreCCLBackend`: add `local_abort_`, per-backend in-flight counter; rewrite the 7 in-scope entry points to capture+enqueue; rewrite destructor (§5.2). `SpyreCCLWork`: hold `shared_ptr<WorkState>`, cv-based `isCompleted`/`wait`/dtor.
- `tests/distributed/test_async_dispatch.py` — NEW. Async behavioral + multi-PG coexistence + failure-path tests.
- `tests/progress_worker_test.cpp` — NEW (if a C++ gtest target exists in torch-spyre; else fold WorkState logic tests into the python/HW suite). Host-buildable WorkState/WaitOutcome logic.

**Reference sheet (verified signatures) used throughout:**
- Context builds return `std::unique_ptr<spyre_comms::WorkSchedule>`; `allreduce(const BufferDesc&, SpyreReductionOpType)`, `broadcast(const BufferDesc&, process_id_t root)`, `gather(std::vector<BufferDesc>& out, const BufferDesc& in, process_id_t root)`, `allgather(std::vector<BufferDesc>& out, const BufferDesc& in)`, `send(const BufferDesc&, process_id_t, int tag=0)`, `recv(const BufferDesc&, process_id_t, int tag=0)`, `barrier()`.
- `WorkScheduleState::State { IDLE=0, RUNNING=1, DONE_SUCCESS=2, DONE_ERROR=3 }` (spyre_comms.hpp:46-51).
- `WorkSchedule`: `start()`, `wait()`, `bool query() const`, `getState()`, `getShutdownReason()`, `needsShutdown()`, `SetStreamAffinity(flex::RuntimeStream*)`.
- Concrete op-list access: `Size()`, `GetOperation(idx).kind()` → `OperationKind::{HOST_COMPUTE, HOST_REDUCE_SUM_ALL, ...}` (operations.hpp:24-38).
- `spyre::getCurrentStream(c10::Device)` → `SpyreStream` by value (copyable); `.synchronize() const`.
- `get_comm_stream()` → `flex::RuntimeStream*`; refcount `spyre_comms_global.incInit()/decInit()/getInits()`; `initialize_library()`/`finalize_library()`.

---

## Task 1: Host-reduce predicate on WorkSchedule (spyre-comms)

**Files:**
- Modify: `flex-opensource/spyre-comms/include/spyre_comms.hpp` (abstract `WorkSchedule`, near the other pure-virtuals ~line 140-150)
- Modify: `flex-opensource/spyre-comms/src/work_schedule.hpp` (declare override), `flex-opensource/spyre-comms/src/work_schedule.cpp` (define)
- Test: `flex-opensource/spyre-comms/src/work_schedule_host_reduce_test.cpp` (new, `*_test.cpp` auto-discovered)

**Interfaces:**
- Produces: `virtual bool spyre_comms::WorkSchedule::containsHostReduceOp() const` — true iff the built schedule contains any `OperationKind::HOST_COMPUTE` or `HOST_REDUCE_SUM_ALL` op. Torch-spyre's §4.6 guard (Task 8) consumes it through the abstract base pointer.

- [ ] **Step 1: Write the failing test**

```cpp
/* Copyright IBM Corp. 2026 */
#include <gtest/gtest.h>
#include "work_schedule.hpp"
#include "operations.hpp"

using spyre_comms_internal::SpyreCommsWorkSchedule;
using spyre_comms_internal::OperationKind;

// A WorkSchedule with only device ops must report NO host reduce.
TEST(WorkScheduleHostReduce, DeviceOnlyScheduleHasNoHostReduce) {
  SpyreCommsWorkSchedule ws;
  // Append a single DEVICE_COMPUTE op via the existing op-construction path.
  // (Use the same helper the Context build uses; see appendDeviceComputeForTest.)
  ws.appendDeviceComputeForTest();
  EXPECT_FALSE(ws.containsHostReduceOp());
}

// A WorkSchedule containing a HOST_REDUCE_SUM_ALL op must report true.
TEST(WorkScheduleHostReduce, HostReduceScheduleDetected) {
  SpyreCommsWorkSchedule ws;
  ws.appendHostReduceSumAllForTest();
  EXPECT_TRUE(ws.containsHostReduceOp());
}
```

Note: if no `append*ForTest` seam exists, add minimal test-only helpers in `work_schedule.hpp` guarded by nothing special (they only construct the existing Operation subclasses and push to `operations_`). Keep them tiny; they exercise the real `kind()` values.

- [ ] **Step 2: Run test to verify it fails**

Run (local host build): `cmake --preset debug-skip-podman && cmake --build --preset debug-skip-podman --target flex_unit_test && ./build/spyre-comms/tests/spyre_comms_unit_test --gtest_filter=WorkScheduleHostReduce.*`
Expected: FAIL — `containsHostReduceOp` not declared / helpers missing.

- [ ] **Step 3: Add the pure-virtual to the abstract base**

In `include/spyre_comms.hpp`, in `class WorkSchedule` alongside the other pure-virtuals:

```cpp
// Returns true iff this schedule contains a host-side reduce/compute operation
// (OperationKind::HOST_COMPUTE or HOST_REDUCE_SUM_ALL). The async progress
// worker refuses to launch such a schedule (known use-after-free in the
// host-compute path; see torch-spyre async design §4.6). Pure virtual so the
// guard is reachable through the abstract base that consumers hold.
virtual bool containsHostReduceOp() const = 0;
```

- [ ] **Step 4: Implement the override on the concrete class**

In `src/work_schedule.hpp` (declaration, near `Size()`/`GetOperation`):

```cpp
bool containsHostReduceOp() const override;
```

In `src/work_schedule.cpp`:

```cpp
bool SpyreCommsWorkSchedule::containsHostReduceOp() const
{
    for (const auto& op : operations_)
    {
        const OperationKind k = op->kind();
        if (k == OperationKind::HOST_COMPUTE || k == OperationKind::HOST_REDUCE_SUM_ALL)
        {
            return true;
        }
    }
    return false;
}
```

(Iterate the member `operations_` directly — it is in scope in the .cpp. Do NOT go through `GetOperation()`/`Size()` here; direct iteration is simpler and avoids the bounds-checked `.at()` per element.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cmake --build --preset debug-skip-podman --target flex_unit_test && ./build/spyre-comms/tests/spyre_comms_unit_test --gtest_filter=WorkScheduleHostReduce.*`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git -C flex-opensource add spyre-comms/include/spyre_comms.hpp spyre-comms/src/work_schedule.hpp spyre-comms/src/work_schedule.cpp spyre-comms/src/work_schedule_host_reduce_test.cpp
git -C flex-opensource commit -s -m "feat(spyre-comms): add WorkSchedule::containsHostReduceOp() host-reduce predicate"
```

---

## Task 2: WorkState + WaitOutcome + ProgressRequest types (torch-spyre)

**Files:**
- Create: `torch-spyre/torch_spyre/csrc/distributed/progress_worker.hpp`
- Test: `torch-spyre/tests/progress_worker_test.cpp` (if a C++ gtest target exists; otherwise assert the same transitions in the Python HW suite Task 10)

**Interfaces:**
- Produces:
  - `enum class ProgressState { ENQUEUED, BUILDING, LAUNCHED, DONE_SUCCESS, DONE_ERROR };`
  - `enum class WaitOutcome { COMPLETED, SHUTDOWN, TIMED_OUT };`
  - `struct WorkState { std::mutex m; std::condition_variable cv; ProgressState state=ENQUEUED; bool cancelled=false; std::string error_reason; std::unique_ptr<spyre_comms::WorkSchedule> ws; };`
  - `struct ProgressRequest { OpType op; std::shared_ptr<spyre_comms::Context> context; spyre_comms::BufferDesc buf; std::vector<spyre_comms::BufferDesc> aux_bufs; CollectiveParams params; spyre::SpyreStream caller_stream; std::chrono::milliseconds op_timeout; std::function<bool()> is_aborted; std::function<void(const std::string&)> on_error; std::function<void()> on_terminal; std::shared_ptr<WorkState> state; };`
  - `struct CollectiveParams { spyre_comms::SpyreReductionOpType reduce_op = SUM; spyre_comms::process_id_t root = 0; spyre_comms::process_id_t peer = 0; int tag = 0; };`
  - Free helpers (defined in Task 3): `set_terminal(WorkState&, ProgressState, std::string reason)`.
- Consumes: nothing (leaf task).

Rationale for `on_terminal`: it is the per-backend in-flight-counter decrement hook (M2). Binding it per-request keeps the worker free of any backend member. `on_error` is the genuine-fault escalation (`report_and_abort`). `is_aborted` is the pre-launch abort/local-abort check.

- [ ] **Step 1: Write the failing test (WorkState terminal helper + transition semantics)**

```cpp
/* Copyright IBM Corp. 2026 */
#include <gtest/gtest.h>
#include "distributed/progress_worker.hpp"

using torch_spyre::distributed::WorkState;
using torch_spyre::distributed::ProgressState;

TEST(WorkStateTest, SetTerminalNotifiesAndStoresReason) {
  WorkState st;
  bool woke = false;
  std::thread waiter([&]{
    std::unique_lock<std::mutex> lk(st.m);
    st.cv.wait(lk, [&]{ return st.state == ProgressState::DONE_ERROR
                            || st.state == ProgressState::DONE_SUCCESS; });
    woke = true;
  });
  set_terminal(st, ProgressState::DONE_ERROR, "boom");
  waiter.join();
  EXPECT_TRUE(woke);
  EXPECT_EQ(st.state, ProgressState::DONE_ERROR);
  EXPECT_EQ(st.error_reason, "boom");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: build the torch-spyre C++ test target (see repo; if none exists, create a minimal `progress_worker_test` gtest target or move this assertion into Task 10's Python test and skip Steps 1-2 here).
Expected: FAIL — header/`set_terminal` undefined.

- [ ] **Step 3: Write the header (types only; `set_terminal` decl)**

```cpp
/* Copyright IBM Corp. 2026 */
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "spyre_comms.hpp"
#include "spyre_comms_tensor.hpp"
#include "spyre_stream.h"
#include <torch/csrc/distributed/c10d/Types.hpp>  // OpType

namespace torch_spyre::distributed {

enum class ProgressState { ENQUEUED, BUILDING, LAUNCHED, DONE_SUCCESS, DONE_ERROR };
enum class WaitOutcome { COMPLETED, SHUTDOWN, TIMED_OUT };

inline bool is_terminal(ProgressState s) {
  return s == ProgressState::DONE_SUCCESS || s == ProgressState::DONE_ERROR;
}

struct WorkState {
  std::mutex m;
  std::condition_variable cv;
  ProgressState state = ProgressState::ENQUEUED;   // guarded by m
  bool cancelled = false;                          // guarded by m (R3)
  std::string error_reason;                        // set before terminal; read after
  std::unique_ptr<spyre_comms::WorkSchedule> ws;   // built + owned by the worker
};

struct CollectiveParams {
  spyre_comms::SpyreReductionOpType reduce_op = spyre_comms::SpyreReductionOpType::SUM;
  spyre_comms::process_id_t root = 0;
  spyre_comms::process_id_t peer = 0;
  int tag = 0;
};

struct ProgressRequest {
  c10d::OpType op;
  std::shared_ptr<spyre_comms::Context> context;
  spyre_comms::BufferDesc buf;
  std::vector<spyre_comms::BufferDesc> aux_bufs;   // output slots for gather/allgather
  CollectiveParams params;
  spyre::SpyreStream caller_stream;
  std::chrono::milliseconds op_timeout;
  std::function<bool()> is_aborted;                // pre-launch abort/local-abort check (M1)
  std::function<void(const std::string&)> on_error;// genuine-fault escalation (report_and_abort)
  std::function<void()> on_terminal;               // per-backend in-flight decrement (M2)
  std::shared_ptr<WorkState> state;
};

// Transition to a terminal state under the WorkState mutex and notify all waiters.
void set_terminal(WorkState& st, ProgressState terminal, std::string reason);

}  // namespace torch_spyre::distributed
```

- [ ] **Step 4: Implement `set_terminal` (minimal, in progress_worker.cpp — created here, expanded in Task 3)**

```cpp
/* Copyright IBM Corp. 2026 */
#include "distributed/progress_worker.hpp"

namespace torch_spyre::distributed {

void set_terminal(WorkState& st, ProgressState terminal, std::string reason) {
  {
    std::lock_guard<std::mutex> lk(st.m);
    if (!reason.empty()) st.error_reason = std::move(reason);
    st.state = terminal;
  }
  st.cv.notify_all();
}

}  // namespace torch_spyre::distributed
```

- [ ] **Step 5: Run to verify it passes**

Run the C++ test target. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add torch_spyre/csrc/distributed/progress_worker.hpp torch_spyre/csrc/distributed/progress_worker.cpp tests/progress_worker_test.cpp
git commit -s -m "feat(distributed): add WorkState/WaitOutcome/ProgressRequest types for async dispatch"
```

---

## Task 3: Process-global progress worker (queue, thread, interruptible wait, refcounted lifecycle)

**Files:**
- Modify: `torch-spyre/torch_spyre/csrc/distributed/progress_worker.hpp` (declare the worker API)
- Modify: `torch-spyre/torch_spyre/csrc/distributed/progress_worker.cpp` (implement)
- Test: `torch-spyre/tests/progress_worker_test.cpp`

**Interfaces:**
- Consumes: Task 2 types.
- Produces (free functions, process-global):
  - `void spyre_global_progress_ref();` — start the worker on first ref (idempotent, refcounted).
  - `void spyre_global_progress_unref(bool is_last);` — on `is_last`, stop+join the worker.
  - `void spyre_global_progress_enqueue(ProgressRequest req);` — push to the FIFO.
  - `WaitOutcome wait_interruptible(spyre_comms::WorkSchedule& ws, std::chrono::milliseconds timeout);` — poll `query()`/`needsShutdown()`/deadline; returns discriminated outcome. **Does NOT consult any abort flag** (M1: launched waits are never interrupted by local abort).

- [ ] **Step 1: Write the failing test (enqueue → build+run via a fake Context → terminal)**

```cpp
// Uses a test-only fake Context whose allreduce() returns a fake WorkSchedule
// that transitions query()->true after N polls. Verifies the worker:
//  - builds (calls context->OP), runs, and drives the state to DONE_SUCCESS
//  - decrements via on_terminal exactly once
//  - a cancelled request is dropped pre-build (state stays as set, build not called)
TEST(ProgressWorker, RunsEnqueuedRequestToSuccess) {
  spyre_global_progress_ref();
  auto st = std::make_shared<WorkState>();
  std::atomic<int> terminals{0};
  auto ctx = std::make_shared<FakeContext>();       // build returns FakeWorkSchedule
  ProgressRequest req{
      .op = c10d::OpType::ALLREDUCE, .context = ctx,
      .op_timeout = std::chrono::seconds(5),
      .is_aborted = []{ return false; },
      .on_error = [](const std::string&){},
      .on_terminal = [&]{ terminals.fetch_add(1); },
      .state = st};
  spyre_global_progress_enqueue(std::move(req));
  { std::unique_lock<std::mutex> lk(st->m);
    st->cv.wait(lk, [&]{ return is_terminal(st->state); }); }
  EXPECT_EQ(st->state, ProgressState::DONE_SUCCESS);
  EXPECT_EQ(terminals.load(), 1);
  spyre_global_progress_unref(true);
}

TEST(ProgressWorker, CancelledRequestDroppedPreBuild) {
  spyre_global_progress_ref();
  auto ctx = std::make_shared<FakeContext>();
  auto st = std::make_shared<WorkState>();
  { std::lock_guard<std::mutex> lk(st->m); st->cancelled = true; }
  ProgressRequest req{ .op = c10d::OpType::ALLREDUCE, .context = ctx,
      .op_timeout = std::chrono::seconds(5), .is_aborted = []{ return false; },
      .on_error = [](const std::string&){}, .on_terminal = []{}, .state = st };
  spyre_global_progress_enqueue(std::move(req));
  // Give the worker a moment; cancelled must mean build never runs.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(ctx->build_calls.load(), 0);
  spyre_global_progress_unref(true);
}
```

(Provide `FakeContext`/`FakeWorkSchedule` as test doubles subclassing the abstract `spyre_comms::Context`/`WorkSchedule`. `FakeWorkSchedule::containsHostReduceOp()` returns false; `query()` returns true after a counter; `getState()` returns `DONE_SUCCESS`.)

- [ ] **Step 2: Run to verify it fails**

Run the C++ test target. Expected: FAIL — worker functions undefined.

- [ ] **Step 3: Declare the worker API in progress_worker.hpp**

```cpp
void spyre_global_progress_ref();
void spyre_global_progress_unref(bool is_last);
void spyre_global_progress_enqueue(ProgressRequest req);
WaitOutcome wait_interruptible(spyre_comms::WorkSchedule& ws,
                               std::chrono::milliseconds timeout);
```

- [ ] **Step 4: Implement the worker in progress_worker.cpp**

```cpp
#include <deque>
#include <thread>
#include "spyre_comms.hpp"                     // get_comm_stream, WorkSchedule
#include "distributed/progress_worker.hpp"

namespace torch_spyre::distributed {
namespace {

struct GlobalProgress {
  std::mutex qm;
  std::condition_variable qcv;
  std::deque<ProgressRequest> queue;
  std::thread thread;
  bool stop = false;
  int refcount = 0;         // guarded by qm
};

GlobalProgress& gp() { static GlobalProgress g; return g; }

void run_one(ProgressRequest& req) {
  // Pre-launch abort / cancel checks (M1: this is the ONLY place local abort applies).
  {
    std::lock_guard<std::mutex> lk(req.state->m);
    if (req.state->cancelled) { req.on_terminal(); return; }  // dtor cancelled it; nothing built
  }
  if (req.is_aborted()) {
    set_terminal(*req.state, ProgressState::DONE_ERROR, "backend aborted before launch");
    req.on_terminal();
    return;
  }
  try {
    // BUILD (blocking OOB pre-exchange happens here, on THIS worker thread).
    std::unique_ptr<spyre_comms::WorkSchedule> ws = build_for(req);   // Task 4 dispatch
    // §4.6 host-reduce guard: refuse to launch a host-compute schedule.
    if (ws->containsHostReduceOp()) {
      set_terminal(*req.state, ProgressState::DONE_ERROR,
                   "host-reduce path excluded in Phase 1 async");
      req.on_terminal();
      return;
    }
    // Producer -> collective ordering fence, on the worker (was caller-thread before).
    req.caller_stream.synchronize();
    ws->SetStreamAffinity(spyre_comms::get_comm_stream());
    // Publish ws + LAUNCHED under m, re-checking cancel (TOCTOU close, §4.4).
    {
      std::lock_guard<std::mutex> lk(req.state->m);
      if (req.state->cancelled) { req.on_terminal(); return; }
      req.state->ws = std::move(ws);
      req.state->state = ProgressState::LAUNCHED;
    }
    req.state->ws->start();                              // LaunchAll: HDMA/CBS rendezvous
    switch (wait_interruptible(*req.state->ws, req.op_timeout)) {
      case WaitOutcome::COMPLETED: {
        const bool err = req.state->ws->getState() ==
                         spyre_comms::WorkScheduleState::State::DONE_ERROR;
        set_terminal(*req.state,
                     err ? ProgressState::DONE_ERROR : ProgressState::DONE_SUCCESS,
                     err ? req.state->ws->getShutdownReason() : std::string());
        break;
      }
      case WaitOutcome::SHUTDOWN:
        set_terminal(*req.state, ProgressState::DONE_ERROR,
                     req.state->ws->getShutdownReason());
        break;
      case WaitOutcome::TIMED_OUT:
        req.on_error("collective timed out after " +
                     std::to_string(req.op_timeout.count()) + " ms");   // -> report_and_abort
        set_terminal(*req.state, ProgressState::DONE_ERROR, "timed out");
        break;
    }
  } catch (const std::exception& e) {
    set_terminal(*req.state, ProgressState::DONE_ERROR, e.what());
    req.on_error(std::string("async collective: ") + e.what());
  } catch (...) {
    set_terminal(*req.state, ProgressState::DONE_ERROR, "unknown error");
    req.on_error("async collective: unknown error");
  }
  req.on_terminal();   // M2: LAST action, after all lambda use; release happens in the counter.
}

void worker_loop() {
  for (;;) {
    ProgressRequest req;
    {
      std::unique_lock<std::mutex> lk(gp().qm);
      gp().qcv.wait(lk, []{ return gp().stop || !gp().queue.empty(); });
      if (gp().stop && gp().queue.empty()) return;
      req = std::move(gp().queue.front());
      gp().queue.pop_front();
    }
    run_one(req);
  }
}

}  // namespace

WaitOutcome wait_interruptible(spyre_comms::WorkSchedule& ws,
                               std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    if (ws.query()) return WaitOutcome::COMPLETED;
    if (ws.needsShutdown()) return WaitOutcome::SHUTDOWN;
    if (timeout.count() > 0 && std::chrono::steady_clock::now() >= deadline)
      return WaitOutcome::TIMED_OUT;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void spyre_global_progress_ref() {
  std::lock_guard<std::mutex> lk(gp().qm);
  if (gp().refcount++ == 0) {
    gp().stop = false;
    gp().thread = std::thread(worker_loop);
  }
}

void spyre_global_progress_unref(bool is_last) {
  std::thread to_join;
  {
    std::lock_guard<std::mutex> lk(gp().qm);
    if (--gp().refcount > 0 && !is_last) return;
    gp().stop = true;
    to_join = std::move(gp().thread);
  }
  gp().qcv.notify_all();
  if (to_join.joinable()) to_join.join();
}

void spyre_global_progress_enqueue(ProgressRequest req) {
  {
    std::lock_guard<std::mutex> lk(gp().qm);
    gp().queue.push_back(std::move(req));
  }
  gp().qcv.notify_one();
}

}  // namespace torch_spyre::distributed
```

Note: `build_for(req)` is the per-op dispatch, added in Task 4. For Steps here, stub `build_for` to call `req.context->allreduce(req.buf, req.params.reduce_op)` so the ALLREDUCE test passes; Task 4 replaces the stub with the full switch and its own tests.

- [ ] **Step 5: Run to verify it passes**

Run the C++ test target `--gtest_filter=ProgressWorker.*`. Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add torch_spyre/csrc/distributed/progress_worker.hpp torch_spyre/csrc/distributed/progress_worker.cpp tests/progress_worker_test.cpp
git commit -s -m "feat(distributed): process-global FIFO progress worker + interruptible wait"
```

---

## Task 4: Per-op build dispatch (`build_for`) — the 7 in-scope ops

**Files:**
- Modify: `torch-spyre/torch_spyre/csrc/distributed/progress_worker.cpp` (add `build_for`)
- Test: `torch-spyre/tests/progress_worker_test.cpp` (extend FakeContext to record which op was built)

**Interfaces:**
- Consumes: `ProgressRequest` (Task 2), Context build signatures (reference sheet).
- Produces: `std::unique_ptr<spyre_comms::WorkSchedule> build_for(ProgressRequest& req);` — dispatches on `req.op` to the correct `context->OP(...)` call, using `req.buf`/`req.aux_bufs`/`req.params`. **No `at::Tensor` access** (C4: all tensor-derived data is already in `buf`/`aux_bufs`/`params`).

- [ ] **Step 1: Write the failing test**

```cpp
TEST(BuildDispatch, EachInScopeOpCallsMatchingContextMethod) {
  auto ctx = std::make_shared<FakeContext>();
  for (auto op : {c10d::OpType::ALLREDUCE, c10d::OpType::BROADCAST,
                  c10d::OpType::GATHER, c10d::OpType::ALLGATHER,
                  c10d::OpType::SEND, c10d::OpType::RECV, c10d::OpType::BARRIER}) {
    ProgressRequest req{ .op = op, .context = ctx, .op_timeout = {},
        .is_aborted=[]{return false;}, .on_error=[](auto&){}, .on_terminal=[]{},
        .state = std::make_shared<WorkState>() };
    auto ws = build_for(req);
    EXPECT_NE(ws, nullptr);
  }
  EXPECT_EQ(ctx->last_built, c10d::OpType::BARRIER);
  EXPECT_EQ(ctx->build_calls.load(), 7);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `build_for` not defined (or the stub only handles ALLREDUCE).

- [ ] **Step 3: Implement `build_for`**

```cpp
std::unique_ptr<spyre_comms::WorkSchedule> build_for(ProgressRequest& req) {
  auto& ctx = *req.context;
  switch (req.op) {
    case c10d::OpType::ALLREDUCE:
      return ctx.allreduce(req.buf, req.params.reduce_op);
    case c10d::OpType::BROADCAST:
      return ctx.broadcast(req.buf, req.params.root);
    case c10d::OpType::GATHER:
      return ctx.gather(req.aux_bufs, req.buf, req.params.root);
    case c10d::OpType::ALLGATHER:
      return ctx.allgather(req.aux_bufs, req.buf);
    case c10d::OpType::SEND:
      return ctx.send(req.buf, req.params.peer, req.params.tag);
    case c10d::OpType::RECV:
      return ctx.recv(req.buf, req.params.peer, req.params.tag);
    case c10d::OpType::BARRIER:
      return ctx.barrier();
    default:
      throw std::runtime_error("async build_for: op not in Phase-1 scope");
  }
}
```

(Declare `build_for` above `run_one` in the anonymous namespace; move the Task 3 stub's body here.)

- [ ] **Step 4: Run to verify it passes**

Run `--gtest_filter=BuildDispatch.*` and re-run `ProgressWorker.*`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torch_spyre/csrc/distributed/progress_worker.cpp tests/progress_worker_test.cpp
git commit -s -m "feat(distributed): per-op build dispatch for the 7 in-scope collectives"
```

---

## Task 5: SpyreCCLWork — shared WorkState, cv-based isCompleted/wait/dtor

**Files:**
- Modify: `torch-spyre/torch_spyre/csrc/distributed/spyre_ccl.hpp` (SpyreCCLWork members + ctor, lines ~300-349)
- Modify: `torch-spyre/torch_spyre/csrc/distributed/spyre_ccl.cpp` (ctor ~1650, dtor ~1663, isCompleted ~1697, isSuccess ~1722, wait ~1728)
- Test: covered by Task 10 HW behavioral tests (async return + wait) and the destroy-before/after-launch cases.

**Interfaces:**
- Consumes: `WorkState` (Task 2). `finish_success()`/`finish_error()` (existing, unchanged).
- Produces: `SpyreCCLWork(OpType, std::shared_ptr<WorkState> state, std::vector<at::Tensor> hold, std::vector<at::Tensor> result, std::chrono::milliseconds default_timeout)`.

- [ ] **Step 1: Replace the `work_schedule_` member with the shared state**

In `spyre_ccl.hpp` SpyreCCLWork private section, replace `std::unique_ptr<spyre_comms::WorkSchedule> work_schedule_;` (line 342) with:

```cpp
std::shared_ptr<torch_spyre::distributed::WorkState> state_;
```

Update the ctor decl (line 321) to take `std::shared_ptr<WorkState> state` instead of `std::unique_ptr<WorkSchedule> work_schedule`. Add `#include "distributed/progress_worker.hpp"`.

- [ ] **Step 2: Update the ctor**

In spyre_ccl.cpp (~1650): initialize `state_(std::move(state))`, drop the `work_schedule_(...)` init. Keep `future_`, `hold_tensors_`, `result_tensors_`, `default_timeout_`.

- [ ] **Step 3: Rewrite isCompleted() (gate on WorkState, never ws->query())**

```cpp
bool SpyreCCLWork::isCompleted() {
  if (completed_.load(std::memory_order_acquire)) return true;
  ProgressState s;
  std::string reason;
  { std::lock_guard<std::mutex> lk(state_->m);
    s = state_->state; reason = state_->error_reason; }
  if (!is_terminal(s)) return false;
  bool expected = false;
  if (completed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    if (s == ProgressState::DONE_ERROR)
      finish_error(reason.empty() ? "[SpyreCCL]: collective completed with DONE_ERROR"
                                  : "[SpyreCCL]: collective aborted: " + reason);
    else finish_success();
  }
  return true;
}
```

- [ ] **Step 4: Rewrite isSuccess() and wait()**

`isSuccess()`:
```cpp
bool SpyreCCLWork::isSuccess() const {
  std::lock_guard<std::mutex> lk(state_->m);
  return state_->state != ProgressState::DONE_ERROR;
}
```

`wait(timeout)`: keep the timeout-precedence preamble (explicit positive per-call wins, else `default_timeout_`, else block). Replace the `ws->query()` poll loop with a cv wait on `state_`:
```cpp
bool SpyreCCLWork::wait(std::chrono::milliseconds timeout) {
  const std::chrono::milliseconds eff =
      (timeout != kUnsetTimeout && timeout.count() > 0) ? timeout : default_timeout_;
  std::unique_lock<std::mutex> lk(state_->m);
  if (!is_terminal(state_->state)) {
    if (eff == kUnsetTimeout || eff.count() <= 0) {
      state_->cv.wait(lk, [&]{ return is_terminal(state_->state); });
    } else if (!state_->cv.wait_for(lk, eff, [&]{ return is_terminal(state_->state); })) {
      // timed out without completing
      lk.unlock();
      bool expected = false;
      if (completed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        finish_error("[SpyreCCL]: collective wait timed out after " +
                     std::to_string(eff.count()) + " ms");
      throw std::runtime_error("[SpyreCCL]: collective wait timed out after " +
                               std::to_string(eff.count()) + " ms");
    }
  }
  const bool err = state_->state == ProgressState::DONE_ERROR;
  std::string reason = state_->error_reason;
  lk.unlock();
  bool expected = false;
  if (completed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    if (err) finish_error(reason.empty() ? "[SpyreCCL]: collective completed with an error"
                                         : "[SpyreCCL]: collective aborted: " + reason);
    else finish_success();
  }
  if (err) throw std::runtime_error(reason.empty()
        ? "[SpyreCCL]: collective completed with an error"
        : "[SpyreCCL]: collective aborted: " + reason);
  return true;
}
```

- [ ] **Step 5: Rewrite the destructor (§4.4 R3/F6/F7 protocol)**

```cpp
SpyreCCLWork::~SpyreCCLWork() {
  std::unique_lock<std::mutex> lk(state_->m);
  switch (state_->state) {
    case ProgressState::ENQUEUED:
    case ProgressState::BUILDING:
      // If still pre-launch we can cancel; but BUILDING means the worker may be
      // mid-build. Safest uniform rule: mark cancelled, then wait for terminal.
      state_->cancelled = true;
      [[fallthrough]];
    case ProgressState::LAUNCHED:
      state_->cv.wait(lk, [&]{ return is_terminal(state_->state); });
      break;
    case ProgressState::DONE_SUCCESS:
    case ProgressState::DONE_ERROR:
      break;
  }
}
```

Note: for ENQUEUED, `cancelled=true` lets the worker drop it without building → it will reach terminal (or the worker sets terminal on drop). We still wait for terminal so `hold_tensors_` outlives any build/DMA. This is the single-driver rule: the dtor never touches `ws`.

- [ ] **Step 6: Build (user) + run the Task 10 destroy-path tests**

Deferred to Task 10 (needs HW). For now: `uv sync --all-extras --active` must compile cleanly. Expected: compiles.

- [ ] **Step 7: Commit**

```bash
git add torch_spyre/csrc/distributed/spyre_ccl.hpp torch_spyre/csrc/distributed/spyre_ccl.cpp
git commit -s -m "feat(distributed): SpyreCCLWork gates on shared WorkState with cv-based wait/dtor"
```

---

## Task 6: SpyreCCLBackend — new members + refcounted worker lifecycle

**Files:**
- Modify: `torch-spyre/torch_spyre/csrc/distributed/spyre_ccl.hpp` (add members ~line 216-249)
- Modify: `torch-spyre/torch_spyre/csrc/distributed/spyre_ccl.cpp` (ctor ~172-181, dtor ~183-192)

**Interfaces:**
- Consumes: `spyre_global_progress_ref/unref` (Task 3).
- Produces: `std::atomic<bool> local_abort_{false}` and `std::atomic<int> inflight_{0}` on SpyreCCLBackend; worker started via `ref` in ctor, `unref` in dtor.

- [ ] **Step 1: Add members to SpyreCCLBackend (spyre_ccl.hpp)**

```cpp
// Backend-local teardown interrupt: gates ONLY pre-launch request drops (M1).
// wait_interruptible never consults it — a launched DMA on the shared stream is
// never abandoned. Set in the destructor / abort().
std::atomic<bool> local_abort_{false};
// Count of this backend's requests the worker has not yet driven to terminal.
// Decremented (release) by the worker's on_terminal hook; the destructor waits
// (acquire) for it to reach 0 before freeing this backend (M2/R8).
std::atomic<int> inflight_{0};
std::mutex inflight_mu_;
std::condition_variable inflight_cv_;
```

- [ ] **Step 2: Start the worker (refcounted) in the ctor**

In spyre_ccl.cpp ctor, after `comm_stream_ = spyre_comms::get_comm_stream();` (line 174) and before the watchdog start:

```cpp
// Start (or ref) the one process-global progress worker. Refcounted so the Nth
// backend reuses the running worker; only the last unref (dtor) joins it.
torch_spyre::distributed::spyre_global_progress_ref();
```

- [ ] **Step 3: Update the destructor (§5.2(a): drain own in-flight, backend-local, no shared-stream shutdown)**

Replace the current dtor body (spyre_ccl.cpp:183-192) with:

```cpp
SpyreCCLBackend::~SpyreCCLBackend() {
  // (§5.2a) Stop new enqueues + let the worker drop this backend's un-launched
  // requests. Backend-local: does NOT shut down the shared comm_stream_.
  aborted_.store(true, std::memory_order_release);
  local_abort_.store(true, std::memory_order_release);
  // Drain: wait until every request this backend issued has reached terminal, so
  // the worker is no longer calling into our on_error/on_terminal/is_aborted
  // lambdas (which capture `this`) and no DMA is still reading held tensors.
  {
    std::unique_lock<std::mutex> lk(inflight_mu_);
    inflight_cv_.wait(lk, [&]{ return inflight_.load(std::memory_order_acquire) == 0; });
  }
  // Stop + join the watchdog (unchanged).
  watchdog_stop_.store(true, std::memory_order_release);
  if (watchdog_thread_.joinable()) watchdog_thread_.join();
  // Unref the process-global worker; is_last is true only on the count-0
  // finalize path. finalize_library() itself is refcounted; the worker join
  // must precede the count-0 comm-stream teardown.
  const bool is_last = (spyre_comms_global.getInits() == 1);
  torch_spyre::distributed::spyre_global_progress_unref(is_last);
  spyre_comms::finalize_library();
}
```

Note on `is_last`: `getInits()` returns the current count; `finalize_library()` (called next) is what actually decrements. Reading `==1` here means "this finalize will hit zero" → last backend → join the worker. Verify this matches `finalize_library`'s gate (`getInits() <= 0` after decInit) during implementation; if off-by-one, gate `unref(is_last)` on the value `finalize_library` will act on. See Task 9 for the exact reconciliation + test.

- [ ] **Step 4: Build (user) — compiles cleanly**

`uv sync --all-extras --active`. Expected: compiles. (Behavioral drain tested in Task 10.)

- [ ] **Step 5: Commit**

```bash
git add torch_spyre/csrc/distributed/spyre_ccl.hpp torch_spyre/csrc/distributed/spyre_ccl.cpp
git commit -s -m "feat(distributed): backend-local abort + inflight drain + refcounted worker lifecycle"
```

---

## Task 7: Rewrite the 7 in-scope entry points to capture + enqueue

**Files:**
- Modify: `torch-spyre/torch_spyre/csrc/distributed/spyre_ccl.cpp` — `allreduce` (~796), `broadcast` (~1192), `gather` (~1220), `allgather` uniform fast path (~692-711), `send` (~1459), `recv` (~1489), `barrier` (~1171).

**Interfaces:**
- Consumes: `ProgressRequest`, `spyre_global_progress_enqueue`, `prepare_buffer_desc` (existing), `spyre::getCurrentStream` (existing pattern), `SpyreCCLWork` (Task 5 ctor).
- Produces: each entry point returns `c10::make_intrusive<SpyreCCLWork>(op, state, hold, result, op_timeout_)` after enqueuing.

**Shared helper (add once, private):**

```cpp
// Build a request wired to this backend and enqueue it. Increments inflight_
// BEFORE enqueue; the worker's on_terminal decrements (release) and notifies.
c10::intrusive_ptr<Work> SpyreCCLBackend::enqueue_async(
    OpType op, const spyre_comms::BufferDesc& buf,
    std::vector<spyre_comms::BufferDesc> aux_bufs,
    torch_spyre::distributed::CollectiveParams params,
    const spyre::SpyreStream& caller_stream,
    std::vector<at::Tensor> hold, std::vector<at::Tensor> result) {
  auto state = std::make_shared<torch_spyre::distributed::WorkState>();
  inflight_.fetch_add(1, std::memory_order_acq_rel);
  torch_spyre::distributed::ProgressRequest req{
      .op = op, .context = group_context_, .buf = buf,
      .aux_bufs = std::move(aux_bufs), .params = params,
      .caller_stream = caller_stream, .op_timeout = op_timeout_,
      .is_aborted = [this]{ return aborted_.load(std::memory_order_acquire)
                                || local_abort_.load(std::memory_order_acquire); },
      .on_error   = [this](const std::string& m){ report_and_abort(m); },
      .on_terminal= [this]{
          inflight_.fetch_sub(1, std::memory_order_release);
          std::lock_guard<std::mutex> lk(inflight_mu_);
          inflight_cv_.notify_all(); },
      .state = state};
  torch_spyre::distributed::spyre_global_progress_enqueue(std::move(req));
  seq_.fetch_add(1, std::memory_order_relaxed);
  return c10::make_intrusive<SpyreCCLWork>(op, state, std::move(hold),
                                           std::move(result), op_timeout_);
}
```

Declare `enqueue_async` in spyre_ccl.hpp private section.

- [ ] **Step 1: Rewrite allreduce (the template for all 7)**

Replace the body after the SUM check (spyre_ccl.cpp:809-821) with:

```cpp
spyre_comms::BufferDesc buf = prepare_buffer_desc(tensors[0]);
auto caller_stream = spyre::getCurrentStream(tensors[0].device());
return enqueue_async(OpType::ALLREDUCE, buf, /*aux=*/{},
                     {.reduce_op = convert_reduce_op_type(opts.reduceOp)},
                     caller_stream, /*hold=*/tensors, /*result=*/tensors);
```

Keep the `abort_guard`, `check_vector_tensor`, and SUM `TORCH_CHECK` preamble. Keep the surrounding try/catch (build no longer throws here, but validation can).

- [ ] **Step 2: Rewrite broadcast, send, recv, barrier (single-buffer / scalar-param ops)**

- broadcast: `params = {.root = static_cast<process_id_t>(opts.rootRank)}`; hold=result=tensors.
- send: `params = {.peer = static_cast<process_id_t>(dstRank), .tag = tag}`; hold=result=tensors.
- recv: same as send with `srcRank`.
- barrier: `buf` = empty/default `BufferDesc`; no tensors; `enqueue_async(OpType::BARRIER, {}, {}, {}, getCurrentStream(defaultDevice), {}, {})`. (Barrier has no tensor; capture the current stream on the default spyre device — confirm the device handle used elsewhere in the current barrier at spyre_ccl.cpp:1171.)

- [ ] **Step 3: Rewrite gather and uniform allgather (output-slot descriptors — C4)**

These need `aux_bufs` = one `BufferDesc` per output slot, **extracted on the caller thread**:

```cpp
// gather (root collects); allgather (everyone collects). Build output descs now.
std::vector<spyre_comms::BufferDesc> out_descs;
out_descs.reserve(output_tensors.size());
for (auto& t : output_tensors) out_descs.push_back(prepare_buffer_desc(t));
spyre_comms::BufferDesc in = prepare_buffer_desc(input_tensor);
auto caller_stream = spyre::getCurrentStream(input_tensor.device());
// hold = input + all outputs (all are touched by the DMA); result = outputs.
std::vector<at::Tensor> hold = output_tensors; hold.push_back(input_tensor);
return enqueue_async(op, in, std::move(out_descs),
                     {.root = static_cast<process_id_t>(rootRank_or_0)},
                     caller_stream, std::move(hold), output_tensors);
```

For uniform allgather use the existing fast-path branch only (spyre_ccl.cpp:692-711); the uneven branch (713-776) stays synchronous — do NOT route it through the worker. Guard: if the allgather is the uneven variant, fall through to the existing synchronous code unchanged.

- [ ] **Step 4: Verify no `at::Tensor` crosses to the worker (C4 audit)**

Read each rewritten entry point. Confirm every value placed in the `ProgressRequest` (`buf`, `aux_bufs`, `params`, `caller_stream`) is a plain value/`BufferDesc`/scalar — no `at::Tensor`, no tensor accessor, captured by the request. The `hold`/`result` tensors live on the `SpyreCCLWork` (main thread), never dereferenced by the worker. Document this audit in the commit message.

- [ ] **Step 5: Build (user) — compiles; then Task 10 exercises behavior**

`uv sync --all-extras --active`. Expected: compiles.

- [ ] **Step 6: Commit**

```bash
git add torch_spyre/csrc/distributed/spyre_ccl.hpp torch_spyre/csrc/distributed/spyre_ccl.cpp
git commit -s -m "feat(distributed): route 7 in-scope collectives through async enqueue (C4-audited: no at::Tensor on worker)"
```

---

## Task 8: Wire the §4.6 host-reduce guard end-to-end + confirm exclusions

**Files:**
- Modify: `torch-spyre/torch_spyre/csrc/distributed/progress_worker.cpp` (guard already added in Task 3 Step 4 — this task adds its test + confirms the excluded ops are untouched)
- Test: `torch-spyre/tests/progress_worker_test.cpp`

**Interfaces:**
- Consumes: `WorkSchedule::containsHostReduceOp()` (Task 1), the worker guard (Task 3).

- [ ] **Step 1: Write the failing test (a host-reduce schedule is refused, not launched)**

```cpp
TEST(ProgressWorker, HostReduceScheduleRefusedNotLaunched) {
  spyre_global_progress_ref();
  auto ctx = std::make_shared<FakeContext>();
  ctx->next_ws_has_host_reduce = true;    // FakeWorkSchedule::containsHostReduceOp()->true
  auto st = std::make_shared<WorkState>();
  ProgressRequest req{ .op = c10d::OpType::ALLREDUCE, .context = ctx,
      .op_timeout = std::chrono::seconds(5), .is_aborted=[]{return false;},
      .on_error=[](auto&){}, .on_terminal=[]{}, .state = st };
  spyre_global_progress_enqueue(std::move(req));
  { std::unique_lock<std::mutex> lk(st->m);
    st->cv.wait(lk, [&]{ return is_terminal(st->state); }); }
  EXPECT_EQ(st->state, ProgressState::DONE_ERROR);
  EXPECT_NE(st->error_reason.find("host-reduce"), std::string::npos);
  EXPECT_FALSE(ctx->last_ws_started);      // start() never called
  spyre_global_progress_unref(true);
}
```

- [ ] **Step 2: Run to verify it fails, then passes**

The guard code exists (Task 3 Step 4); extend `FakeContext`/`FakeWorkSchedule` with `next_ws_has_host_reduce` + a `started` flag so the test can observe `start()` was skipped. Run `--gtest_filter=ProgressWorker.HostReduceScheduleRefusedNotLaunched`. Expected: PASS.

- [ ] **Step 3: Confirm the excluded multi-leg ops are NOT routed through the worker**

Grep the 4 excluded entry points and assert they still call the inline synchronous `start()+wait()` path unchanged:

Run: `grep -n "enqueue_async\|start()\|->wait()" torch_spyre/csrc/distributed/spyre_ccl.cpp | sed -n '/alltoall\|reduce_scatter/p'`
Expected: `alltoall` (848), `alltoall_base` (934), `reduce_scatter` (1328), uneven allgather (713-776) contain NO `enqueue_async` and still use inline `start()`/`wait()`. Document in commit.

- [ ] **Step 4: Commit**

```bash
git add torch_spyre/csrc/distributed/progress_worker.cpp tests/progress_worker_test.cpp
git commit -s -m "test(distributed): host-reduce guard refuses launch; confirm excluded ops stay synchronous"
```

---

## Task 9: Reconcile worker lifecycle with `finalize_library` refcount (§5.2(b))

**Files:**
- Modify: `torch-spyre/torch_spyre/csrc/distributed/spyre_ccl.cpp` (dtor `is_last` reconciliation from Task 6 Step 3)
- Test: `torch-spyre/tests/progress_worker_test.cpp` (ref/unref counting)

**Interfaces:**
- Consumes: `spyre_comms_global.getInits()`, `spyre_global_progress_ref/unref`.

- [ ] **Step 1: Write the failing test (ref/unref balance; last unref joins)**

```cpp
TEST(ProgressWorker, RefcountStartsOnceJoinsOnLast) {
  // 3 refs, 3 unrefs; worker thread alive across the middle, joined only at last.
  spyre_global_progress_ref();
  spyre_global_progress_ref();
  spyre_global_progress_ref();
  EXPECT_TRUE(spyre_global_progress_is_running());   // add a test-only accessor
  spyre_global_progress_unref(false);
  spyre_global_progress_unref(false);
  EXPECT_TRUE(spyre_global_progress_is_running());
  spyre_global_progress_unref(true);
  EXPECT_FALSE(spyre_global_progress_is_running());
}
```

Add `bool spyre_global_progress_is_running();` (reads `gp().thread.joinable()` under `qm`), test-only but harmless.

- [ ] **Step 2: Run to verify it fails, implement the accessor, verify it passes**

Implement `spyre_global_progress_is_running()`. Run `--gtest_filter=ProgressWorker.RefcountStartsOnceJoinsOnLast`. Expected: PASS.

- [ ] **Step 3: Reconcile `is_last` in the backend dtor**

Confirm against `spyre_comms.cpp:283-322`: `finalize_library()` runs `decInit()` and does the count-0 teardown when `getInits() <= 0` afterward. In the dtor (Task 6 Step 3), the worker `unref` must join **before** `finalize_library()`'s count-0 comm-stream teardown, and only on the truly-last backend. Since `spyre_global_progress_ref` is called once per backend ctor, the progress refcount tracks backend count directly — so pass `is_last` = `(progress refcount will reach 0)`, i.e. let `spyre_global_progress_unref` decide internally and drop the `is_last` gate on `getInits()`:

Change Task 6's dtor to:
```cpp
  torch_spyre::distributed::spyre_global_progress_unref(/*is_last unused*/ false);
  spyre_comms::finalize_library();
```
and make `spyre_global_progress_unref` join whenever its own refcount hits 0 (it already does: `if (--refcount > 0) return;`). Remove the `is_last` parameter entirely (simplify the signature to `spyre_global_progress_unref()`), since the worker's own refcount is authoritative and 1:1 with backends. Update Task 3/Task 6 call sites.

Rationale: two independent refcounts (progress worker vs library) that must agree is a bug risk. The progress worker's own count is sufficient and self-consistent.

- [ ] **Step 4: Update signature + all call sites, rebuild, re-run ref/unref test**

`spyre_global_progress_unref()` (no arg). Grep for call sites, update. Run the refcount test (adjust to no-arg). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torch_spyre/csrc/distributed/progress_worker.hpp torch_spyre/csrc/distributed/progress_worker.cpp torch_spyre/csrc/distributed/spyre_ccl.cpp tests/progress_worker_test.cpp
git commit -s -m "fix(distributed): worker lifecycle keyed on its own refcount (1:1 with backends), joined on last unref"
```

---

## Task 10: HW behavioral + regression tests (torchrun, ≥2 ranks)

**Files:**
- Create: `torch-spyre/tests/distributed/test_async_dispatch.py`
- Reuse: the installed-binary + device-count-guard remote script pattern (`run_query_state_gate_verify.sh`) for execution.

**Interfaces:**
- Consumes: the full async backend. Runs under `torchrun --nproc-per-node 2`.

- [ ] **Step 1: Write the async-behavioral test (Work returns before completion)**

```python
# Copyright IBM Corp. 2026  (14-line Apache header)
import regex  # noqa: F401
import torch
import torch.distributed as dist
import torch_spyre  # noqa: F401

def test_allreduce_async_returns_before_completion():
    # init_process_group with spyre backend (see existing tests/distributed setup)
    t = torch.ones(64, dtype=torch.float16, device="spyre")
    work = dist.all_reduce(t, async_op=True)
    # Contract: async op returns a not-yet-necessarily-complete handle.
    # We cannot assert "not complete" deterministically (it may finish fast),
    # but we CAN assert the handle is a real Work and wait() then yields the sum.
    work.wait()
    expected = float(dist.get_world_size())
    assert torch.allclose(t.cpu(), torch.full((64,), expected, dtype=torch.float16))
```

- [ ] **Step 2: Write the multi-PG coexistence + C2 teardown test (the key v3 guard)**

```python
def test_world_and_tp_subgroup_coexist_and_teardown():
    # world PG + a TP subgroup via new_group (mirror tests/distributed/test_subgroup.py:111-147)
    world = dist.group.WORLD
    rank = dist.get_rank()
    tp = dist.new_group(ranks=[0, 1]) if dist.get_world_size() >= 2 else world
    # Interleave async collectives on world and the subgroup: must not deadlock.
    a = torch.ones(64, dtype=torch.float16, device="spyre")
    w1 = dist.all_reduce(a, async_op=True)                 # world
    b = torch.ones(64, dtype=torch.float16, device="spyre")
    w2 = dist.all_reduce(b, group=tp, async_op=True)       # subgroup
    w1.wait(); w2.wait()
    # C2: destroying the subgroup must NOT poison the world PG.
    dist.destroy_process_group(tp)
    c = torch.ones(64, dtype=torch.float16, device="spyre")
    dist.all_reduce(c)                                     # world still works
    assert torch.allclose(c.cpu(),
        torch.full((64,), float(dist.get_world_size()), dtype=torch.float16))
```

- [ ] **Step 3: Write the ordering-stress test**

```python
def test_back_to_back_collectives_preserve_order():
    for _ in range(50):
        t = torch.full((64,), float(dist.get_rank() + 1), dtype=torch.float16, device="spyre")
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
    # No hang / no matcher desync == pass (assert final value is broadcast of rank-0 sum).
```

- [ ] **Step 4: Write the destroy-before/after-launch Work test**

```python
def test_destroy_work_handle_variants():
    # ENQUEUED-ish: create and immediately drop the handle without wait.
    t = torch.ones(64, dtype=torch.float16, device="spyre")
    _ = dist.all_reduce(t, async_op=True)   # handle dropped -> dtor must not hang/UAF
    dist.barrier()                           # sync point; if the above UAF'd we'd crash
    # LAUNCHED: wait fully (already covered), plus a hold-tensor-freed-safely path.
```

- [ ] **Step 5: Author the remote run script (mirror run_query_state_gate_verify.sh)**

Create `run_async_dispatch_verify.sh` at workspace root: no build step, runs the installed torch-spyre test via `torchrun --nproc-per-node 2 -m pytest tests/distributed/test_async_dispatch.py`, device-count guard (≥2 vfio), TORCH_DEVICE_BACKEND_AUTOLOAD as needed, verdict on pytest exit + no `SIGSEGV|Aborted|timed out|deadlock` markers.

- [ ] **Step 6: (User builds+installs, then) run on HW**

Ask the user to build/install both repos (spyre-comms via their flex build, torch-spyre via `uv sync`), then run `run_async_dispatch_verify.sh` on the 4-spyre host.
Expected: all tests PASS; no hang, no segfault; multi-PG teardown leaves world PG working.

- [ ] **Step 7: Commit**

```bash
git add tests/distributed/test_async_dispatch.py
git commit -s -m "test(distributed): async behavioral + multi-PG coexistence/teardown + ordering HW tests"
# run_async_dispatch_verify.sh committed separately at workspace root if tracked
```

---

## Self-Review

**Spec coverage (§ → task):**
- §3 architecture (process-global worker builds+runs) → Tasks 3, 4, 6.
- §3.1 invariants: single FIFO (Task 3), single-issuer precondition (documented; §7 ordering test Task 10), caller-stream capture on caller thread (Task 7 Step 1/3), gate-on-WorkState (Task 5 Step 3), sole-driver (Task 3 + Task 5 dtor), tensor lifetime (Task 5/7 hold_tensors).
- §3.2 scope: in-scope 7 ops (Task 7), excluded ops untouched (Task 8 Step 3).
- §4.1 ProgressRequest / §4.2 WorkState → Task 2.
- §4.3 worker + F1/M3 discriminated WaitOutcome + F5 interruptible wait → Task 3.
- §4.4 SpyreCCLWork isCompleted/wait/dtor → Task 5.
- §4.5 entry-point capture+enqueue → Task 7.
- §4.6 host-reduce guard → Task 1 (predicate) + Task 3 Step 4 (wired) + Task 8 (tested).
- §5.2(a) backend-local teardown drain + M1 + M2 → Task 6 + Task 5 dtor. §5.2(b) count-0 worker join → Task 9.
- §5.3 error propagation (throw on DONE_ERROR) → Task 5 Step 4.
- §5.4 watchdog interaction (report_and_abort still flips shared stream on genuine fault) → unchanged; `on_error` routes to it (Task 7 helper).
- §7 tests → Task 10. R8 (M2 ordering) → Task 6 + Task 7 helper (inflight inc before enqueue, dec release in on_terminal). R9 (dead-peer escalation) → Task 3 TIMED_OUT path + Task 10 (documented; a true dead-peer HW test is optional/destructive).
- Reviewer item 1 (real tested host-reduce predicate) → Task 1 + Task 8. Item 2 (local_abort pre-launch only) → Task 3 `run_one` (checked pre-build; `wait_interruptible` does not consult it) + Task 7 `is_aborted` used only in the pre-launch check. Item 3 (no at::Tensor on worker) → Task 7 Step 4 audit.

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N" — each task repeats its own code. `build_for` stub in Task 3 is explicitly replaced in Task 4 (noted). `is_last` reconciliation in Task 6 is explicitly resolved in Task 9 (the plan flags it as provisional and Task 9 removes the parameter).

**Type consistency:** `WorkState`, `ProgressState`, `WaitOutcome`, `ProgressRequest`, `CollectiveParams`, `set_terminal`, `is_terminal`, `wait_interruptible`, `build_for`, `enqueue_async`, `spyre_global_progress_ref/unref/enqueue/is_running` — names used identically across Tasks 2-9. `spyre_global_progress_unref` signature changes from `(bool)` to `()` in Task 9, which updates all call sites (Task 3, Task 6) — flagged explicitly there.

**Known follow-ups (not gating, carried from review):** a true dead-peer HW test (R9) is destructive and optional; the `build_for` BARRIER path captures a current stream on the default device — confirm the exact device handle against the existing barrier at spyre_ccl.cpp:1171 during Task 7.
