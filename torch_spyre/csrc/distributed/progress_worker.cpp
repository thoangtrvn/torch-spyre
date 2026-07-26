/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "progress_worker.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <spyre_comms.hpp>
#include <string>
#include <thread>
#include <utility>

namespace torch_spyre::distributed {

void set_terminal(WorkState& st, ProgressState terminal, std::string reason) {
  {
    std::lock_guard<std::mutex> lk(st.m);
    if (!reason.empty()) st.error_reason = std::move(reason);
    st.state = terminal;
  }
  st.cv.notify_all();
}

namespace {

struct GlobalProgress {
  std::mutex qm;
  std::condition_variable qcv;
  std::deque<ProgressRequest> queue;
  std::thread thread;
  bool stop = false;
  int refcount = 0;  // guarded by qm
};

GlobalProgress& gp() {
  static GlobalProgress g;
  return g;
}

// TASK-4 STUB: only handles ALLREDUCE so this file compiles/links and the
// Task-3 worker plumbing can be exercised end to end. Task 4 replaces this
// with the full 7-op dispatch switch over req.op.
std::unique_ptr<spyre_comms::WorkSchedule> build_for(ProgressRequest& req) {
  return req.context->allreduce(req.buf, req.params.reduce_op);
}

void run_one(ProgressRequest& req) {
  // Pre-launch abort / cancel checks (M1: this is the ONLY place local abort
  // applies).
  {
    std::lock_guard<std::mutex> lk(req.state->m);
    if (req.state->cancelled) {
      req.on_terminal();
      return;  // dtor cancelled it; nothing built
    }
  }
  if (req.is_aborted()) {
    set_terminal(*req.state, ProgressState::DONE_ERROR,
                 "backend aborted before launch");
    req.on_terminal();
    return;
  }
  try {
    // BUILD (blocking OOB pre-exchange happens here, on THIS worker thread).
    std::unique_ptr<spyre_comms::WorkSchedule> ws = build_for(req);
    // Section 4.6 host-reduce guard: refuse to launch a host-compute
    // schedule.
    if (ws->containsHostReduceOp()) {
      set_terminal(*req.state, ProgressState::DONE_ERROR,
                   "host-reduce path excluded in Phase 1 async");
      req.on_terminal();
      return;
    }
    // Producer -> collective ordering fence, on the worker (was
    // caller-thread before).
    req.caller_stream.synchronize();
    ws->SetStreamAffinity(spyre_comms::get_comm_stream());
    // Publish ws + LAUNCHED under m, re-checking cancel (TOCTOU close,
    // Section 4.4).
    {
      std::lock_guard<std::mutex> lk(req.state->m);
      if (req.state->cancelled) {
        req.on_terminal();
        return;
      }
      req.state->ws = std::move(ws);
      req.state->state = ProgressState::LAUNCHED;
    }
    req.state->ws->start();  // LaunchAll: HDMA/CBS rendezvous
    switch (wait_interruptible(*req.state->ws, req.op_timeout)) {
      case WaitOutcome::COMPLETED: {
        const bool err = req.state->ws->getState() ==
                         spyre_comms::WorkScheduleState::State::DONE_ERROR;
        set_terminal(
            *req.state,
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
                     std::to_string(req.op_timeout.count()) +
                     " ms");  // -> report_and_abort
        set_terminal(*req.state, ProgressState::DONE_ERROR, "timed out");
        break;
    }
  }
  catch (const std::exception& e) {
    set_terminal(*req.state, ProgressState::DONE_ERROR, e.what());
    req.on_error(std::string("async collective: ") + e.what());
  }
  catch (...) {
    set_terminal(*req.state, ProgressState::DONE_ERROR, "unknown error");
    req.on_error("async collective: unknown error");
  }
  req.on_terminal();  // M2: LAST action, after all lambda use.
}

void worker_loop() {
  for (;;) {
    ProgressRequest req;
    {
      std::unique_lock<std::mutex> lk(gp().qm);
      gp().qcv.wait(lk, [] { return gp().stop || !gp().queue.empty(); });
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
