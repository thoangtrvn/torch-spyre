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
#pragma once

#include <pybind11/chrono.h>
#include <torch/python.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <spyre_comms.hpp>
#include <string>
#include <thread>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <vector>

#include "distributed/progress_worker.hpp"
#include "module.h"

namespace c10d {

/***********************************************
 * Custom Exception Classes
 ***********************************************/
/**
 * @brief Exception thrown when an operation is not supported by the Spyre CCL
 * backend.
 *
 * This exception is used to indicate that a particular collective operation
 * or feature is not implemented or supported by the Spyre communication
 * library.
 */
class SpyreCCLNotSupportedException : public std::runtime_error {
 public:
  /**
   * @brief Constructs a not supported exception with backend name and
   * operation.
   *
   * @param backend_name Name of the backend (e.g., "SpyreCCL")
   * @param operation_name Name of the unsupported operation
   */
  SpyreCCLNotSupportedException(const std::string& backend_name,
                                const std::string& operation_name)
      : std::runtime_error("[" + backend_name + "]: The \"" + operation_name +
                           "\" operation is not supported.") {}
};

/***********************************************
 * Wrapper torch.distributed Backend for the Sypre Collective Library
 ***********************************************/
class SpyreCCLBackend : public c10d::Backend {
 public:
  SpyreCCLBackend(const c10::intrusive_ptr<::c10d::Store>& store, int rank,
                  int size,
                  std::chrono::milliseconds op_timeout = kUnsetTimeout);

  ~SpyreCCLBackend();

  /*
   * Informative
   */
  [[nodiscard]] const std::string getBackendName() const override {
    return std::string("SpyreCCL");
  }

  /*
   * Sequence number support — required by _ProcessGroupWrapper when using
   * compound backends (e.g. cpu:gloo,spyre:spyreccl).  Follows the Gloo
   * model: start at 0 and increment on every collective.  The base class
   * Backend::setSequenceNumberForGroup() would TORCH_CHECK(false), so we
   * override to provide a working implementation.
   */
  void setSequenceNumberForGroup() override {
    // Gloo just starts at 0 — no store coordination needed.
  }
  uint64_t getSequenceNumberForGroup() override {
    return seq_.load(std::memory_order_relaxed);
  }

  /*
   * Allgather
   */
  [[nodiscard]] c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  [[nodiscard]] c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer, at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  /*
   * Allreduce
   */
  [[nodiscard]] c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  [[nodiscard]] c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  /*
   * Alltoall
   */
  [[nodiscard]] c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  [[nodiscard]] c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  /*
   * Barrier
   */
  [[nodiscard]] c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  /*
   * Broadcast
   */
  [[nodiscard]] c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  /*
   * Gather
   */
  [[nodiscard]] c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  /*
   * Reduce
   */
  [[nodiscard]] c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  /*
   * Reduce-Scatter
   */
  [[nodiscard]] c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  /*
   * Scatter
   */
  [[nodiscard]] c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  /*
   * Point-to-Point
   */
  [[nodiscard]] c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors,
                                              int dstRank, int tag) override;

  [[nodiscard]] c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors,
                                              int srcRank, int tag) override;

  [[nodiscard]] c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors, int tag) override;

  /*
   * Shutdown
   *
   * NOTE: The underlying spyre_comms library currently exposes no way to
   * cancel an in-flight WorkSchedule (there is no comms-level abort). These
   * therefore cannot forcibly interrupt a collective that is already running
   * on the hardware — they mark the backend as aborted so no further
   * collectives are launched and log the request. True cancellation of an
   * in-flight DMA requires a new spyre_comms API (tracked separately).
   */
  void abort() override;
  void shutdown() override;

  /*
   * Backend registration
   */
  [[nodiscard]] static c10::intrusive_ptr<Backend> createSpyreCCLBackend(
      const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
      const std::chrono::duration<float>& timeout);

 private:
  std::shared_ptr<spyre_comms::Context> group_context_;
  // Owned by spyre_comms global state. Valid from initialize_library()
  // until finalize_library(). SpyreCCLWork objects hold WorkSchedules
  // that reference this stream — they must not outlive the backend
  // (which calls finalize_library() in its destructor).
  flex::RuntimeStream* comm_stream_ = nullptr;
  std::atomic<uint64_t> seq_{0};
  // Set by abort()/shutdown(), OR by report_and_abort() when this rank's own
  // collective throws, OR by the watchdog thread when a peer's failure is
  // observed via the Store. Once true no further collectives are launched
  // (abort_guard) -- a permanent, one-way ratchet for the lifetime of this
  // backend, matching ProcessGroupNCCL: a faulted communicator is never
  // revived.
  std::atomic<bool> aborted_{false};
  // The process-group timeout from init_process_group(timeout=...), captured at
  // construction and used as the default deadline for Work::wait() when the
  // caller passes no explicit (positive) per-call timeout. kUnsetTimeout means
  // "block indefinitely" (no PG timeout configured). Immutable after
  // construction, so no synchronization is needed to read it.
  const std::chrono::milliseconds op_timeout_;

  // Retained from the constructor (previously discarded) for the cross-rank
  // fail-fast watchdog: a rank writes its own failure here so peers'
  // watchdog_loop() threads can detect it and abort promptly, instead of
  // sitting blocked until an unrelated subsystem's own timeout fires. See
  // report_and_abort()/watchdog_loop() and
  // flex-opensource/docs/process-group-fail-fast-investigation.md (Phase 1a).
  c10::intrusive_ptr<::c10d::Store> store_;
  std::thread watchdog_thread_;
  std::atomic<bool> watchdog_stop_{false};

  // Per-group failure-signal Store key. The base key is a single global
  // constant shared by every backend; suffixing it with a hash of this group's
  // sorted global ranks scopes peer-validation-failure propagation to this
  // group's members, so one subgroup's detected failure no longer aborts
  // coexisting sibling groups through the Store. (The shared comm_stream_
  // hardware-shutdown path remains process-fatal across all groups by design —
  // see report_and_abort().) Immutable after construction (set before the
  // watchdog thread starts), so no synchronization is needed to read it.
  std::string error_store_key_;

  // Backend-local teardown interrupt: gates ONLY pre-launch request drops (M1).
  // wait_interruptible never consults it — a launched DMA on the shared stream
  // is never abandoned. Set in the destructor / abort().
  std::atomic<bool> local_abort_{false};
  // Count of this backend's requests the worker has not yet driven to terminal.
  // Decremented (release) by the worker's on_terminal hook; the destructor
  // waits (acquire) for it to reach 0 before freeing this backend (M2/R8).
  std::atomic<int> inflight_{0};
  std::mutex inflight_mu_;
  std::condition_variable inflight_cv_;

  [[nodiscard]] spyre_comms::BufferDesc prepare_buffer_desc(
      const at::Tensor& input_tensor);

  // Layout-safe all_reduce(SUM) for a ≥2-D tensor. Both paths are HW-verified
  // correct (the original crashes were the transposed torch.ones layout, fixed
  // by the aten::ones/aten::full kernels in ops/eager.py -- NOT chunking).
  // Dispatches on SPYRE_ALLREDUCE_2D_WHOLE_TENSOR: default (unset/0) ->
  // allreduce_2d_chunked (bandwidth-optimal); =1 -> allreduce_2d_whole_tensor
  // (correct-by-construction fallback, ~2x-4x bytes). See the note above
  // allreduce_2d_compose() in spyre_ccl.cpp and
  // docs/issues/2d-allreduce-restickify-split-and-pivot.md.
  c10::intrusive_ptr<c10d::Work> allreduce_2d_compose(
      at::Tensor& tensor, const c10d::AllreduceOptions& opts);
  // DEFAULT: bandwidth-optimal reduce_scatter + all_gather over dim-0 chunks.
  c10::intrusive_ptr<c10d::Work> allreduce_2d_chunked(
      at::Tensor& tensor, const c10d::AllreduceOptions& opts);
  // FALLBACK (SPYRE_ALLREDUCE_2D_WHOLE_TENSOR=1): whole-tensor all_gather +
  // on-device sum (no narrow/clone/offset); correct-by-construction.
  c10::intrusive_ptr<c10d::Work> allreduce_2d_whole_tensor(
      at::Tensor& tensor, const c10d::AllreduceOptions& opts);

  // Build a request wired to this backend and enqueue it onto the process-
  // global async progress worker. Increments inflight_ BEFORE enqueue; the
  // worker's on_terminal hook decrements (release) and notifies inflight_cv_.
  // Everything captured into the request (buf, aux_bufs, params,
  // caller_stream) is a plain value -- no at::Tensor is ever visible to the
  // worker (C4). hold/result tensors live on the returned SpyreCCLWork and
  // are only touched by the calling thread.
  [[nodiscard]] c10::intrusive_ptr<Work> enqueue_async(
      OpType op, const spyre_comms::BufferDesc& buf,
      std::vector<spyre_comms::BufferDesc> aux_bufs,
      torch_spyre::distributed::CollectiveParams params,
      const spyre::SpyreStream& caller_stream, std::vector<at::Tensor> hold,
      std::vector<at::Tensor> result);

  // Perform one asymmetric pairwise exchange for the variable-size collectives:
  // send all of send_base's bytes to peer and receive all of recv_base's bytes
  // from peer, decomposed into a balanced sendrecv (min(S,R) bytes, any size)
  // plus credit-bounded one-directional send/recv sub-legs for the |S-R|
  // remainder. Each sub-leg is its own WorkSchedule (SetStreamAffinity + start
  // + wait), so every one-directional run stays within send_credit and the HDMA
  // credit guard never fires (broadcast/reduce are avoided entirely). round_tag
  // must be distinct per round and identical on both peers (both derive the
  // same decomposition from the shared sizes). Returns the last sub-leg's
  // schedule (nullptr if S==R==0).
  [[nodiscard]] std::unique_ptr<spyre_comms::WorkSchedule> exchange_uneven(
      const spyre_comms::BufferDesc& send_base,
      const spyre_comms::BufferDesc& recv_base, int peer, int round_tag);
  void check_single_tensor(const at::Tensor& tensor);
  void check_vector_tensor(const std::vector<at::Tensor>& tensors,
                           int min_allowed = 1, int max_allowed = 1);

  // Order the dedicated comm stream after the caller's current compute stream
  // so a collective never reads an input the producing compute has not yet
  // finished writing. See the definition for the current (host-side fence)
  // implementation and the device-event follow-up.
  void order_after_caller_stream(const at::Tensor& ref_tensor);

  // Throws if abort()/shutdown() has been called, preventing new collectives
  // from being launched on a backend that is tearing down.
  void abort_guard(const char* op);

  // Marks this backend permanently aborted (idempotent -- a no-op if already
  // aborted, by this rank's own failure, a peer's, or an explicit
  // abort()/shutdown()), writes msg to the shared Store error key so peers'
  // watchdog_loop() threads learn of it, and flags comm_stream_ for shutdown
  // with msg as the reason so a peer already blocked in synchronize() returns
  // promptly instead of waiting for an unrelated timeout. Called both by the
  // collective methods' own catch blocks (this rank's failure) and by
  // watchdog_loop() (a peer's failure) -- the two converge on identical
  // handling.
  void report_and_abort(const std::string& msg);

  // Background thread (started in the constructor, stopped+joined in the
  // destructor before finalize_library()) that polls the Store for any
  // peer's failure and calls report_and_abort() on this rank when found.
  void watchdog_loop();
};

/***********************************************
 * Wrapper backend for the Sypre Collective Library - Work
 ***********************************************/
class SpyreCCLWork : public Work {
  friend class SpyreCCLBackend;

 public:
  /**
   * @param opType         The collective op type (for diagnostics).
   * @param state          Shared progress state, published by the async
   *                       progress worker (Task 2/3). This Work observes it
   *                       via state->cv; it never owns or drives the
   *                       underlying WorkSchedule (single-driver rule -- the
   *                       worker is the sole owner/driver of state->ws).
   * @param hold_tensors   Input/output tensors kept alive by this Work for the
   *                       full duration of the async op. Because the schedule
   *                       captures raw device pointers, releasing these tensors
   *                       before completion would free memory the collective is
   *                       still using (use-after-free). Holding a reference
   * here ties their lifetime to the Work.
   * @param result_tensors The output tensors used to complete the Future so
   *                       fut.value() / functional-collective consumers observe
   *                       the collective result. Must be a subset of
   *                       hold_tensors.
   * @param default_timeout The process-group timeout to apply in wait() when
   *                       the caller passes no explicit (positive) per-call
   *                       timeout. kUnsetTimeout means "block indefinitely".
   */
  SpyreCCLWork(OpType opType,
               std::shared_ptr<torch_spyre::distributed::WorkState> state,
               std::vector<at::Tensor> hold_tensors = {},
               std::vector<at::Tensor> result_tensors = {},
               std::chrono::milliseconds default_timeout = kUnsetTimeout);

  /**
   * @param opType         The collective op type (for diagnostics).
   * @param work_schedule  Owned WorkSchedule for the synchronous (excluded-op)
   *                       path. This Work is the sole owner/driver of the
   *                       schedule -- it queries/waits on it directly (unlike
   *                       the async ctor above, which only observes a shared
   *                       WorkState published by the async progress worker).
   * @param hold_tensors   Input/output tensors kept alive by this Work for the
   *                       full duration of the op. Because the schedule
   *                       captures raw device pointers, releasing these tensors
   *                       before completion would free memory the collective is
   *                       still using (use-after-free). Holding a reference
   * here ties their lifetime to the Work.
   * @param result_tensors The output tensors used to complete the Future so
   *                       fut.value() / functional-collective consumers observe
   *                       the collective result. Must be a subset of
   *                       hold_tensors.
   * @param default_timeout The process-group timeout to apply in wait() when
   *                       the caller passes no explicit (positive) per-call
   *                       timeout. kUnsetTimeout means "block indefinitely".
   */
  SpyreCCLWork(OpType opType,
               std::unique_ptr<spyre_comms::WorkSchedule> work_schedule,
               std::vector<at::Tensor> hold_tensors = {},
               std::vector<at::Tensor> result_tensors = {},
               std::chrono::milliseconds default_timeout = kUnsetTimeout);
  ~SpyreCCLWork() override;
  [[nodiscard]] bool isCompleted() override;
  [[nodiscard]] bool isSuccess() const override;
  [[nodiscard]] bool wait(
      std::chrono::milliseconds timeout = kUnsetTimeout) override;
  [[nodiscard]] virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture()
      override;

 private:
  // Transition the Future to its terminal state exactly once. finish_success()
  // completes it with the result tensor list; finish_error() propagates the
  // failure. Guarded by the completed_ CAS in the callers.
  void finish_success();
  void finish_error(const std::string& msg);

  c10::intrusive_ptr<c10::ivalue::Future> future_;
  // Set iff constructed via the async ctor (shared state published by the
  // async progress worker); null on the sync path.
  std::shared_ptr<torch_spyre::distributed::WorkState> state_;
  // Set iff constructed via the sync ctor (owned/driven directly by this
  // Work); null on the async path. Exactly one of state_/work_schedule_ is
  // non-null.
  std::unique_ptr<spyre_comms::WorkSchedule> work_schedule_;
  std::vector<at::Tensor> hold_tensors_;
  std::vector<at::Tensor> result_tensors_;
  std::atomic<bool> completed_{false};
  // PG-level default deadline applied by wait() when the caller passes no
  // explicit positive timeout. kUnsetTimeout means "block indefinitely".
  const std::chrono::milliseconds default_timeout_;
};

}  // namespace c10d
