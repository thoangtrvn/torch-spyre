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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <spyre_comms.hpp>
#include <spyre_comms_tensor.hpp>
#include <string>
#include <torch/csrc/distributed/c10d/Types.hpp>  // OpType
#include <vector>

#include "spyre_stream.h"

namespace torch_spyre::distributed {

enum class ProgressState {
  ENQUEUED,
  BUILDING,
  LAUNCHED,
  DONE_SUCCESS,
  DONE_ERROR
};
enum class WaitOutcome { COMPLETED, SHUTDOWN, TIMED_OUT };

inline bool is_terminal(ProgressState s) {
  return s == ProgressState::DONE_SUCCESS || s == ProgressState::DONE_ERROR;
}

struct WorkState {
  std::mutex m;
  std::condition_variable cv;
  ProgressState state = ProgressState::ENQUEUED;  // guarded by m
  bool cancelled = false;                         // guarded by m (R3)
  std::string error_reason;  // set before terminal; read after
  std::unique_ptr<spyre_comms::WorkSchedule> ws;  // built + owned by the worker
};

struct CollectiveParams {
  spyre_comms::SpyreReductionOpType reduce_op =
      spyre_comms::SpyreReductionOpType::SUM;
  spyre_comms::process_id_t root = 0;
  spyre_comms::process_id_t peer = 0;
  int tag = 0;
};

struct ProgressRequest {
  c10d::OpType op;
  std::shared_ptr<spyre_comms::Context> context;
  spyre_comms::BufferDesc buf;
  std::vector<spyre_comms::BufferDesc>
      aux_bufs;  // output slots for gather/allgather
  CollectiveParams params;
  spyre::SpyreStream caller_stream;
  std::chrono::milliseconds op_timeout;
  std::function<bool()> is_aborted;  // pre-launch abort/local-abort check (M1)
  std::function<void(const std::string&)>
      on_error;  // genuine-fault escalation (report_and_abort)
  std::function<void()> on_terminal;  // per-backend in-flight decrement (M2)
  std::shared_ptr<WorkState> state;
};

// Transition to a terminal state under the WorkState mutex and notify all
// waiters.
void set_terminal(WorkState& st, ProgressState terminal, std::string reason);

}  // namespace torch_spyre::distributed
