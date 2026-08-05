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
#include "spyre_ccl.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_allocator.h"
#include "spyre_stream.h"
#include "types_mapping.h"

namespace c10d {

namespace {
// Single well-known Store key used to signal a collective failure to every
// peer in the process group, matching ProcessGroupNCCL's own reference
// pattern (one shared key, not one per rank -- sidesteps needing to know
// Store::check()'s multi-key ALL-vs-ANY semantics). The value encodes which
// rank failed and why, so a peer's watchdog_loop() can produce an actionable
// message.
constexpr const char* kSpyreCCLErrorStoreKey = "spyre_ccl/error_signal";
constexpr auto kWatchdogPollInterval = std::chrono::milliseconds(200);

// FNV-1a hash of the sorted global group ranks, as a decimal string. Used as a
// stable per-group suffix for the failure-signal Store key: identical on every
// member (same membership -> same string), distinct across groups with
// different membership. Uniqueness (not cryptographic strength) is all that is
// required.
std::string group_key_suffix(std::vector<spyre_comms::process_id_t> ranks) {
  std::sort(ranks.begin(), ranks.end());
  uint64_t h = 1469598103934665603ULL;
  for (auto r : ranks) {
    for (int b = 0; b < 8; ++b) {
      h ^= static_cast<uint64_t>((r >> (b * 8)) & 0xffULL);
      h *= 1099511628211ULL;
    }
  }
  return std::to_string(h);
}

// Store-based allgather of every member's global rank, returned in group-local
// index order (result[i] == global rank of group-local rank i). Each member
// publishes its own global rank at its group-local index, then reads all size
// entries (get_to_str blocks until each peer has published). Runs in the
// group's own PrefixStore, so the fixed key prefix cannot collide across
// groups.
std::vector<spyre_comms::process_id_t> allgather_group_global_ranks(
    const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
    int64_t my_global_rank) {
  const std::string prefix = "spyre_ccl/subgroup_grank/";
  store->set(prefix + std::to_string(rank), std::to_string(my_global_rank));
  std::vector<spyre_comms::process_id_t> group_ranks(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i) {
    std::string v = store->get_to_str(prefix + std::to_string(i));
    group_ranks[static_cast<size_t>(i)] =
        static_cast<spyre_comms::process_id_t>(std::stoll(v));
  }
  return group_ranks;
}
}  // namespace

/***********************************************
 * Wrapper Backend for the Sypre Collective Library
 ***********************************************/
SpyreCCLBackend::SpyreCCLBackend(const c10::intrusive_ptr<::c10d::Store>& store,
                                 int rank, int size,
                                 std::chrono::milliseconds op_timeout)
    : Backend(rank, size),
      group_context_(nullptr),
      comm_stream_(nullptr),
      op_timeout_(op_timeout) {
  DEBUGINFO("# [Spyre CCL]: Constructor for ", getBackendName());

  /*
   * Start the communication library
   * Pass it the shared runtime library handle, and default stream.
   *
   * initialize_library()/finalize_library() are reference counted inside the
   * comms library, so it is safe for every ProcessGroup to call this pair; the
   * device is only torn down when the last group finalizes.
   */
  spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                  spyre::getDefaultStreamRuntimeHandle());
  std::shared_ptr<spyre_comms::Context> world_context =
      spyre_comms::get_world_context();
  if (nullptr == world_context) {
    // Balance the init refcount we just took: because this constructor throws,
    // the destructor (which calls finalize_library()) will NOT run.
    spyre_comms::finalize_library();
    std::string _err_msg =
        "[" + getBackendName() + "]: Failed to capture the world context";
    throw std::runtime_error(_err_msg);
  }

  /*
   * Subgroup support.
   *
   * For the default (world) ProcessGroup the c10d rank/size match the world
   * context's, and we reuse it directly. For a new_group() with a strict subset
   * of ranks, we build the group's global-rank vector via a Store allgather and
   * hand it to spyre_comms::create_context(), which returns a logical
   * communicator over the world's already-established transport (no re-init).
   * The rank/size the constructor receives are group-local; create_context maps
   * them to global internally via the group_ranks vector.
   */
  const bool is_world_group = (static_cast<spyre_comms::process_id_t>(size) ==
                               world_context->getSize()) &&
                              (static_cast<spyre_comms::process_id_t>(rank) ==
                               world_context->getRank());

  std::vector<spyre_comms::process_id_t> group_ranks;
  try {
    if (is_world_group) {
      // World group: global ranks are 0..size-1, in order.
      group_ranks.resize(static_cast<size_t>(size));
      for (int i = 0; i < size; ++i) {
        group_ranks[static_cast<size_t>(i)] =
            static_cast<spyre_comms::process_id_t>(i);
      }
      group_context_ = world_context;
    } else {
      // Subgroup: allgather every member's global rank (in group-local index
      // order) through the group's own PrefixStore, then build the sub-context.
      group_ranks = allgather_group_global_ranks(
          store, rank, size, static_cast<int64_t>(world_context->getRank()));
      group_context_ = spyre_comms::create_context(world_context, group_ranks);
    }
  }
  catch (...) {
    // Balance the init refcount taken above before propagating: because this
    // constructor throws, the destructor (which calls finalize_library()) will
    // NOT run.
    spyre_comms::finalize_library();
    throw;
  }
  if (nullptr == group_context_) {
    spyre_comms::finalize_library();
    throw std::runtime_error("[" + getBackendName() +
                             "]: Failed to create the group context");
  }

  // Per-group failure-signal key (see error_store_key_ in the header): scope
  // the shared Store error key to this group's members so one subgroup's
  // detected failure no longer aborts coexisting sibling groups. Set BEFORE the
  // watchdog thread starts (which reads it), so no synchronization is needed.
  error_store_key_ =
      std::string(kSpyreCCLErrorStoreKey) + "/" + group_key_suffix(group_ranks);

  // Use the dedicated comm stream so collectives run independently
  // of the compute stream, enabling compute/communication overlap.
  comm_stream_ = spyre_comms::get_comm_stream();

  // Start (or ref) the one process-global progress worker. Refcounted so the
  // Nth backend reuses the running worker; only the last unref (dtor) joins
  // it.
  torch_spyre::distributed::spyre_global_progress_ref();

  // Phase 1a cross-rank fail-fast: retain the Store (previously discarded
  // after construction) and start the background watchdog that polls it for
  // a peer's failure. See report_and_abort()/watchdog_loop().
  store_ = store;
  watchdog_thread_ = std::thread(&SpyreCCLBackend::watchdog_loop, this);
}

SpyreCCLBackend::~SpyreCCLBackend() {
  // (§5.2a) Stop new enqueues + let the worker drop this backend's
  // un-launched requests. Backend-local: does NOT shut down the shared
  // comm_stream_.
  aborted_.store(true, std::memory_order_release);
  local_abort_.store(true, std::memory_order_release);
  // Drain: wait until every request this backend issued has reached
  // terminal, so the worker is no longer calling into our
  // on_error/on_terminal/is_aborted lambdas (which capture `this`) and no DMA
  // is still reading held tensors.
  {
    std::unique_lock<std::mutex> lk(inflight_mu_);
    inflight_cv_.wait(
        lk, [&] { return inflight_.load(std::memory_order_acquire) == 0; });
  }
  // Stop + join the watchdog (unchanged).
  watchdog_stop_.store(true, std::memory_order_release);
  if (watchdog_thread_.joinable()) {
    watchdog_thread_.join();
  }
  // Unref the process-global worker. The worker's own refcount is 1:1 with
  // backends (ref'd once per ctor), so it is authoritative on its own: the
  // worker joins whenever THIS unref drops its count to 0, with no need to
  // consult spyre-comms's internal getInits(). This join must precede
  // finalize_library()'s count-0 comm-stream teardown.
  torch_spyre::distributed::spyre_global_progress_unref();
  spyre_comms::finalize_library();
}

/* **********************************************
 * Internal support functions
 ********************************************** */

/**
 * @brief Converts PyTorch reduction operation type to Spyre reduction operation
 * type.
 *
 * Maps c10d::ReduceOp enum values to spyre_comms::SpyreReductionOpType enum
 * values. Currently only supports SUM operation; other operations return
 * UNSUPPORTED.
 *
 * @param reduce_op The PyTorch reduction operation type to convert
 * @return The corresponding Spyre reduction operation type, or UNSUPPORTED if
 * not supported
 */
spyre_comms::SpyreReductionOpType convert_reduce_op_type(
    const ReduceOp reduce_op) {
  switch (reduce_op) {
    case ReduceOp::SUM:
      return spyre_comms::SpyreReductionOpType::SUM;
    default:
      return spyre_comms::SpyreReductionOpType::UNSUPPORTED;
  }
}

inline std::pair<flex::sen_datatype_enum, flex::sen_datatype_enum>
convert_string_to_datatype_pair(const std::string& type_name) {
  /* val-1 = type on CPU-side
   * val-2 = type on Spyre-side
   */
  static const std::unordered_map<
      std::string, std::pair<flex::sen_datatype_enum, flex::sen_datatype_enum>>
      type_map = {
          // Boolean and string
          {"bool",
           {flex::sen_datatype_enum::boolean,
            flex::sen_datatype_enum::sen_fp16}},
          {"string",
           {flex::sen_datatype_enum::string, flex::sen_datatype_enum::string}},

          // IEEE floats
          {"fp8_143",
           {flex::sen_datatype_enum::float8, flex::sen_datatype_enum::sen_fp8}},
          // TODO(tmhoangt): figure out why there is not FP8 variant specific in
          // sen_datatype_enum
          {"fp8_152",
           {flex::sen_datatype_enum::float8, flex::sen_datatype_enum::sen_fp8}},
          {"float16",
           {flex::sen_datatype_enum::float16,
            flex::sen_datatype_enum::sen_fp16}},
          {"float32",
           {flex::sen_datatype_enum::float32,
            flex::sen_datatype_enum::float32}},
          {"float64",
           {flex::sen_datatype_enum::float64,
            flex::sen_datatype_enum::float64}},
          {"float128",
           {flex::sen_datatype_enum::float128,
            flex::sen_datatype_enum::float128}},
          {"float256",
           {flex::sen_datatype_enum::float256,
            flex::sen_datatype_enum::float256}},

          // Decimal
          {"decimal32",
           {flex::sen_datatype_enum::decimal32,
            flex::sen_datatype_enum::decimal32}},
          {"decimal64",
           {flex::sen_datatype_enum::decimal64,
            flex::sen_datatype_enum::decimal64}},
          {"decimal128",
           {flex::sen_datatype_enum::decimal128,
            flex::sen_datatype_enum::decimal128}},

          // bfloat
          {"bfloat16",
           {flex::sen_datatype_enum::bfloat16,
            flex::sen_datatype_enum::sen_fp16}},
          {"bfloat16_compute",
           {flex::sen_datatype_enum::bfloat16,
            flex::sen_datatype_enum::float32}},

          // Signed ints
          {"int1",
           {flex::sen_datatype_enum::int1, flex::sen_datatype_enum::sen_int1}},
          {"int2",
           {flex::sen_datatype_enum::int2, flex::sen_datatype_enum::sen_int2}},
          {"int4",
           {flex::sen_datatype_enum::int4, flex::sen_datatype_enum::sen_int4}},
          {"int8",
           {flex::sen_datatype_enum::int8, flex::sen_datatype_enum::sen_int8}},
          {"int16",
           {flex::sen_datatype_enum::int16,
            flex::sen_datatype_enum::sen_int16}},
          {"int32",
           {flex::sen_datatype_enum::int32,
            flex::sen_datatype_enum::sen_int32}},
          {"int64",
           {flex::sen_datatype_enum::int64,
            flex::sen_datatype_enum::sen_int32}},

          // Unsigned ints
          {"uint1",
           {flex::sen_datatype_enum::uint1,
            flex::sen_datatype_enum::sen_uint1}},
          {"uint2",
           {flex::sen_datatype_enum::uint2,
            flex::sen_datatype_enum::sen_uint2}},
          {"uint4",
           {flex::sen_datatype_enum::uint4,
            flex::sen_datatype_enum::sen_uint4}},
          {"uint8",
           {flex::sen_datatype_enum::uint8,
            flex::sen_datatype_enum::sen_uint8}},
          {"uint16",
           {flex::sen_datatype_enum::uint16,
            flex::sen_datatype_enum::sen_uint16}},
          {"uint32",
           {flex::sen_datatype_enum::uint32,
            flex::sen_datatype_enum::sen_uint32}},
          {"uint64",
           {flex::sen_datatype_enum::uint64,
            flex::sen_datatype_enum::sen_uint32}},

          // Quantized ints
          {"qint1",
           {flex::sen_datatype_enum::qint1, flex::sen_datatype_enum::qint1}},
          {"qint2",
           {flex::sen_datatype_enum::qint2, flex::sen_datatype_enum::qint2}},
          {"qint4",
           {flex::sen_datatype_enum::qint4, flex::sen_datatype_enum::qint4}},
          {"qint8",
           {flex::sen_datatype_enum::qint8, flex::sen_datatype_enum::qint8}},
          {"qint16",
           {flex::sen_datatype_enum::qint16, flex::sen_datatype_enum::qint16}},
          {"qint32",
           {flex::sen_datatype_enum::qint32, flex::sen_datatype_enum::qint32}},
          {"qint64",
           {flex::sen_datatype_enum::qint64, flex::sen_datatype_enum::qint64}},

          {"quint1",
           {flex::sen_datatype_enum::quint1, flex::sen_datatype_enum::quint1}},
          {"quint2",
           {flex::sen_datatype_enum::quint2, flex::sen_datatype_enum::quint2}},
          {"quint4",
           {flex::sen_datatype_enum::quint4, flex::sen_datatype_enum::quint4}},
          {"quint8",
           {flex::sen_datatype_enum::quint8, flex::sen_datatype_enum::quint8}},
          {"quint16",
           {flex::sen_datatype_enum::quint16,
            flex::sen_datatype_enum::quint16}},
          {"quint32",
           {flex::sen_datatype_enum::quint32,
            flex::sen_datatype_enum::quint32}},
          {"quint64",
           {flex::sen_datatype_enum::quint64,
            flex::sen_datatype_enum::quint64}},

          // Complex
          {"complex64",
           {flex::sen_datatype_enum::complex64,
            flex::sen_datatype_enum::complex64}},
          {"complex128",
           {flex::sen_datatype_enum::complex128,
            flex::sen_datatype_enum::complex128}},

          // Sentient types
          {"sen_fp8",
           {flex::sen_datatype_enum::sen_fp8,
            flex::sen_datatype_enum::sen_fp8}},
          {"sen_fp16",
           {flex::sen_datatype_enum::sen_fp16,
            flex::sen_datatype_enum::sen_fp16}},
          {"sen_fp8_compute",
           {flex::sen_datatype_enum::sen_fp8,
            flex::sen_datatype_enum::float32}},
          {"sen_fp16_compute",
           {flex::sen_datatype_enum::sen_fp16,
            flex::sen_datatype_enum::float32}},

          {"sen_int1",
           {flex::sen_datatype_enum::sen_int1,
            flex::sen_datatype_enum::sen_int1}},
          {"sen_int2",
           {flex::sen_datatype_enum::sen_int2,
            flex::sen_datatype_enum::sen_int2}},
          {"sen_int4",
           {flex::sen_datatype_enum::sen_int4,
            flex::sen_datatype_enum::sen_int4}},
          {"sen_int8",
           {flex::sen_datatype_enum::sen_int8,
            flex::sen_datatype_enum::sen_int8}},
          {"sen_int16",
           {flex::sen_datatype_enum::sen_int16,
            flex::sen_datatype_enum::sen_int16}},
          {"sen_int24",
           {flex::sen_datatype_enum::sen_int24,
            flex::sen_datatype_enum::sen_int24}},
          {"sen_int32",
           {flex::sen_datatype_enum::sen_int32,
            flex::sen_datatype_enum::sen_int32}},
          {"sen_int4_compute",
           {flex::sen_datatype_enum::sen_int4, flex::sen_datatype_enum::int32}},
          {"sen_int8_compute",
           {flex::sen_datatype_enum::sen_int8, flex::sen_datatype_enum::int32}},

          {"sen_uint1",
           {flex::sen_datatype_enum::sen_uint1,
            flex::sen_datatype_enum::sen_uint1}},
          {"sen_uint2",
           {flex::sen_datatype_enum::sen_uint2,
            flex::sen_datatype_enum::sen_uint2}},
          {"sen_uint4",
           {flex::sen_datatype_enum::sen_uint4,
            flex::sen_datatype_enum::sen_uint4}},
          {"sen_uint8",
           {flex::sen_datatype_enum::sen_uint8,
            flex::sen_datatype_enum::sen_uint8}},
          {"sen_uint16",
           {flex::sen_datatype_enum::sen_uint16,
            flex::sen_datatype_enum::sen_uint16}},
          {"sen_uint24",
           {flex::sen_datatype_enum::sen_uint24,
            flex::sen_datatype_enum::sen_uint24}},
          {"sen_uint32",
           {flex::sen_datatype_enum::sen_uint32,
            flex::sen_datatype_enum::sen_uint32}},
      };

  auto it = type_map.find(type_name);
  if (it != type_map.end()) {
    return it->second;
  }
  return {flex::sen_datatype_enum::dt_undef, flex::sen_datatype_enum::dt_undef};
}

/**
 * @brief Converts a PyTorch tensor to a spyre_comms::BufferDesc.
 *
 * Produces a lightweight buffer descriptor with host pointer, device address,
 * byte count, and dtype — without constructing a TensorInfo/TensorShape
 * hierarchy.
 *
 * @param input_tensor The PyTorch tensor (must be on Spyre device)
 * @return BufferDesc with host pointer, device address, byte count, dtype
 */
spyre_comms::BufferDesc SpyreCCLBackend::prepare_buffer_desc(
    const at::Tensor& input_tensor) {
  auto* raw_data_ptr = input_tensor.storage().data_ptr().get();
  auto* raw_ctx = input_tensor.storage().data_ptr().get_context();
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(raw_ctx);

  // Compute dtype directly from PyTorch scalar type
  auto str_type = spyre::torchScalarToString[input_tensor.scalar_type()];
  const auto [sen_dtype_cpu, _sen_dtype_dev] =
      convert_string_to_datatype_pair(str_type);

  // A 0-numel tensor (e.g. one rank's empty contribution to a variable-size
  // all_gatherv/reduce_scatterv) never gets a real Spyre allocation, so its
  // context is legitimately NULL -- not an error. Every exchange_uneven/
  // Context call already treats a 0-byte BufferDesc as a no-op without
  // dereferencing device_addr, so a nullptr address here is never touched.
  if (input_tensor.numel() == 0) {
    return spyre_comms::BufferDesc{raw_data_ptr, nullptr, 0, sen_dtype_cpu,
                                   false};
  }

  if (ctx == nullptr) {
    TORCH_CHECK(false,
                "prepare_buffer_desc: get_context() returned NULL — tensor has "
                "no device address");
  }

  // Compute byte count matching TensorInfo::DataSize():
  // (volume * bits_per_element + 7) / 8, with rounding for sub-byte types.
  size_t byte_count =
      static_cast<size_t>((static_cast<uint64_t>(input_tensor.numel()) *
                               flex::DataTypeSizeBits(sen_dtype_cpu) +
                           7) /
                          8);

  return spyre_comms::BufferDesc{
      raw_data_ptr, &ctx->composite_addr, byte_count, sen_dtype_cpu,
      false  // is_host_only
  };
}

std::unique_ptr<spyre_comms::WorkSchedule> SpyreCCLBackend::exchange_uneven(
    const spyre_comms::BufferDesc& send_base,
    const spyre_comms::BufferDesc& recv_base, int peer, int round_tag) {
  const size_t S = send_base.byte_count;
  const size_t R = recv_base.byte_count;
  const size_t balanced = std::min(S, R);
  const size_t limit = spyre_comms::max_one_directional_transfer_bytes();
  constexpr size_t kMinTransfer = 128;

  std::unique_ptr<spyre_comms::WorkSchedule> last;
  auto run = [&](std::unique_ptr<spyre_comms::WorkSchedule> ws) {
    ws->SetStreamAffinity(comm_stream_);
    ws->start();
    ws->wait();
    last = std::move(ws);
  };

  // Distinct tag namespace per round (both peers derive the identical
  // decomposition from the shared sizes, so they assign matching sub-leg tags).
  constexpr int kTagStride = 1024;
  int tag = round_tag * kTagStride;

  // Balanced portion: one bidirectional sendrecv of `balanced` bytes each way.
  // Bidirectional runs are exempt from the one-directional credit guard at any
  // size, so this is a single sub-leg regardless of how large `balanced` is.
  if (balanced > 0) {
    spyre_comms::SubRangeBuffer s =
        spyre_comms::make_byte_subrange(send_base, 0, balanced);
    spyre_comms::SubRangeBuffer r =
        spyre_comms::make_byte_subrange(recv_base, 0, balanced);
    // s and r stay alive across this call; sendrecv copies their addresses into
    // its own schedule (computeChunkExchangeInfos), so they may drop
    // afterwards.
    run(group_context_->sendrecv(s.desc, r.desc, peer, tag));
  }
  ++tag;

  // Remainder: |S - R| bytes in a single direction, chunked so each sub-leg is
  // <= send_credit chunks (<= limit bytes) and therefore passes the guard.
  auto push_one_directional = [&](const spyre_comms::BufferDesc& base,
                                  size_t total, bool is_send) {
    size_t off = balanced;
    while (off < total) {
      size_t len = std::min(total - off, limit);
      // Keep a trailing sub-leg at >= the 128-byte minimum transfer.
      if ((total - off) - len != 0 && (total - off) - len < kMinTransfer) {
        len = (total - off) - kMinTransfer;
      }
      spyre_comms::SubRangeBuffer sub =
          spyre_comms::make_byte_subrange(base, off, len);
      run(is_send ? group_context_->send(sub.desc, peer, tag)
                  : group_context_->recv(sub.desc, peer, tag));
      off += len;
      ++tag;
    }
  };

  if (S > R) {
    push_one_directional(send_base, S, /*is_send=*/true);
  } else if (R > S) {
    push_one_directional(recv_base, R, /*is_send=*/false);
  }

  return last;
}

/**
 * @brief Validates that a single tensor meets requirements for collective
 * operations.
 *
 * Checks that the tensor is contiguous, dense (not sparse), and suitable for
 * Spyre device operations. Throws an error if any validation fails.
 *
 * @param tensor The tensor to validate
 * @throws TORCH_CHECK exception if tensor is not contiguous or is sparse
 */
void SpyreCCLBackend::check_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "The tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "The tensor has to be dense");
  }
  // Must be a Spyre (PrivateUse1) device tensor. prepare_buffer_desc()
  // reinterprets the storage's DataPtr context as a spyre::SharedOwnerCtx; a
  // CPU/other-device tensor would make that a garbage reinterpret and produce
  // an invalid device address. Reject it here with a clear error instead.
  if (tensor.device().type() != c10::DeviceType::PrivateUse1) {
    TORCH_CHECK(false, "[", getBackendName(),
                "]: collective tensors must be on the Spyre device; got ",
                tensor.device().str());
  }
}

/**
 * @brief Order the dedicated comm stream after the caller's current compute
 * stream.
 *
 * The collective runs asynchronously on comm_stream_, which is independent of
 * the stream that produced the input tensor. Without ordering, the collective
 * could DMA the input before the producing compute has finished writing it,
 * yielding wrong results (e.g. a TP all_reduce right after a matmul).
 *
 * Current implementation is a host-side fence: it blocks the host until the
 * caller's current stream is idle, guaranteeing the input is fully written
 * before the collective is launched. The output side is ordered by
 * SpyreCCLWork::wait()/synchronize(), which host-blocks on the comm stream so
 * that any downstream compute the caller launches after wait() sees the result.
 *
 * TODO(perf): replace this host-side fence with a device-side event wait
 * (record a flex::RuntimeEvent on the caller's compute stream and
 * comm_stream_->waitForEvent(event)) once torch-spyre exposes the current
 * stream's flex::RuntimeStream* handle. That preserves host-side overlap; the
 * fence here serializes the producer on the host. Note the existing
 * flex::RuntimeEvent is itself a host-side (P0) primitive today.
 */
void SpyreCCLBackend::order_after_caller_stream(const at::Tensor& ref_tensor) {
  spyre::getCurrentStream(ref_tensor.device()).synchronize();
}

/**
 * @brief Validates a vector of tensors for collective operations.
 *
 * Checks that the number of tensors in the vector is within the specified range
 * [min_allowed, max_allowed], and validates each individual tensor using
 * check_single_tensor(). Throws an error if validation fails.
 *
 * @param tensors The vector of tensors to validate
 * @param min_allowed Minimum number of tensors allowed in the vector
 * @param max_allowed Maximum number of tensors allowed in the vector
 * @throws TORCH_CHECK exception if tensor count is out of range or any tensor
 * is invalid
 */
void SpyreCCLBackend::check_vector_tensor(
    const std::vector<at::Tensor>& tensors, int min_allowed, int max_allowed) {
  if (static_cast<int>(tensors.size()) < min_allowed) {
    std::string _err_msg = "[" + getBackendName() +
                           "]: Too few tensors. Expected at least " +
                           std::to_string(min_allowed) +
                           " Actual: " + std::to_string(tensors.size());
    TORCH_CHECK(false, _err_msg);
  }
  if (static_cast<int>(tensors.size()) > max_allowed) {
    std::string _err_msg = "[" + getBackendName() +
                           "]: Too many tensors. Expected at most " +
                           std::to_string(max_allowed) +
                           " Actual: " + std::to_string(tensors.size());
    TORCH_CHECK(false, _err_msg);
  }
  for (auto& tensor : tensors) {
    check_single_tensor(tensor);
  }
}

/**
 * @brief Build a ProgressRequest wired to this backend and enqueue it onto
 * the process-global async progress worker.
 *
 * Increments inflight_ BEFORE enqueue so the destructor's drain-wait never
 * races a request that has not yet been counted. The worker's on_terminal
 * hook decrements (release) and notifies inflight_cv_ once the op reaches a
 * terminal state. Everything captured into the ProgressRequest (buf,
 * aux_bufs, params, caller_stream) is a plain value/BufferDesc/scalar --
 * no at::Tensor crosses to the worker (C4). hold/result tensors live on the
 * returned SpyreCCLWork, observed only by the calling thread.
 *
 * DISJOINT-MUTEX: on_terminal locks inflight_mu_ only, never state->m -- the
 * worker may call on_terminal while already holding state->m, so touching
 * state->m here would deadlock.
 */
c10::intrusive_ptr<Work> SpyreCCLBackend::enqueue_async(
    OpType op, const spyre_comms::BufferDesc& buf,
    std::vector<spyre_comms::BufferDesc> aux_bufs,
    torch_spyre::distributed::CollectiveParams params,
    const spyre::SpyreStream& caller_stream, std::vector<at::Tensor> hold,
    std::vector<at::Tensor> result) {
  auto state = std::make_shared<torch_spyre::distributed::WorkState>();
  inflight_.fetch_add(1, std::memory_order_acq_rel);
  torch_spyre::distributed::ProgressRequest req{
      .op = op,
      .context = group_context_,
      .buf = buf,
      .aux_bufs = std::move(aux_bufs),
      .params = params,
      .caller_stream = caller_stream,
      .op_timeout = op_timeout_,
      .is_aborted =
          [this] {
            return aborted_.load(std::memory_order_acquire) ||
                   local_abort_.load(std::memory_order_acquire);
          },
      .on_error = [this](const std::string& m) { report_and_abort(m); },
      .on_terminal =
          [this] {
            inflight_.fetch_sub(1, std::memory_order_release);
            std::lock_guard<std::mutex> lk(inflight_mu_);
            inflight_cv_.notify_all();
          },
      .state = state};
  torch_spyre::distributed::spyre_global_progress_enqueue(std::move(req));
  // NOTE: seq_ is NOT bumped here. Collective ops (allreduce, broadcast,
  // gather, allgather, barrier) bump seq_ at their call sites because every
  // rank in the group issues them. send/recv are P2P and run on only the 2
  // participating ranks -- bumping seq_ here would desynchronize
  // getSequenceNumberForGroup() across the group and trigger spurious
  // "Detected mismatch between collectives on ranks" errors on the next
  // collective call.
  return c10::make_intrusive<SpyreCCLWork>(op, state, std::move(hold),
                                           std::move(result), op_timeout_);
}

/* **********************************************
 * Interface functions
 *
 * NOTE: All collectives are inherently async (non-blocking). Callers
 * passing asyncOp=false will not get synchronous completion — they must
 * call work->wait() explicitly. This matches the behavior of NCCL/Gloo.
 ********************************************** */
c10::intrusive_ptr<Work> SpyreCCLBackend::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllgatherOptions& opts) {
  DEBUGINFO("allgather: outputTensors.size=", outputTensors.size(),
            "inputTensors.size=", inputTensors.size());
  abort_guard("allgather");
  try {
    const int world = static_cast<int>(group_context_->getSize());
    const int me = static_cast<int>(group_context_->getRank());

    if (static_cast<int>(outputTensors.size()) != 1) {
      std::string _err_msg =
          "[" + getBackendName() +
          "]: Too many tensors in the output list. Expected exactly 1" +
          " Actual: " + std::to_string(outputTensors.size());
      TORCH_CHECK(false, _err_msg);
    }
    if (static_cast<int>(outputTensors[0].size()) != world) {
      std::string _err_msg =
          "[" + getBackendName() +
          "]: Incorrect output list size. The list size should be exactly " +
          std::to_string(world) +
          " Actual: " + std::to_string(outputTensors[0].size());
      TORCH_CHECK(false, _err_msg);
    }
    check_vector_tensor(inputTensors, 1, 1);
    check_vector_tensor(outputTensors[0], 1, world);

    std::vector<at::Tensor>& outs = outputTensors[0];
    at::Tensor& input = inputTensors[0];

    spyre_comms::BufferDesc input_buf = prepare_buffer_desc(input);

    // Precompute output descriptors and detect the uniform case (every output
    // slot has the same byte size as this rank's input). Uniform all-gather
    // keeps the existing host-staged Context::allgather fast path. A variable
    // (per-rank differently-sized) all-gather — the vLLM MoE all_gatherv case —
    // takes the N-broadcast decomposition below, which matches the NCCL
    // reference (pynccl.all_gatherv) and reuses the verified broadcast
    // primitive. world == 1 is trivially uniform (one output slot == input).
    std::vector<spyre_comms::BufferDesc> output_bufs;
    output_bufs.reserve(outs.size());
    bool uniform = true;
    for (auto& outputTensor : outs) {
      output_bufs.push_back(prepare_buffer_desc(outputTensor));
      if (output_bufs.back().byte_count != input_buf.byte_count) {
        uniform = false;
      }
    }

    if (uniform) {
      DEBUGINFO("allgather (uniform): enqueuing async allgather,",
                "output_bufs.size=", output_bufs.size(), "rank=", me,
                "size=", world);

      auto caller_stream = spyre::getCurrentStream(input.device());
      std::vector<at::Tensor> hold = outs;
      hold.push_back(input);
      // Collective op: every rank in the group issues this, so seq_ stays
      // in sync across ranks.
      seq_.fetch_add(1, std::memory_order_relaxed);
      return enqueue_async(OpType::ALLGATHER, input_buf, std::move(output_bufs),
                           torch_spyre::distributed::CollectiveParams{},
                           caller_stream, std::move(hold), outs);
    }

    // ── Variable-size all-gather: pairwise sendrecv rounds ──
    // broadcast is unsafe here: Broadcast_Unicast's multi-peer one-directional
    // fan-out trips the HDMA credit guard at MoE sizes
    // ((world-1)*ceil(size/2MB) > send_credit). Instead, round r pairs this
    // rank with peer p = ((r - me) % world + world) % world (the symmetric
    // matching alltoall_base uses). Each round this rank SENDS its whole
    // contribution (input, sizes[me] bytes) and RECEIVES peer p's whole
    // contribution into outs[p] (sizes[p] bytes) — an asymmetric exchange that
    // exchange_uneven decomposes into a balanced sendrecv + credit-bounded
    // one-directional remainder, every sub-leg either bidirectional
    // (guard-exempt) or under send_credit. outs[p] is a whole, caller-owned
    // tensor, so there is no dim-0 offset slicing of a tiled tensor and no
    // per-round scratch (no UAF class). All ranks iterate r = 0..world-1 in the
    // same order (deterministic cross-rank launch order); sequential rounds are
    // deadlock-free. See docs/design/allgatherv-reducescatterv-design.md.
    //
    // NOTE: this rank cannot validate that outs[p] on OTHER ranks matches rank
    // p's contribution size — like alltoall's split lists, the caller must
    // supply cross-rank-consistent sizes (vLLM passes an explicit sizes
    // vector).
    TORCH_CHECK(input_buf.byte_count == output_bufs[me].byte_count, "[",
                getBackendName(), "]: allgather output slot for this rank (",
                output_bufs[me].byte_count,
                " bytes) must match this rank's input (", input_buf.byte_count,
                " bytes)");

    // The input is SENT every round — it must be fully written before the DMA
    // reads it (A5).
    order_after_caller_stream(input);

    // Self-leg: this rank's own slot is a whole-tensor copy of input
    // (layout-safe, same shape). Fence so it lands before the caller reads
    // outs.
    if (input_buf.byte_count > 0) {
      outs[me].copy_(input);
      order_after_caller_stream(outs[me]);
    }

    std::unique_ptr<spyre_comms::WorkSchedule> last_ws;
    for (int r = 0; r < world; ++r) {
      const int p = ((r - me) % world + world) % world;
      if (p == me) {
        continue;  // own slot handled by the self-leg copy above
      }
      auto ws = exchange_uneven(input_buf, output_bufs[p], p, r);
      if (ws) {
        last_ws = std::move(ws);
      }
    }

    seq_.fetch_add(1, std::memory_order_relaxed);
    std::vector<at::Tensor> hold = outs;
    hold.push_back(input);
    if (!last_ws) {
      // Degenerate: no cross-rank bytes moved (every rank contributes 0). The
      // self copy already produced the result; synthesize a completed Work so
      // the Future/wait path is well-formed (mirrors alltoall_base).
      spyre_comms::BufferDesc self_b = prepare_buffer_desc(outs[me]);
      last_ws = group_context_->device_copy(self_b, self_b);
      last_ws->SetStreamAffinity(comm_stream_);
      last_ws->start();
      last_ws->wait();
    }
    return c10::make_intrusive<SpyreCCLWork>(
        OpType::ALLGATHER, std::move(last_ws), std::move(hold), outs,
        op_timeout_);
  }
  catch (const std::exception& e) {
    report_and_abort(std::string("allgather: ") + e.what());
    throw;
  }
  catch (...) {
    report_and_abort("allgather: unknown error");
    throw;
  }
}

c10::intrusive_ptr<Work> SpyreCCLBackend::_allgather_base(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  // Do not intend to support: It is deprecated
  // https://github.com/pytorch/pytorch/blob/62226611ded023ff1119b103ed3f540f75e38e9d/torch/csrc/distributed/c10d/Backend.hpp#L197-L209
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
  abort_guard("allreduce");
  try {
    check_vector_tensor(tensors, 1, 1);

    if (opts.reduceOp != ReduceOp::SUM) {
      std::string _err_msg = "[" + getBackendName() +
                             "]: Allreduce only supports SUM operation." +
                             " Actual: " + std::to_string(opts.reduceOp);
      TORCH_CHECK(false, _err_msg);
    }

    spyre_comms::BufferDesc buf = prepare_buffer_desc(tensors[0]);
    DEBUGINFO("allreduce: buf.host_ptr=", buf.host_ptr,
              "buf.device_addr=", buf.device_addr,
              "is_host_only=", buf.is_host_only);

    // Splittability guard: allreduce reduce-scatters the buffer across
    // ranks, so each rank needs at least one 128-byte stick. A tensor
    // smaller than world_size sticks makes LIBCOLL's split geometry round
    // the per-rank base size to 0 and abort (envelope.cpp:
    // DT_CHECK(base_size > 0)) -- an uncatchable abort on the async worker
    // thread, and a silent no-op that would corrupt in release (NDEBUG)
    // builds. Reject it here on the caller thread as a catchable error
    // instead. 128 = one fp16 stick (64 fp16 elems).
    const int world_sz = static_cast<int>(group_context_->getSize());
    constexpr size_t kStickBytes = 128;
    TORCH_CHECK(
        buf.byte_count >= static_cast<size_t>(world_sz) * kStickBytes,
        "[", getBackendName(), "]: allreduce tensor too small to split across ",
        world_sz, " ranks: have ", buf.byte_count, " bytes, need >= ",
        static_cast<size_t>(world_sz) * kStickBytes,
        " (>= 1 stick of 128 bytes per rank).");

    auto caller_stream = spyre::getCurrentStream(tensors[0].device());
    // Collective op: every rank in the group issues this, so seq_ stays in
    // sync across ranks.
    seq_.fetch_add(1, std::memory_order_relaxed);
    return enqueue_async(OpType::ALLREDUCE, buf, /*aux_bufs=*/{},
                         {.reduce_op = convert_reduce_op_type(opts.reduceOp)},
                         caller_stream,
                         /*hold=*/tensors, /*result=*/tensors);
  }
  catch (const std::exception& e) {
    report_and_abort(std::string("allreduce: ") + e.what());
    throw;
  }
  catch (...) {
    report_and_abort("allreduce: unknown error");
    throw;
  }
}

c10::intrusive_ptr<Work> SpyreCCLBackend::allreduce_coalesced(
    std::vector<at::Tensor>& tensors, const AllreduceCoalescedOptions& opts) {
  // Do not intend to support: No public interface
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

// All-to-all as N independent single-peer pairwise legs (never a multi-peer
// batch — see docs/alltoall-transport-design.md). Round r pairs this rank with
// partner p = (r - rank) mod N, a symmetric perfect matching (both ranks
// compute the same round and select each other), so each round is a set of
// independent 2-party exchanges; running the rounds serially is deadlock-free
// by construction. The p == rank round is a local device copy (no network).
// Phase 1 is correctness-first: serial, balanced (equal per-peer send/recv
// extents), stick-aligned, single-chunk buffers only;
// asymmetric/large/interleaved cases fail fast (Phase 2).
c10::intrusive_ptr<Work> SpyreCCLBackend::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllToAllOptions& opts) {
  abort_guard("alltoall");
  try {
    const int world = static_cast<int>(group_context_->getSize());
    const int me = static_cast<int>(group_context_->getRank());
    if (static_cast<int>(inputTensors.size()) != world ||
        static_cast<int>(outputTensors.size()) != world) {
      TORCH_CHECK(
          false, "[", getBackendName(),
          "]: alltoall requires input and output tensor lists of length "
          "world_size (",
          world, "); got ", inputTensors.size(), " and ", outputTensors.size());
    }
    for (auto& t : inputTensors) check_single_tensor(t);
    for (auto& t : outputTensors) check_single_tensor(t);

    // Precompute buffer descriptors and validate EVERY leg up front — before
    // any network op — so an unsupported (Phase 1: asymmetric per-peer) size
    // fails cleanly on this rank with no partial transfers already in flight
    // against peers this rank would otherwise not reach.
    std::vector<spyre_comms::BufferDesc> send_bufs;
    std::vector<spyre_comms::BufferDesc> recv_bufs;
    send_bufs.reserve(world);
    recv_bufs.reserve(world);
    for (int p = 0; p < world; ++p) {
      send_bufs.push_back(prepare_buffer_desc(inputTensors[p]));
      recv_bufs.push_back(prepare_buffer_desc(outputTensors[p]));
      if (send_bufs[p].byte_count == 0 && recv_bufs[p].byte_count == 0) {
        continue;
      }
      // Phase 1: balanced only — per-peer send extent must equal recv extent
      // (the existing sendrecv/device_copy require it). Asymmetric legs are
      // Phase 2.
      TORCH_CHECK(send_bufs[p].byte_count == recv_bufs[p].byte_count, "[",
                  getBackendName(),
                  "]: alltoall Phase 1 supports only balanced (equal per-peer "
                  "send/recv) sizes; peer ",
                  p, " send=", send_bufs[p].byte_count,
                  " bytes recv=", recv_bufs[p].byte_count,
                  " bytes. Asymmetric all-to-all is not yet supported.");
    }

    // The send buffers must be fully written before the DMA reads them (A5).
    order_after_caller_stream(inputTensors[0]);

    std::unique_ptr<spyre_comms::WorkSchedule> last_ws;
    for (int r = 0; r < world; ++r) {
      const int p = ((r - me) % world + world) % world;
      if (send_bufs[p].byte_count == 0 && recv_bufs[p].byte_count == 0) {
        continue;
      }

      std::unique_ptr<spyre_comms::WorkSchedule> ws;
      if (p == me) {
        ws = group_context_->device_copy(recv_bufs[p], send_bufs[p]);
      } else {
        ws = group_context_->sendrecv(send_bufs[p], recv_bufs[p], p, /*tag=*/r);
      }
      ws->SetStreamAffinity(comm_stream_);
      ws->start();
      ws->wait();
      last_ws = std::move(ws);
    }
    TORCH_CHECK(last_ws, "[", getBackendName(),
                "]: alltoall produced no work (all legs empty)");

    seq_.fetch_add(1, std::memory_order_relaxed);
    std::vector<at::Tensor> hold(outputTensors.begin(), outputTensors.end());
    hold.insert(hold.end(), inputTensors.begin(), inputTensors.end());
    std::vector<at::Tensor> result = outputTensors;
    return c10::make_intrusive<SpyreCCLWork>(
        OpType::ALLTOALL, std::move(last_ws), std::move(hold),
        std::move(result), op_timeout_);
  }
  catch (const std::exception& e) {
    report_and_abort(std::string("alltoall: ") + e.what());
    throw;
  }
  catch (...) {
    report_and_abort("alltoall: unknown error");
    throw;
  }
}

c10::intrusive_ptr<Work> SpyreCCLBackend::alltoall_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes, const AllToAllOptions& opts) {
  abort_guard("alltoall_base");
  try {
    check_single_tensor(inputTensor);
    check_single_tensor(outputTensor);

    const int world = static_cast<int>(group_context_->getSize());
    const int me = static_cast<int>(group_context_->getRank());

    const int64_t in_rows = inputTensor.size(0);
    const int64_t out_rows = outputTensor.size(0);
    TORCH_CHECK(in_rows > 0 && out_rows > 0, "[", getBackendName(),
                "]: alltoall_base requires non-empty tensors");

    // world == 1: trivially output = input (whole tensor, identical shape and
    // layout, so a verbatim device copy is correct). Handles the degenerate
    // single-rank case and guarantees the round loop below (world > 1) always
    // produces at least one network leg (a non-null Work to return).
    if (world == 1) {
      spyre_comms::BufferDesc in_b = prepare_buffer_desc(inputTensor);
      spyre_comms::BufferDesc out_b = prepare_buffer_desc(outputTensor);
      order_after_caller_stream(inputTensor);
      auto ws = group_context_->device_copy(out_b, in_b);
      ws->SetStreamAffinity(comm_stream_);
      ws->start();
      ws->wait();
      seq_.fetch_add(1, std::memory_order_relaxed);
      return c10::make_intrusive<SpyreCCLWork>(
          OpType::ALLTOALL_BASE, std::move(ws),
          std::vector<at::Tensor>{outputTensor, inputTensor},
          std::vector<at::Tensor>{outputTensor}, op_timeout_);
    }

    // Layout-aware per-peer decomposition. A >=2D device tensor is stored in a
    // tiled SpyreTensorLayout (device_size = [ceil(inner/64), rows, 64] for a
    // rank-2 tensor), so a logical dim-0 slice is NOT a contiguous device byte
    // range and cannot be sliced at the transport layer by raw byte offset (the
    // confirmed 2-D failure, docs/alltoall-transport-design.md §13.1). Instead
    // we let torch-spyre's own layout-aware ops carve out each peer's dim-0
    // sub-chunk as a fresh, standard-layout device tensor (clone), move it as a
    // WHOLE tensor (the proven, layout-safe path), and scatter the received
    // tensor back into the output sub-region (copy_ into a narrowed view). This
    // relies on the now-fixed dim-0 (row-major) storage_offset handling in the
    // eager device copy spyre::copy_from_d2d (was the §13.6 blocker; core fix
    // hardware-confirmed 2026-07-17). See §13.2/§13.8.
    //
    // NOTE: .clone() (NOT .contiguous()) — a dim-0 narrow of a contiguous
    // tensor is itself "contiguous" (only a storage offset), so .contiguous()
    // returns the view and prepare_buffer_desc would read the PARENT
    // allocation's composite_addr. clone() forces a fresh allocation whose
    // composite_addr IS the sub-chunk.

    // Split sizes count dim-0 elements; each is numel/size(0) elements wide.
    const int64_t in_row_elems = inputTensor.numel() / in_rows;
    const int64_t out_row_elems = outputTensor.numel() / out_rows;
    const size_t in_row_bytes = static_cast<size_t>(in_row_elems) *
                                static_cast<size_t>(inputTensor.element_size());
    const size_t out_row_bytes =
        static_cast<size_t>(out_row_elems) *
        static_cast<size_t>(outputTensor.element_size());

    // Empty split list ⇒ uniform even split across the world.
    std::vector<int64_t> in_splits = inputSplitSizes;
    std::vector<int64_t> out_splits = outputSplitSizes;
    if (in_splits.empty()) {
      TORCH_CHECK(in_rows % world == 0, "[", getBackendName(),
                  "]: alltoall_base with empty inputSplitSizes requires input "
                  "dim0 (",
                  in_rows, ") divisible by world size (", world, ")");
      in_splits.assign(world, in_rows / world);
    }
    if (out_splits.empty()) {
      TORCH_CHECK(
          out_rows % world == 0, "[", getBackendName(),
          "]: alltoall_base with empty outputSplitSizes requires output "
          "dim0 (",
          out_rows, ") divisible by world size (", world, ")");
      out_splits.assign(world, out_rows / world);
    }
    TORCH_CHECK(static_cast<int>(in_splits.size()) == world &&
                    static_cast<int>(out_splits.size()) == world,
                "[", getBackendName(),
                "]: split-size lists must have length world_size (", world,
                ")");

    // Prefix (element) offsets for each peer's dim-0 sub-chunk.
    std::vector<int64_t> in_off(world, 0);
    std::vector<int64_t> out_off(world, 0);
    for (int j = 1; j < world; ++j) {
      in_off[j] = in_off[j - 1] + in_splits[j - 1];
      out_off[j] = out_off[j - 1] + out_splits[j - 1];
    }
    TORCH_CHECK(in_off[world - 1] + in_splits[world - 1] == in_rows, "[",
                getBackendName(), "]: inputSplitSizes must sum to input dim0 (",
                in_rows, ")");
    TORCH_CHECK(out_off[world - 1] + out_splits[world - 1] == out_rows, "[",
                getBackendName(),
                "]: outputSplitSizes must sum to output dim0 (", out_rows, ")");

    // For a 1-D tensor, dim-0 IS the innermost (stick) dimension, so a dim-0
    // narrow is a stick-dim offset — only the stick-aligned case is supported
    // by the copy_from_d2d fix (non-stick-aligned innermost offsets remain an
    // xfail). For >=2D, dim-0 is NOT the stick dim, so arbitrary dim-0 offsets
    // are supported. Require 128-byte (stick) alignment only for the 1-D case.
    const bool require_stick_align =
        (inputTensor.dim() == 1 || outputTensor.dim() == 1);
    constexpr size_t kStickBytes = 128;

    // Validate EVERY leg up front — before any op — so an unsupported split
    // (asymmetric, or non-stick-aligned 1-D) fails cleanly with no partial
    // transfers in flight against peers this rank would otherwise not reach.
    for (int p = 0; p < world; ++p) {
      const size_t s_off = static_cast<size_t>(in_off[p]) * in_row_bytes;
      const size_t s_len = static_cast<size_t>(in_splits[p]) * in_row_bytes;
      const size_t r_off = static_cast<size_t>(out_off[p]) * out_row_bytes;
      const size_t r_len = static_cast<size_t>(out_splits[p]) * out_row_bytes;
      if (s_len == 0 && r_len == 0) {
        continue;
      }
      // Balanced only — the single-peer sendrecv requires equal send/recv byte
      // extents. Asymmetric (MoE) splits are Phase 2.
      TORCH_CHECK(
          s_len == r_len, "[", getBackendName(),
          "]: alltoall_base Phase 1 supports only balanced (equal "
          "per-peer send/recv) splits; peer ",
          p, " send=", s_len, " bytes recv=", r_len,
          " bytes. Asymmetric all-to-all (MoE token routing) is not yet "
          "supported.");
      if (require_stick_align) {
        TORCH_CHECK(
            s_off % kStickBytes == 0 && s_len % kStickBytes == 0 &&
                r_off % kStickBytes == 0 && r_len % kStickBytes == 0,
            "[", getBackendName(),
            "]: alltoall_base on a 1-D tensor requires 128-byte (stick) "
            "aligned per-peer offsets/sizes (dim-0 is the stick "
            "dimension); peer ",
            p, " send_off=", s_off, " send_len=", s_len, " recv_off=", r_off,
            " recv_len=", r_len);
      }
    }

    // Ensure the producing compute has landed before we read/clone the input.
    order_after_caller_stream(inputTensor);

    // Keep every per-peer send/recv tensor alive for the WHOLE collective. The
    // spyre allocator frees a device tensor's memory IMMEDIATELY on destruction
    // (spyre_allocator.cpp ReportAndDelete: USE_DEFERRED is not defined, and
    // comm_stream_ is never recordStream()'d), so letting these per-round
    // buffers destruct each iteration would return their memory to the
    // allocator and the NEXT round's clone/empty_like could reuse it while this
    // round's HDMA is still settling — nondeterministic cross-round corruption
    // (HARDWARE-CONFIRMED 2026-07-17, §13.5). Retaining them here restores the
    // long-lived-buffer invariant the list-form path already has. Freed
    // together after the final sync.
    std::vector<at::Tensor> keep_alive;
    keep_alive.reserve(static_cast<size_t>(2 * world));

    std::unique_ptr<spyre_comms::WorkSchedule> last_ws;

    for (int r = 0; r < world; ++r) {
      const int p = ((r - me) % world + world) % world;
      const int64_t in_o = in_off[p];
      const int64_t in_l = in_splits[p];
      const int64_t out_o = out_off[p];
      const int64_t out_l = out_splits[p];

      if (in_l == 0 && out_l == 0) {
        continue;
      }

      if (p == me) {
        // Local move: output[out slice] = input[in slice], layout-aware.
        outputTensor.narrow(0, out_o, out_l)
            .copy_(inputTensor.narrow(0, in_o, in_l));
        continue;
      }

      // Materialize a contiguous, standard-layout per-peer send tensor and a
      // matching receive tensor.
      at::Tensor send_p = inputTensor.narrow(0, in_o, in_l).clone();
      at::Tensor recv_p = at::empty_like(outputTensor.narrow(0, out_o, out_l));
      // send_p must be fully materialized before the HDMA reads its device
      // memory (the transfer runs on comm_stream_, not the caller stream).
      order_after_caller_stream(send_p);

      spyre_comms::BufferDesc send_buf = prepare_buffer_desc(send_p);
      spyre_comms::BufferDesc recv_buf = prepare_buffer_desc(recv_p);
      auto ws = group_context_->sendrecv(send_buf, recv_buf, p, /*tag=*/r);
      ws->SetStreamAffinity(comm_stream_);
      ws->start();
      ws->wait();

      // Scatter the received rows into the output sub-region (layout-aware),
      // then retain both buffers so their device memory is not freed/reused
      // until the whole collective completes.
      outputTensor.narrow(0, out_o, out_l).copy_(recv_p);
      keep_alive.push_back(std::move(send_p));
      keep_alive.push_back(std::move(recv_p));
      last_ws = std::move(ws);
    }

    // All caller-stream ops (self-copies, clones, scatters) must complete
    // before the retained buffers are freed on return and before the caller
    // reads output.
    order_after_caller_stream(outputTensor);

    if (!last_ws) {
      // No cross-rank legs (every peer split was 0 — a valid but degenerate
      // all-local exchange; the self copy above already produced the result).
      // Synthesize a completed Work (identity device copy) so the Future/wait
      // path is well-formed.
      spyre_comms::BufferDesc out_b = prepare_buffer_desc(outputTensor);
      last_ws = group_context_->device_copy(out_b, out_b);
      last_ws->SetStreamAffinity(comm_stream_);
      last_ws->start();
      last_ws->wait();
    }

    seq_.fetch_add(1, std::memory_order_relaxed);
    std::vector<at::Tensor> hold = {outputTensor, inputTensor};
    return c10::make_intrusive<SpyreCCLWork>(
        OpType::ALLTOALL_BASE, std::move(last_ws), std::move(hold),
        std::vector<at::Tensor>{outputTensor}, op_timeout_);
  }
  catch (const std::exception& e) {
    report_and_abort(std::string("alltoall_base: ") + e.what());
    throw;
  }
  catch (...) {
    report_and_abort("alltoall_base: unknown error");
    throw;
  }
}

c10::intrusive_ptr<Work> SpyreCCLBackend::barrier(const BarrierOptions& opts) {
  abort_guard("barrier");
  try {
    // Barrier has no tensor to key a device off of -- use the default spyre
    // device's current stream (mirrors the previous synchronous barrier(),
    // which likewise did not reference any caller tensor/device).
    auto caller_stream = spyre::getCurrentStream();
    // Collective op: every rank in the group issues this, so seq_ stays in
    // sync across ranks.
    seq_.fetch_add(1, std::memory_order_relaxed);
    return enqueue_async(OpType::BARRIER, spyre_comms::BufferDesc{},
                         /*aux_bufs=*/{},
                         torch_spyre::distributed::CollectiveParams{},
                         caller_stream, /*hold=*/{}, /*result=*/{});
  }
  catch (const std::exception& e) {
    report_and_abort(std::string("barrier: ") + e.what());
    throw;
  }
  catch (...) {
    report_and_abort("barrier: unknown error");
    throw;
  }
}

c10::intrusive_ptr<Work> SpyreCCLBackend::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  abort_guard("broadcast");
  try {
    check_vector_tensor(tensors, 1, 1);

    spyre_comms::BufferDesc buf = prepare_buffer_desc(tensors[0]);

    auto caller_stream = spyre::getCurrentStream(tensors[0].device());
    // Collective op: every rank in the group issues this, so seq_ stays in
    // sync across ranks.
    seq_.fetch_add(1, std::memory_order_relaxed);
    return enqueue_async(
        OpType::BROADCAST, buf, /*aux_bufs=*/{},
        {.root = static_cast<spyre_comms::process_id_t>(opts.rootRank)},
        caller_stream, /*hold=*/tensors, /*result=*/tensors);
  }
  catch (const std::exception& e) {
    report_and_abort(std::string("broadcast: ") + e.what());
    throw;
  }
  catch (...) {
    report_and_abort("broadcast: unknown error");
    throw;
  }
}

c10::intrusive_ptr<Work> SpyreCCLBackend::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const GatherOptions& opts) {
  abort_guard("gather");
  try {
    if (opts.rootRank == group_context_->getRank()) {
      if (static_cast<int>(outputTensors.size()) != 1) {
        std::string _err_msg =
            "[" + getBackendName() +
            "]: Too many tensors in the output list. Expected exactly 1" +
            " Actual: " + std::to_string(outputTensors.size());
        TORCH_CHECK(false, _err_msg);
      }
      if (static_cast<int>(outputTensors[0].size()) !=
          static_cast<int>(group_context_->getSize())) {
        std::string _err_msg =
            "[" + getBackendName() +
            "]: Incorrect output list size. The list size should be exactly " +
            std::to_string(group_context_->getSize()) +
            " Actual: " + std::to_string(outputTensors[0].size());
        TORCH_CHECK(false, _err_msg);
      }
    }
    check_vector_tensor(inputTensors, 1, 1);

    spyre_comms::BufferDesc input_buf = prepare_buffer_desc(inputTensors[0]);

    std::vector<spyre_comms::BufferDesc> output_bufs;
    std::vector<at::Tensor> hold = {inputTensors[0]};
    std::vector<at::Tensor> result;
    if (opts.rootRank == group_context_->getRank()) {
      for (auto& outputTensor : outputTensors[0]) {
        output_bufs.push_back(prepare_buffer_desc(outputTensor));
      }
      hold.insert(hold.end(), outputTensors[0].begin(), outputTensors[0].end());
      result = outputTensors[0];
    } else {
      result = {inputTensors[0]};
    }

    auto caller_stream = spyre::getCurrentStream(inputTensors[0].device());
    // Collective op: every rank in the group issues this, so seq_ stays in
    // sync across ranks.
    seq_.fetch_add(1, std::memory_order_relaxed);
    return enqueue_async(
        OpType::GATHER, input_buf, std::move(output_bufs),
        {.root = static_cast<spyre_comms::process_id_t>(opts.rootRank)},
        caller_stream, std::move(hold), std::move(result));
  }
  catch (const std::exception& e) {
    report_and_abort(std::string("gather: ") + e.what());
    throw;
  }
  catch (...) {
    report_and_abort("gather: unknown error");
    throw;
  }
}

c10::intrusive_ptr<Work> SpyreCCLBackend::reduce(
    std::vector<at::Tensor>& tensors, const ReduceOptions& opts) {
  abort_guard("reduce");
  try {
    check_vector_tensor(tensors, 1, 1);
    if (opts.reduceOp != ReduceOp::SUM) {
      std::string _err_msg = "[" + getBackendName() +
                             "]: Reduce only supports SUM operation." +
                             " Actual: " + std::to_string(opts.reduceOp);
      TORCH_CHECK(false, _err_msg);
    }

    spyre_comms::BufferDesc buf = prepare_buffer_desc(tensors[0]);

    auto ws = group_context_->reduce(buf, convert_reduce_op_type(opts.reduceOp),
                                     opts.rootRank);
    order_after_caller_stream(tensors[0]);
    ws->SetStreamAffinity(comm_stream_);
    ws->start();
    seq_.fetch_add(1, std::memory_order_relaxed);
    return c10::make_intrusive<SpyreCCLWork>(OpType::REDUCE, std::move(ws),
                                             tensors, tensors, op_timeout_);
  }
  catch (const std::exception& e) {
    report_and_abort(std::string("reduce: ") + e.what());
    throw;
  }
  catch (...) {
    report_and_abort("reduce: unknown error");
    throw;
  }
}

// Reduce-scatter as a pairwise sendrecv shuffle with on-device accumulation
// (docs/design/allgatherv-reducescatterv-design.md). Context::reduce is avoided
// entirely: it is host-staged (D2H all -> CPU sum -> H2D, the async-free path
// behind the earlier allreduce UAF) and each non-root SEND is exposed to the
// same one-directional credit throw as broadcast. Instead, round r pairs this
// rank with peer p = ((r - me) % world + world) % world; this rank SENDS its
// chunk destined for p (input_list[p], sizes[p] bytes) and RECEIVES p's chunk
// destined for this rank (sizes[me] bytes) into a scratch, then accumulates it
// on-device into an output seeded from this rank's own chunk (input_list[me]).
// Every transfer is a balanced-or-under-credit sub-leg (exchange_uneven), so
// the credit guard never fires; the single reused scratch is allocated once (no
// per-round alloc/free, no UAF class). All ranks iterate r = 0..world-1 in the
// same order (deterministic cross-rank launch order); sequential rounds are
// deadlock-free by construction. This is not yet a ring/recursive-halving
// reduce-scatter (that is follow-up for NCCL-competitive performance) but it is
// correct, credit-safe, and never touches the host-reduce path.
c10::intrusive_ptr<Work> SpyreCCLBackend::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  abort_guard("reduce_scatter");
  try {
    const int world = static_cast<int>(group_context_->getSize());
    const int me = static_cast<int>(group_context_->getRank());

    TORCH_CHECK(static_cast<int>(outputTensors.size()) == 1, "[",
                getBackendName(),
                "]: reduce_scatter expects exactly 1 output tensor; got ",
                outputTensors.size());
    TORCH_CHECK(static_cast<int>(inputTensors.size()) == 1, "[",
                getBackendName(),
                "]: reduce_scatter expects exactly 1 input list; got ",
                inputTensors.size());

    at::Tensor& output = outputTensors[0];
    std::vector<at::Tensor>& ins = inputTensors[0];
    TORCH_CHECK(static_cast<int>(ins.size()) == world, "[", getBackendName(),
                "]: reduce_scatter input list must have length world_size (",
                world, "); got ", ins.size());

    if (opts.reduceOp != ReduceOp::SUM) {
      std::string _err_msg = "[" + getBackendName() +
                             "]: reduce_scatter only supports SUM operation." +
                             " Actual: " + std::to_string(opts.reduceOp);
      TORCH_CHECK(false, _err_msg);
    }

    check_single_tensor(output);
    for (auto& t : ins) {
      check_single_tensor(t);
    }

    spyre_comms::BufferDesc output_buf = prepare_buffer_desc(output);

    // Precompute every chunk descriptor up front.
    std::vector<spyre_comms::BufferDesc> in_bufs;
    in_bufs.reserve(world);
    for (int i = 0; i < world; ++i) {
      in_bufs.push_back(prepare_buffer_desc(ins[i]));
    }
    TORCH_CHECK(output_buf.byte_count == in_bufs[me].byte_count, "[",
                getBackendName(), "]: reduce_scatter output (",
                output_buf.byte_count,
                " bytes) must match this rank's input chunk input_list[", me,
                "] (", in_bufs[me].byte_count, " bytes)");

    // world == 1: the single rank's own chunk is the result (no reduction).
    if (world == 1) {
      order_after_caller_stream(ins[me]);
      auto ws = group_context_->device_copy(output_buf, in_bufs[me]);
      ws->SetStreamAffinity(comm_stream_);
      ws->start();
      ws->wait();
      seq_.fetch_add(1, std::memory_order_relaxed);
      return c10::make_intrusive<SpyreCCLWork>(
          OpType::REDUCE_SCATTER, std::move(ws),
          std::vector<at::Tensor>{output, ins[me]},
          std::vector<at::Tensor>{output}, op_timeout_);
    }

    // Seed the accumulator (output) with this rank's own contribution to its
    // own chunk; the copy runs on the caller stream after the inputs' own H2D.
    // The fence then guarantees BOTH the seed and every input chunk are fully
    // written before the exchange DMAs read them (A5).
    output.copy_(ins[me]);
    order_after_caller_stream(output);

    // One reused scratch for the received peer contribution — this rank always
    // receives its own chunk's share (sizes[me] bytes) from every peer, so a
    // single sizes[me]-sized buffer serves all rounds. Allocated once (no
    // per-round alloc/free), so the free-mid-collective UAF class cannot arise.
    at::Tensor recv_scratch = at::empty_like(output);
    spyre_comms::BufferDesc scratch_buf = prepare_buffer_desc(recv_scratch);

    std::unique_ptr<spyre_comms::WorkSchedule> last_ws;
    for (int r = 0; r < world; ++r) {
      const int p = ((r - me) % world + world) % world;
      if (p == me) {
        continue;  // own chunk already seeded into output
      }
      // Send my chunk destined for p (in_bufs[p], sizes[p]); receive p's chunk
      // destined for me into the scratch (sizes[me]).
      auto ws = exchange_uneven(in_bufs[p], scratch_buf, p, r);
      if (ws) {
        last_ws = std::move(ws);
      }
      // Accumulate the received contribution on-device. exchange_uneven already
      // host-waited, so recv_scratch is fully written before this add reads it;
      // the fence after ensures the add completes before the next round's recv
      // reuses the scratch.
      output.add_(recv_scratch);
      order_after_caller_stream(output);
    }

    seq_.fetch_add(1, std::memory_order_relaxed);
    std::vector<at::Tensor> hold = {output, recv_scratch};
    hold.insert(hold.end(), ins.begin(), ins.end());
    if (!last_ws) {
      // Degenerate: no cross-rank bytes moved (every chunk 0). The seed already
      // produced the result; synthesize a completed Work so the Future/wait
      // path is well-formed (mirrors alltoall_base / all_gatherv).
      last_ws = group_context_->device_copy(output_buf, output_buf);
      last_ws->SetStreamAffinity(comm_stream_);
      last_ws->start();
      last_ws->wait();
    }
    return c10::make_intrusive<SpyreCCLWork>(
        OpType::REDUCE_SCATTER, std::move(last_ws), std::move(hold),
        std::vector<at::Tensor>{output}, op_timeout_);
  }
  catch (const std::exception& e) {
    report_and_abort(std::string("reduce_scatter: ") + e.what());
    throw;
  }
  catch (...) {
    report_and_abort("reduce_scatter: unknown error");
    throw;
  }
}

c10::intrusive_ptr<Work> SpyreCCLBackend::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::send(std::vector<at::Tensor>& tensors,
                                               int dstRank, int tag) {
  abort_guard("send");
  try {
    check_vector_tensor(tensors, 1, 1);

    spyre_comms::BufferDesc buf = prepare_buffer_desc(tensors[0]);

    auto caller_stream = spyre::getCurrentStream(tensors[0].device());
    // P2P op: only the 2 participating ranks issue this. Do NOT bump seq_
    // here -- it would desynchronize getSequenceNumberForGroup() across the
    // group (see enqueue_async's NOTE above).
    return enqueue_async(
        OpType::SEND, buf, /*aux_bufs=*/{},
        {.peer = static_cast<spyre_comms::process_id_t>(dstRank), .tag = tag},
        caller_stream, /*hold=*/tensors, /*result=*/tensors);
  }
  catch (const std::exception& e) {
    report_and_abort(std::string("send: ") + e.what());
    throw;
  }
  catch (...) {
    report_and_abort("send: unknown error");
    throw;
  }
}

c10::intrusive_ptr<Work> SpyreCCLBackend::recv(std::vector<at::Tensor>& tensors,
                                               int srcRank, int tag) {
  abort_guard("recv");
  try {
    check_vector_tensor(tensors, 1, 1);

    spyre_comms::BufferDesc buf = prepare_buffer_desc(tensors[0]);

    auto caller_stream = spyre::getCurrentStream(tensors[0].device());
    // P2P op: only the 2 participating ranks issue this. Do NOT bump seq_
    // here -- it would desynchronize getSequenceNumberForGroup() across the
    // group (see enqueue_async's NOTE above).
    return enqueue_async(
        OpType::RECV, buf, /*aux_bufs=*/{},
        {.peer = static_cast<spyre_comms::process_id_t>(srcRank), .tag = tag},
        caller_stream, /*hold=*/tensors, /*result=*/tensors);
  }
  catch (const std::exception& e) {
    report_and_abort(std::string("recv: ") + e.what());
    throw;
  }
  catch (...) {
    report_and_abort("recv: unknown error");
    throw;
  }
}

c10::intrusive_ptr<Work> SpyreCCLBackend::recvAnysource(
    std::vector<at::Tensor>& tensors, int tag) {
  // Do not intend to support: Too much protocol overhead, and not commonly used
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Backend> SpyreCCLBackend::createSpyreCCLBackend(
    const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
    const std::chrono::duration<float>& timeout) {
  // c10d hands the process-group timeout in as seconds (duration<float>).
  // Convert to whole milliseconds for SpyreCCLWork::wait(). A non-positive
  // value means "no PG timeout" — normalize it to kUnsetTimeout so wait()
  // treats it as "block indefinitely" rather than a zero-length deadline.
  const auto op_timeout =
      timeout.count() > 0.0f
          ? std::chrono::duration_cast<std::chrono::milliseconds>(timeout)
          : kUnsetTimeout;
  return c10::make_intrusive<SpyreCCLBackend>(store, rank, size, op_timeout);
}

void SpyreCCLBackend::abort_guard(const char* op) {
  if (aborted_.load(std::memory_order_acquire)) {
    TORCH_CHECK(false, "[", getBackendName(), "]: backend has been aborted; ",
                op, " may not be launched");
  }
}

void SpyreCCLBackend::abort() {
  // There is no spyre_comms-level primitive to cancel a WorkSchedule that is
  // already running on the hardware, so abort() cannot forcibly interrupt an
  // in-flight collective. What it can do safely is stop new collectives from
  // being launched (abort_guard checks this flag) so a failing group tears
  // down instead of issuing more work. True mid-flight cancellation needs a
  // new comms API (reported to the runtime owners).
  aborted_.store(true, std::memory_order_release);

  // Fail-fast (Phase 0): half-close the shared OOB sockets so any peer blocked
  // in an OOB read wakes immediately via EOF, rather than waiting for this
  // rank's normal (barrier-first) finalize teardown. This is a no-op unless we
  // are the last live spyre-comms reference, so it cannot break a healthy
  // sibling ProcessGroup that still shares the singleton sockets. It does NOT
  // finalize the library or close fds — the destructor's finalize_library()
  // still does that. Idempotent, so a repeated abort() is safe.
  spyre_comms::abort_oob_connections();

  DEBUGINFO("# [Spyre CCL]: abort() requested for ", getBackendName());
}

void SpyreCCLBackend::shutdown() {
  DEBUGINFO("# [Spyre CCL]: shutdown() requested for ", getBackendName());
  abort();
}

void SpyreCCLBackend::report_and_abort(const std::string& msg) {
  bool expected = false;
  if (!aborted_.compare_exchange_strong(expected, true,
                                        std::memory_order_acq_rel)) {
    // Already aborted -- by this rank's own earlier failure, a peer's, or an
    // explicit abort()/shutdown(). Permanent one-way ratchet (matches
    // ProcessGroupNCCL: a faulted communicator is never revived), so there
    // is nothing more to do.
    return;
  }

  DEBUGINFO("# [Spyre CCL]: report_and_abort() for ", getBackendName(), ": ",
            msg);

  if (store_) {
    try {
      store_->set(error_store_key_, std::to_string(getRank()) + ":" + msg);
    }
    catch (...) {
      // Best-effort: a failed Store write must not mask the original
      // failure this call exists to report.
    }
  }

  // Set the reason BEFORE flipping the shutdown flag: setShutdownReason()
  // and setShutdown() use separate synchronization primitives (a
  // mutex-guarded string vs. an atomic bool), so this ordering is what
  // guarantees any thread that observes needsShutdown()==true also sees the
  // reason already committed, not a stale/empty one.
  if (comm_stream_) {
    comm_stream_->setShutdownReason(msg);
    comm_stream_->setShutdown(true);
  }

  // Fail-fast (Phase 0, unchanged): half-close the shared OOB sockets so any
  // peer blocked in an OOB read wakes immediately via EOF. Idempotent.
  spyre_comms::abort_oob_connections();
}

void SpyreCCLBackend::watchdog_loop() {
  while (!watchdog_stop_.load(std::memory_order_acquire)) {
    if (aborted_.load(std::memory_order_acquire)) {
      // Already aborted (locally or by a peer) -- the outcome is fixed;
      // idle until the destructor joins us instead of busy-polling a Store
      // that can no longer change anything.
      std::this_thread::sleep_for(kWatchdogPollInterval);
      continue;
    }

    bool peer_failed = false;
    try {
      peer_failed = store_ && store_->check({error_store_key_});
    }
    catch (...) {
      // Transient Store errors (e.g. during teardown) are not actionable
      // here; just retry on the next poll.
    }

    if (peer_failed) {
      std::string detail =
          "(peer failure detected, but could not read detail from Store)";
      try {
        detail = store_->get_to_str(error_store_key_);
      }
      catch (...) {
        // Keep the fallback message above.
      }
      report_and_abort("peer-reported failure: " + detail);
      continue;
    }

    std::this_thread::sleep_for(kWatchdogPollInterval);
  }
}

/***********************************************
 * Wrapper Work for the Sypre Collective Library
 ***********************************************/
SpyreCCLWork::SpyreCCLWork(
    OpType opType, std::shared_ptr<torch_spyre::distributed::WorkState> state,
    std::vector<at::Tensor> hold_tensors,
    std::vector<at::Tensor> result_tensors,
    std::chrono::milliseconds default_timeout)
    : Work(-1, opType),
      future_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))),
      state_(std::move(state)),
      hold_tensors_(std::move(hold_tensors)),
      result_tensors_(std::move(result_tensors)),
      default_timeout_(default_timeout) {}

SpyreCCLWork::SpyreCCLWork(OpType opType,
                           std::unique_ptr<spyre_comms::WorkSchedule> ws,
                           std::vector<at::Tensor> hold_tensors,
                           std::vector<at::Tensor> result_tensors,
                           std::chrono::milliseconds default_timeout)
    : Work(-1, opType),
      future_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))),
      work_schedule_(std::move(ws)),
      hold_tensors_(std::move(hold_tensors)),
      result_tensors_(std::move(result_tensors)),
      default_timeout_(default_timeout) {}

SpyreCCLWork::~SpyreCCLWork() {
  if (state_) {
    // Single-driver rule: the async progress worker is the sole owner/driver
    // of state_->ws once published. This destructor never touches ws -- it
    // only observes state_->state via the cv and, for pre-launch states,
    // requests cancellation. Waiting here (rather than detaching) keeps
    // hold_tensors_ alive until the worker has actually reached a terminal
    // state, so an in-flight DMA never reads/writes freed tensor storage.
    std::unique_lock<std::mutex> lk(state_->m);
    switch (state_->state) {
      case torch_spyre::distributed::ProgressState::ENQUEUED:
      case torch_spyre::distributed::ProgressState::BUILDING:
        // If still pre-launch we can cancel; but BUILDING means the worker
        // may be mid-build. Safest uniform rule: mark cancelled, then wait
        // for terminal.
        state_->cancelled = true;
        [[fallthrough]];
      case torch_spyre::distributed::ProgressState::LAUNCHED:
        state_->cv.wait(lk, [&] {
          return torch_spyre::distributed::is_terminal(state_->state);
        });
        break;
      case torch_spyre::distributed::ProgressState::DONE_SUCCESS:
      case torch_spyre::distributed::ProgressState::DONE_ERROR:
        break;
    }
  } else {
    // If the Work is destroyed while the transfer is still reading/writing
    // the held tensors' device memory, releasing hold_tensors_ would free
    // storage out from under an in-flight DMA (use-after-free). Drain the
    // schedule first.
    //
    // Key this on the schedule's real state (query()), not completed_: a
    // wait() that hit its timeout marks completed_ = true while the DMA is
    // still live, so completed_ alone is not sufficient to prove it is safe
    // to free.
    // (Caveat: if the peer is dead this drain can block during teardown --
    // the lesser evil vs. memory corruption. A comms-level cancel would let
    // us abort instead of drain.)
    if (work_schedule_ && !work_schedule_->query()) {
      try {
        work_schedule_->synchronize();
      }
      catch (...) {
        // Destructors must not throw; a failed transfer is surfaced via
        // wait().
      }
    }
  }
}

void SpyreCCLWork::finish_success() {
  c10::List<at::Tensor> outputs;
  outputs.reserve(result_tensors_.size());
  for (const auto& t : result_tensors_) {
    outputs.push_back(t);
  }
  future_->markCompleted(c10::IValue(std::move(outputs)));
}

void SpyreCCLWork::finish_error(const std::string& msg) {
  future_->setError(std::make_exception_ptr(std::runtime_error(msg)));
}

bool SpyreCCLWork::isCompleted() {
  if (completed_.load(std::memory_order_acquire)) return true;
  if (state_) {
    // Gate on the shared WorkState published by the async progress worker.
    // Never call ws->query() here -- the worker is the sole driver of ws
    // (single-driver rule); this Work only observes
    // state_->state/error_reason under state_->m.
    torch_spyre::distributed::ProgressState s;
    std::string reason;
    {
      std::lock_guard<std::mutex> lk(state_->m);
      s = state_->state;
      reason = state_->error_reason;
    }
    if (!torch_spyre::distributed::is_terminal(s)) return false;
    bool expected = false;
    if (completed_.compare_exchange_strong(expected, true,
                                           std::memory_order_acq_rel)) {
      // Reflect the real terminal state: only complete the Future on genuine
      // success; on DONE_ERROR propagate the failure (A7). Prefer the
      // recorded error_reason (e.g. a cross-rank peer failure signaled via
      // the process group's watchdog) over the generic message when one was
      // set.
      if (s == torch_spyre::distributed::ProgressState::DONE_ERROR) {
        finish_error(reason.empty()
                         ? "[SpyreCCL]: collective completed with DONE_ERROR"
                         : "[SpyreCCL]: collective aborted: " + reason);
      } else {
        finish_success();
      }
    }
    return true;
  } else {
    if (work_schedule_ && work_schedule_->query()) {
      bool expected = false;
      if (completed_.compare_exchange_strong(expected, true,
                                             std::memory_order_acq_rel)) {
        // Reflect the real terminal state: only complete the Future on
        // genuine success; on DONE_ERROR propagate the failure (A7). Prefer
        // the schedule's shutdown reason (e.g. a cross-rank peer failure
        // signaled via the process group's watchdog) over the generic
        // message when one was recorded -- see
        // report_and_abort()/getShutdownReason().
        if (work_schedule_->getState() ==
            spyre_comms::WorkScheduleState::State::DONE_ERROR) {
          std::string reason = work_schedule_->getShutdownReason();
          finish_error(reason.empty()
                           ? "[SpyreCCL]: collective completed with DONE_ERROR"
                           : "[SpyreCCL]: collective aborted: " + reason);
        } else {
          finish_success();
        }
      }
    }
    return completed_.load(std::memory_order_acquire);
  }
}

bool SpyreCCLWork::isSuccess() const {
  if (state_) {
    std::lock_guard<std::mutex> lk(state_->m);
    return state_->state != torch_spyre::distributed::ProgressState::DONE_ERROR;
  }
  if (!work_schedule_) return true;
  return work_schedule_->getState() !=
         spyre_comms::WorkScheduleState::State::DONE_ERROR;
}

bool SpyreCCLWork::wait(std::chrono::milliseconds timeout) {
  // Timeout precedence (matches the c10d convention): an explicit positive
  // per-call timeout always wins; otherwise fall back to the process-group
  // default captured from init_process_group(timeout=...). If neither is set,
  // block indefinitely. This is why init_process_group(timeout=...) now has an
  // effect on this backend where it previously did not.
  const std::chrono::milliseconds effective_timeout =
      (timeout != kUnsetTimeout && timeout.count() > 0) ? timeout
                                                        : default_timeout_;
  if (state_) {
    using torch_spyre::distributed::is_terminal;
    using torch_spyre::distributed::ProgressState;

    std::unique_lock<std::mutex> lk(state_->m);
    if (!is_terminal(state_->state)) {
      // kUnsetTimeout (or a non-positive value) means "block indefinitely".
      if (effective_timeout == kUnsetTimeout ||
          effective_timeout.count() <= 0) {
        state_->cv.wait(lk, [&] { return is_terminal(state_->state); });
      } else if (!state_->cv.wait_for(lk, effective_timeout, [&] {
                   return is_terminal(state_->state);
                 })) {
        // Timed out without reaching a terminal state.
        lk.unlock();
        bool expected = false;
        if (completed_.compare_exchange_strong(expected, true,
                                               std::memory_order_acq_rel)) {
          finish_error("[SpyreCCL]: collective wait timed out after " +
                       std::to_string(effective_timeout.count()) + " ms");
        }
        throw std::runtime_error(
            "[SpyreCCL]: collective wait timed out after " +
            std::to_string(effective_timeout.count()) + " ms");
      }
    }
    const bool err = state_->state == ProgressState::DONE_ERROR;
    const std::string reason = state_->error_reason;
    lk.unlock();

    bool expected = false;
    if (completed_.compare_exchange_strong(expected, true,
                                           std::memory_order_acq_rel)) {
      if (err) {
        finish_error(reason.empty()
                         ? "[SpyreCCL]: collective completed with an error"
                         : "[SpyreCCL]: collective aborted: " + reason);
      } else {
        finish_success();
      }
    }

    // c10d contract: wait() surfaces failures by throwing, not by returning
    // false. A false return is reserved for a timeout that did not complete.
    if (err) {
      throw std::runtime_error(
          reason.empty() ? "[SpyreCCL]: collective completed with an error"
                         : "[SpyreCCL]: collective aborted: " + reason);
    }
    return true;
  }

  // Sync path: this Work owns and drives work_schedule_ directly.
  if (!completed_.load(std::memory_order_acquire) && work_schedule_) {
    // kUnsetTimeout (or a non-positive value) means "block indefinitely".
    if (effective_timeout == kUnsetTimeout || effective_timeout.count() <= 0) {
      work_schedule_->wait();
    } else {
      // The underlying WorkSchedule::wait() has no deadline, so poll query()
      // until the timeout elapses. This bounds the wait so a dead peer cannot
      // hang the caller forever (P1). TODO: replace with a native timed wait
      // once spyre_comms exposes one.
      //
      // This is, in practice, the branch actually taken on essentially every
      // call: c10d's default process-group timeout (30 minutes) is a
      // positive value, so effective_timeout is essentially never
      // kUnsetTimeout even when the caller requests none explicitly.
      // Confirmed via Phase 1a hang investigation (2026-07-14): the
      // work_schedule_->wait() branch above was never observed to run.
      //
      // needsShutdown() MUST also be checked here, not just the deadline:
      // report_and_abort() flags comm_stream_ for shutdown (a local error,
      // or a cross-rank peer failure signaled via the Store watchdog), but
      // query() only checks in_flight_==0 -- it has no way to observe that
      // flag. Without this check, a rank blocked in this exact loop (e.g.
      // waiting on data that will never arrive because the peer failed
      // before ever sending) would poll query() forever until the full PG
      // timeout, never noticing the peer's failure at all.
      const auto deadline =
          std::chrono::steady_clock::now() + effective_timeout;
      while (!work_schedule_->query()) {
        if (work_schedule_->needsShutdown()) {
          // Fall through via the schedule's own wait(): now fast (returns
          // promptly once needsShutdown() is observed -- see
          // RuntimeStream::synchronize()), and it correctly transitions
          // current_state_ to DONE_ERROR, which the shared completion
          // handling below (and any later isSuccess()/wait() call) relies
          // on -- an inline throw here would leave that state stale.
          work_schedule_->wait();
          break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          bool expected = false;
          if (completed_.compare_exchange_strong(expected, true,
                                                 std::memory_order_acq_rel)) {
            finish_error("[SpyreCCL]: collective timed out after " +
                         std::to_string(effective_timeout.count()) + " ms");
          }
          throw std::runtime_error(
              "[SpyreCCL]: collective wait timed out after " +
              std::to_string(effective_timeout.count()) + " ms");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    bool expected = false;
    if (completed_.compare_exchange_strong(expected, true,
                                           std::memory_order_acq_rel)) {
      if (work_schedule_->getState() ==
          spyre_comms::WorkScheduleState::State::DONE_ERROR) {
        std::string reason = work_schedule_->getShutdownReason();
        finish_error(reason.empty()
                         ? "[SpyreCCL]: collective completed with DONE_ERROR"
                         : "[SpyreCCL]: collective aborted: " + reason);
      } else {
        finish_success();
      }
    }
  }

  // c10d contract: wait() surfaces failures by throwing, not by returning
  // false. A false return is reserved for a timeout that did not complete.
  // Prefer the schedule's shutdown reason when one was recorded (see
  // isCompleted() above for why this can be empty).
  if (!isSuccess()) {
    std::string reason =
        work_schedule_ ? work_schedule_->getShutdownReason() : std::string();
    throw std::runtime_error(
        reason.empty() ? "[SpyreCCL]: collective completed with an error"
                       : "[SpyreCCL]: collective aborted: " + reason);
  }
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> SpyreCCLWork::getFuture() {
  return future_;
}

}  // namespace c10d
