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

#include <iostream>
#include <memory>
#include <string>
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

/***********************************************
 * Wrapper Backend for the Sypre Collective Library
 ***********************************************/
SpyreCCLBackend::SpyreCCLBackend(const c10::intrusive_ptr<::c10d::Store>& store,
                                 int rank, int size)
    : Backend(rank, size), group_context_(nullptr), comm_stream_(nullptr) {
  DEBUGINFO("# [Spyre CCL]: Constructor for ", getBackendName());

  /*
   * Start the communication library
   * Pass it the shared runtime library handle, and default stream.
   */
  spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                  spyre::getDefaultStreamRuntimeHandle());
  group_context_ = spyre_comms::get_world_context();
  if (nullptr == group_context_) {
    std::string _err_msg =
        "[" + getBackendName() + "]: Failed to capture the world context";
    throw std::runtime_error(_err_msg);
  }

  // Use the dedicated comm stream so collectives run independently
  // of the compute stream, enabling compute/communication overlap.
  comm_stream_ = spyre_comms::get_comm_stream();
}

SpyreCCLBackend::~SpyreCCLBackend() {
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
  if (ctx == nullptr) {
    TORCH_CHECK(false,
                "prepare_buffer_desc: get_context() returned NULL — tensor has "
                "no device address");
  }

  // Compute dtype directly from PyTorch scalar type
  auto str_type = spyre::torchScalarToString[input_tensor.scalar_type()];
  const auto [sen_dtype_cpu, _sen_dtype_dev] =
      convert_string_to_datatype_pair(str_type);

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
  // Add check for spyre device tensor
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
        " Actual: " + std::to_string(outputTensors.size());
    TORCH_CHECK(false, _err_msg);
  }
  check_vector_tensor(inputTensors, 1, 1);

  spyre_comms::BufferDesc input_buf = prepare_buffer_desc(inputTensors[0]);

  std::vector<spyre_comms::BufferDesc> output_bufs;
  for (auto& outputTensor : outputTensors[0]) {
    output_bufs.push_back(prepare_buffer_desc(outputTensor));
  }

  DEBUGINFO("allgather: calling group_context_->allgather,",
            "output_bufs.size=", output_bufs.size(),
            "rank=", group_context_->getRank(),
            "size=", group_context_->getSize());

  auto ws = group_context_->allgather(output_bufs, input_buf);

  DEBUGINFO("allgather: ws returned, starting");

  ws->SetStreamAffinity(comm_stream_);
  ws->start();

  DEBUGINFO("allgather: ws started, returning Work");

  seq_.fetch_add(1, std::memory_order_relaxed);
  return c10::make_intrusive<SpyreCCLWork>(OpType::ALLGATHER, std::move(ws));
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

  auto ws =
      group_context_->allreduce(buf, convert_reduce_op_type(opts.reduceOp));
  ws->SetStreamAffinity(comm_stream_);
  ws->start();
  seq_.fetch_add(1, std::memory_order_relaxed);
  return c10::make_intrusive<SpyreCCLWork>(OpType::ALLREDUCE, std::move(ws));
}

c10::intrusive_ptr<Work> SpyreCCLBackend::allreduce_coalesced(
    std::vector<at::Tensor>& tensors, const AllreduceCoalescedOptions& opts) {
  // Do not intend to support: No public interface
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllToAllOptions& opts) {
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::alltoall_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes, const AllToAllOptions& opts) {
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::barrier(const BarrierOptions& opts) {
  auto ws = group_context_->barrier();
  ws->SetStreamAffinity(comm_stream_);
  ws->start();
  seq_.fetch_add(1, std::memory_order_relaxed);
  return c10::make_intrusive<SpyreCCLWork>(OpType::BARRIER, std::move(ws));
}

c10::intrusive_ptr<Work> SpyreCCLBackend::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  check_vector_tensor(tensors, 1, 1);

  spyre_comms::BufferDesc buf = prepare_buffer_desc(tensors[0]);

  auto ws = group_context_->broadcast(buf, opts.rootRank);
  ws->SetStreamAffinity(comm_stream_);
  ws->start();
  seq_.fetch_add(1, std::memory_order_relaxed);
  return c10::make_intrusive<SpyreCCLWork>(OpType::BROADCAST, std::move(ws));
}

c10::intrusive_ptr<Work> SpyreCCLBackend::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const GatherOptions& opts) {
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
          " Actual: " + std::to_string(outputTensors.size());
      TORCH_CHECK(false, _err_msg);
    }
  }
  check_vector_tensor(inputTensors, 1, 1);

  spyre_comms::BufferDesc input_buf = prepare_buffer_desc(inputTensors[0]);

  std::vector<spyre_comms::BufferDesc> output_bufs;
  if (opts.rootRank == group_context_->getRank()) {
    for (auto& outputTensor : outputTensors[0]) {
      output_bufs.push_back(prepare_buffer_desc(outputTensor));
    }
  }

  auto ws = group_context_->gather(output_bufs, input_buf, opts.rootRank);
  ws->SetStreamAffinity(comm_stream_);
  ws->start();
  seq_.fetch_add(1, std::memory_order_relaxed);
  return c10::make_intrusive<SpyreCCLWork>(OpType::GATHER, std::move(ws));
}

c10::intrusive_ptr<Work> SpyreCCLBackend::reduce(
    std::vector<at::Tensor>& tensors, const ReduceOptions& opts) {
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
  ws->SetStreamAffinity(comm_stream_);
  ws->start();
  seq_.fetch_add(1, std::memory_order_relaxed);
  return c10::make_intrusive<SpyreCCLWork>(OpType::REDUCE, std::move(ws));
}

c10::intrusive_ptr<Work> SpyreCCLBackend::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Work> SpyreCCLBackend::send(std::vector<at::Tensor>& tensors,
                                               int dstRank, int tag) {
  check_vector_tensor(tensors, 1, 1);

  spyre_comms::BufferDesc buf = prepare_buffer_desc(tensors[0]);

  auto ws = group_context_->send(buf, dstRank, tag);
  ws->SetStreamAffinity(comm_stream_);
  ws->start();
  // P2P operations must NOT increment the collective sequence counter.
  // Only ranks participating in send/recv call this method, so incrementing
  // would desynchronize the counter across ranks, causing "Detected mismatch
  // between collectives on ranks" errors on the next collective call.
  return c10::make_intrusive<SpyreCCLWork>(OpType::SEND, std::move(ws));
}

c10::intrusive_ptr<Work> SpyreCCLBackend::recv(std::vector<at::Tensor>& tensors,
                                               int srcRank, int tag) {
  check_vector_tensor(tensors, 1, 1);

  spyre_comms::BufferDesc buf = prepare_buffer_desc(tensors[0]);

  auto ws = group_context_->recv(buf, srcRank, tag);
  ws->SetStreamAffinity(comm_stream_);
  ws->start();
  // P2P operations must NOT increment the collective sequence counter.
  // See comment in send() above.
  return c10::make_intrusive<SpyreCCLWork>(OpType::RECV, std::move(ws));
}

c10::intrusive_ptr<Work> SpyreCCLBackend::recvAnysource(
    std::vector<at::Tensor>& tensors, int tag) {
  // Do not intend to support: Too much protocol overhead, and not commonly used
  throw SpyreCCLNotSupportedException(getBackendName(), __func__);
}

c10::intrusive_ptr<Backend> SpyreCCLBackend::createSpyreCCLBackend(
    const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<SpyreCCLBackend>(store, rank, size);
}

/***********************************************
 * Wrapper Work for the Sypre Collective Library
 ***********************************************/
SpyreCCLWork::SpyreCCLWork(OpType opType,
                           std::unique_ptr<spyre_comms::WorkSchedule> ws)
    : Work(-1, opType),
      future_(c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()))),
      work_schedule_(std::move(ws)) {}

bool SpyreCCLWork::isCompleted() {
  if (completed_) return true;
  if (work_schedule_ && work_schedule_->query()) {
    completed_ = true;
    future_->markCompleted();
  }
  return completed_;
}

bool SpyreCCLWork::isSuccess() const {
  if (!work_schedule_) return true;
  return work_schedule_->getState() !=
         spyre_comms::WorkScheduleState::State::DONE_ERROR;
}

bool SpyreCCLWork::wait(std::chrono::milliseconds timeout) {
  if (completed_) return isSuccess();
  if (!work_schedule_) return true;
  work_schedule_->wait();
  completed_ = true;
  future_->markCompleted();
  return isSuccess();
}

c10::intrusive_ptr<c10::ivalue::Future> SpyreCCLWork::getFuture() {
  return future_;
}

}  // namespace c10d
