/*
 * Copyright 2025 The Torch-Spyre Authors.
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

#include "spyre_stream.h"

#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "flex/flex.hpp"
#include "logging.h"
#include "module.h"
#include "spyre_allocator.h"
#include "spyre_guard.h"
#include "spyre_mem.h"
#include "spyre_tensor_impl.h"

namespace spyre {
namespace {

// TODO(tmhoangt): torch-spyre manages the pool and mapping; flex runtime just
// creates/destroys individual streams when asked.

// Global stream pool (shared across all threads)
struct StreamPool {
  std::mutex mutex;

  // Per-device stream pools
  std::unordered_map<c10::DeviceIndex, std::vector<c10::StreamId>>
      low_priority_streams;
  std::unordered_map<c10::DeviceIndex, std::vector<c10::StreamId>>
      high_priority_streams;

  // Round-robin indices
  std::unordered_map<c10::DeviceIndex, size_t> next_low_priority_idx;
  std::unordered_map<c10::DeviceIndex, size_t> next_high_priority_idx;

  // Mapping from c10::StreamId to flex::RuntimeStream*
  std::unordered_map<c10::StreamId, flex::RuntimeStream*> stream_handle_map;

  // Per-device initialization flags
  std::unordered_map<c10::DeviceIndex, std::once_flag> device_init_flags;
};

StreamPool& getStreamPool() {
  static StreamPool pool;
  return pool;
}

thread_local std::unordered_map<c10::DeviceIndex, c10::StreamId>
    current_streams;

}  // anonymous namespace

// Stream pool configuration
// Per device:
// - Stream 0: Default stream (always available, priority 0)
// - Streams 1-32: Low priority streams (priority 0)
// - Streams 33-64: High priority streams (priority -1)
constexpr int kStreamsPerDevice = 32;
constexpr int kHighPriorityStreamsPerDevice = 32;

// Constructor
SpyreStream::SpyreStream()
    : stream_(getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1,
                                           SpyreGuardImpl::tls_idx))
                  .unwrap()) {}
SpyreStream::SpyreStream(c10::Stream stream) : stream_(stream) {
  TORCH_CHECK(stream_.device_type() == c10::DeviceType::PrivateUse1,
              "SpyreStream requires PrivateUse1 device type, got ",
              stream_.device_type());
}

c10::StreamId SpyreStream::id() const {
  return stream_.id();
}

c10::Device SpyreStream::device() const {
  return stream_.device();
}

int SpyreStream::priority() const {
  // Determine priority from stream ID
  if (id() <= kStreamsPerDevice) {
    return 0;  // Low priority: stream 0 (default) and streams 1-32
  } else {
    return -1;
  }
}

bool SpyreStream::query() const {
  c10::DeviceGuard guard(stream_.device());

  DEBUGINFO("SpyreStream::query() - stream ", id(), " on device ",
            device().index());

  flex::RuntimeStream* handle = getRuntimeHandle();
  return handle->query();
}

void SpyreStream::synchronize() const {
  c10::DeviceGuard guard(stream_.device());

  DEBUGINFO("SpyreStream::synchronize() - stream ", id(), " on device ",
            device().index());

  flex::RuntimeStream* handle = getRuntimeHandle();
  handle->synchronize();
}

void SpyreStream::synchronize(std::chrono::milliseconds timeout) const {
  c10::DeviceGuard guard(stream_.device());

  DEBUGINFO("SpyreStream::synchronize(timeout=", timeout.count(),
            "ms) - stream ", id(), " on device ", device().index());

  if (timeout.count() < 0) {
    // A negative timeout is a caller bug, not a wait request — refuse rather
    // than treat it as an immediate timeout.
    TORCH_CHECK(false, "SpyreStream::synchronize: negative timeout (",
                timeout.count(), "ms)");
  }

  flex::RuntimeStream* handle = getRuntimeHandle();

  // Poll query() on a fixed cadence rather than busy-spinning. The total wait
  // is bounded by timeout + one poll interval.
  constexpr auto poll_interval = std::chrono::milliseconds(1);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!handle->query()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      // TODO: flex::RuntimeStream exposes no queuedInFlight() or streamId()
      // (the in-flight count is the private in_flight_ atomic and streams have
      // no id of their own — see flex/runtime_stream/runtime_stream.hpp).
      // Re-enable these diagnostics if flex grows public accessors:
      //   const unsigned queued = handle->queuedInFlight();
      //   ... " (flex stream ", handle->streamId(), ") on device " ...
      TORCH_CHECK(false,
                  "SpyreStreamSynchronizeTimeout: stream ", id(),
                  " on device ",
                  device().index(), " did not become idle within ",
                  timeout.count(), "ms (still busy at the poll deadline). ",
                  "FIFO state: query()=busy; the in-flight op count is not "
                  "exposed by the stream API. The stream is NOT reset — the "
                  "operation may still be running; a subsequent unbounded "
                  "synchronize() or device reset is required to reclaim it.");
    }
    std::this_thread::sleep_for(poll_interval);
  }
}

c10::Stream SpyreStream::unwrap() const {
  return stream_;
}

void SpyreStream::copyProgramAsync(void* prog_cpu_ptr,
                                   const flex::CompositeAddress* device_address,
                                   uint64_t program_bytes) const {
  // Transfer exactly the program's own byte count. The device allocation may be
  // rounded up beyond program_bytes; copying total_size() here would read past
  // the end of the host program buffer, so the explicit count is bound to the
  // program size and must fit inside the allocation.
  TORCH_CHECK(program_bytes <= device_address->total_size(),
              "Program byte count (", program_bytes,
              ") exceeds device allocation (", device_address->total_size(),
              ")");
  copyAsyncImpl(prog_cpu_ptr, device_address, nullptr, true, program_bytes);
}

void SpyreStream::copyAsync(const at::Tensor& src,
                            const at::Tensor& dst) const {
  DEBUGINFO("src (", src.scalar_type(), ") is on:", src.device());
  DEBUGINFO("dst (", dst.scalar_type(), ") on:", dst.device());

  // TODO(tmhoangt): add type conversion node
  // Type checking - no type conversion support yet
  TORCH_CHECK(
      src.scalar_type() == dst.scalar_type(),
      "Spyre backend does not support type conversion yet during copy.");

  // Determine copy direction
  bool host2device = src.is_cpu() && dst.is_privateuseone();
  bool device2host = src.is_privateuseone() && dst.is_cpu();

  const at::Tensor* dev_tensor = host2device ? &dst : &src;
  const at::Tensor* cpu_tensor = host2device ? &src : &dst;

  if (host2device || device2host) {
    // Host-to-device or device-to-host copy
    void* cpu_ptr = const_cast<void*>(cpu_tensor->storage().data());

    // Get SpyreTensorLayout using the public API
    SpyreTensorLayout stl = get_spyre_tensor_layout(*dev_tensor);

    // Extract device allocation from Spyre tensor storage
    auto* spyre_impl =
        static_cast<SpyreTensorImpl*>(dev_tensor->unsafeGetTensorImpl());
    auto& storage = spyre_impl->storage();
    auto* ctx = static_cast<SharedOwnerCtx*>(storage.data_ptr().get_context());

    DataConversionInfo dci = generate_dci(
        cpu_tensor, dev_tensor, stl, cpu_tensor->storage_offset(), host2device);

    copyAsyncImpl(cpu_ptr, &ctx->composite_addr, &dci, host2device);

  } else {
    TORCH_CHECK(false, "Unsupported copy types: src on ", src.device(),
                " dst on ", dst.device());
  }
}

flex::RuntimeStream* SpyreStream::getRuntimeHandle() const {
  if (flex_handle_ != nullptr) {
    return flex_handle_;
  }
  flex_handle_ = resolveRuntimeHandle();
  return flex_handle_;
}

flex::RuntimeStream* SpyreStream::resolveRuntimeHandle() const {
  auto& pool = getStreamPool();
  std::lock_guard<std::mutex> lock(pool.mutex);

  auto it = pool.stream_handle_map.find(id());
  TORCH_CHECK(it != pool.stream_handle_map.end(),
              "SpyreStream: no flex handle for stream id ", id(),
              " — was the stream pool initialized for this device?");
  return it->second;
}

void SpyreStream::fillAsync(const flex::CompositeAddress* dst, double value,
                            DataFormats dtype, bool use_dmai) const {
  // Device-side MEMORY_FILL DMA. flex::RuntimeStream::fillAsync converts the
  // scalar value to the hardware fill pattern (via RuntimeOperationFill::
  // valueToFillPattern) internally, so no FillParams construction is needed
  // here.
  getRuntimeHandle()->fillAsync(dst, value, dtype, use_dmai);
}

void SpyreStream::copyAsyncImpl(void* cpu_ptr,
                                const flex::CompositeAddress* device_address,
                                const DataConversionInfo* dci, bool host2device,
                                uint64_t copy_bytes) const {
  // Wrap dci in shared_ptr for flex API
  auto dci_ptr = dci ? std::make_shared<data_conversion_info>(*dci) : nullptr;

  // Get the flex runtime stream handle
  flex::RuntimeStream* flex_stream = getRuntimeHandle();

  // A non-zero copy_bytes transfers exactly that many bytes; otherwise the full
  // device allocation is transferred (the tensor-copy default).
  const uint64_t transfer_bytes =
      copy_bytes != 0 ? copy_bytes : device_address->total_size();

  // Create and launch operation via flex's typed params API (the generic
  // launchOperation(op) overload was removed from flex).
  auto* params = flex::createDmaParams(cpu_ptr, transfer_bytes, host2device,
                                       device_address, std::move(dci_ptr));
  if (host2device) {
    flex_stream->launchOperationH2D(params);
  } else {
    flex_stream->launchOperationD2H(params);
  }
  flex::destroyDmaParams(params);
}

void SpyreStream::executeProgramAsync(
    const KernelArtifacts& arts, const std::vector<at::Tensor>& args) const {
  // NOTE: Maybe it's better/faster if we know the exact number of arguments
  // as it is tracked inside KerntlArtifacts
  std::vector<const flex::CompositeAddress*> tensor_allocs;
  for (size_t i = 0; i < args.size(); ++i) {
    auto* ctx = static_cast<SharedOwnerCtx*>(
        args[i].storage().data_ptr().get_context());
    tensor_allocs.push_back(&ctx->composite_addr);
  }

  // Program. Build via flex's typed params API (the generic
  // launchOperation(op) overload and the RuntimeOperationCompute direct-launch
  // ctor were removed from flex).
  auto* ctx = static_cast<SharedOwnerCtx*>(arts.device_alloc.get_context());
  auto* params = flex::createComputeParams(
      &ctx->composite_addr, std::move(tensor_allocs), arts.bundle_mlir_path);

  // Get the flex runtime stream handle
  flex::RuntimeStream* flex_stream = getRuntimeHandle();
  flex_stream->launchOperationCompute(params);
  flex::destroyComputeParams(params);
}

void initializeStreamPoolImpl(c10::DeviceIndex device_index) {
  auto& pool = getStreamPool();
  std::lock_guard<std::mutex> lock(pool.mutex);

  // Register the default stream (ID 0) using the concrete flex handle.
  // This ensures getRuntimeHandle() resolves stream 0 to the real RuntimeStream
  // instance owned by RuntimeContext.
  auto runtime = GlobalRuntime::get();
  pool.stream_handle_map[0] = runtime->getDefaultStream();

  // Initialize low priority streams (IDs 1 to kStreamsPerDevice)
  pool.low_priority_streams[device_index].reserve(kStreamsPerDevice);
  for (int i = 1; i <= kStreamsPerDevice; ++i) {
    pool.low_priority_streams[device_index].push_back(i);
  }
  pool.next_low_priority_idx[device_index] = 0;

  // Initialize high priority streams
  pool.high_priority_streams[device_index].reserve(
      kHighPriorityStreamsPerDevice);
  for (int i = 1; i <= kHighPriorityStreamsPerDevice; ++i) {
    pool.high_priority_streams[device_index].push_back(kStreamsPerDevice + i);
  }
  pool.next_high_priority_idx[device_index] = 0;
}

void initializeStreamPool(c10::DeviceIndex device_index) {
  auto& pool = getStreamPool();
  std::call_once(pool.device_init_flags[device_index], initializeStreamPoolImpl,
                 device_index);
}

SpyreStream getDefaultStream(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }
  initializeStreamPool(device.index());
  return SpyreStream(c10::Stream(c10::Stream::DEFAULT, device));
}

SpyreStream getCurrentStream(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }

  auto it = current_streams.find(device.index());
  if (it == current_streams.end()) {
    return getDefaultStream(device);
  }

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, it->second));
}

SpyreStream setCurrentStream(SpyreStream stream) {
  auto device = stream.device();
  auto old_stream = getCurrentStream(device);
  current_streams[device.index()] = stream.id();
  return old_stream;
}

SpyreStream getStreamFromPool(c10::Device device, int priority) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }

  // Ensure runtime is initialized before creating streams
  // This is critical when torch.Stream() is called before any tensor operations
  startRuntime();

  initializeStreamPool(device.index());

  auto& pool = getStreamPool();
  std::lock_guard<std::mutex> lock(pool.mutex);

  c10::StreamId stream_id;
  if (priority == 0) {
    // Low priority stream - round-robin from low priority pool
    auto& streams = pool.low_priority_streams[device.index()];
    auto& idx = pool.next_low_priority_idx[device.index()];

    stream_id = streams[idx];
    idx = (idx + 1) % streams.size();

  } else {
    // High priority stream - round-robin from high priority pool
    auto& streams = pool.high_priority_streams[device.index()];
    auto& idx = pool.next_high_priority_idx[device.index()];

    stream_id = streams[idx];
    idx = (idx + 1) % streams.size();
  }

  // Create corresponding flex stream handle (if not exists). flex removed
  // RuntimeContext::toPriority(int); map the torch priority (0=normal, <0=high)
  // to the RuntimeStreamPriority enum directly.
  if (pool.stream_handle_map.find(stream_id) == pool.stream_handle_map.end()) {
    auto runtime = GlobalRuntime::get();
    flex::RuntimeStreamPriority stream_priority =
        priority < 0 ? flex::RuntimeStreamPriority::HIGH
                     : flex::RuntimeStreamPriority::NORMAL;
    flex::RuntimeStream* flex_handle = runtime->createStream(stream_priority);
    pool.stream_handle_map[stream_id] = flex_handle;
  }

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, stream_id));
}

namespace {

// Bounded wait on one flex handle. Returns true if idle within the timeout,
// false at the deadline (the caller composes the multi-stream error so the
// message names every stream that was still busy).
bool waitForHandle(flex::RuntimeStream* handle,
                   std::chrono::milliseconds timeout) {
  constexpr auto poll_interval = std::chrono::milliseconds(1);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!handle->query()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(poll_interval);
  }
  return true;
}

}  // anonymous namespace

// Shared body of synchronizeDevice / synchronizeDeviceBounded: no timeout ->
// the original unbounded per-handle wait; with timeout -> bounded poll that
// names every stream still busy at the deadline in one typed error.
void synchronizeDeviceImpl(c10::optional<c10::Device> device,
                           std::optional<std::chrono::milliseconds> timeout);

void synchronizeDevice(c10::optional<c10::Device> device) {
  synchronizeDeviceImpl(std::move(device), std::nullopt);
}

void synchronizeDeviceBounded(c10::optional<c10::Device> device,
                              std::chrono::milliseconds timeout) {
  // A negative timeout is a caller bug, not a wait request — refuse rather
  // than treat it as an immediate timeout (mirrors the single-stream
  // overload; validated here, the one entry lacking it).
  TORCH_CHECK(timeout.count() >= 0,
              "synchronize_bounded: negative timeout (", timeout.count(),
              "ms)");
  synchronizeDeviceImpl(std::move(device), timeout);
}

void synchronizeDeviceImpl(c10::optional<c10::Device> device,
                           std::optional<std::chrono::milliseconds> timeout) {
  auto sync_one_device = [&](c10::Device dev) {
    if (dev.index() == -1) {
      dev = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
    }
    const auto device_index = dev.index();

    std::vector<flex::RuntimeStream*> handles_to_sync;
    // Parallel to handles_to_sync: the pool (c10) stream id for each handle,
    // used to name busy streams in the timeout error. flex::RuntimeStream has
    // no streamId() of its own (see the TODO in the single-stream overload).
    std::vector<c10::StreamId> pool_ids;
    {
      auto& pool = getStreamPool();
      std::lock_guard<std::mutex> lock(pool.mutex);

      // Default stream (ID 0) is always present when the pool is initialized
      auto default_it = pool.stream_handle_map.find(0);
      if (default_it != pool.stream_handle_map.end()) {
        handles_to_sync.push_back(default_it->second);
        pool_ids.push_back(default_it->first);
      }

      auto collect = [&](auto& stream_map) {
        auto it = stream_map.find(device_index);
        if (it == stream_map.end()) return;
        for (auto sid : it->second) {
          auto h = pool.stream_handle_map.find(sid);
          if (h != pool.stream_handle_map.end()) {
            handles_to_sync.push_back(h->second);
            pool_ids.push_back(h->first);
          }
        }
      };
      collect(pool.low_priority_streams);
      collect(pool.high_priority_streams);
    }  // lock released

    auto runtime = GlobalRuntime::get();
    c10::DeviceGuard guard(dev);
    if (!timeout.has_value()) {
      for (auto handle : handles_to_sync) {
        handle->synchronize();
      }
      return;
    }

    // Bounded pass with ONE shared deadline: every stream draws from the
    // remaining time, so total wall time is bounded by a single timeout (a
    // per-stream full timeout would multiply by the stream count while the
    // error claimed one timeout). Check every stream so one exception names
    // the complete set of stuck streams — a multi-stream hang should be fully
    // diagnosed from a single error, not discovered one synchronize at a
    // time.
    const auto deadline = std::chrono::steady_clock::now() + *timeout;
    std::vector<c10::StreamId> busy_ids;
    for (size_t i = 0; i < handles_to_sync.size(); ++i) {
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - std::chrono::steady_clock::now());
      if (remaining.count() <= 0 ||
          !waitForHandle(handles_to_sync[i], remaining)) {
        busy_ids.push_back(pool_ids[i]);
      }
    }
    if (!busy_ids.empty()) {
      std::ostringstream joined;
      for (size_t i = 0; i < busy_ids.size(); ++i) {
        if (i > 0) joined << ", ";
        joined << busy_ids[i];
      }
      TORCH_CHECK(false,
                  "SpyreStreamSynchronizeTimeout: device ", device_index,
                  " did not become idle within ", timeout->count(),
                  "ms. Pool stream IDs still busy: [", joined.str(),
                  "]. The streams are NOT reset — the operations may still be "
                  "running; a subsequent unbounded synchronize() or device "
                  "reset is required to reclaim them.");
    }
  };
  if (device.has_value()) {
    sync_one_device(device.value());
  } else {
    sync_one_device(
        c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx));
  }
}

}  // namespace spyre
