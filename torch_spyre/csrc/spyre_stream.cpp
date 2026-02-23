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

#include <flex/stream_handle.hpp>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_mem.h"
#include "spyre_tensor_impl.h"

namespace spyre {

// Stream pool configuration
constexpr int kStreamsPerDevice = 32;
constexpr int kHighPriorityStreamsPerDevice = 32;

// Constructor
SpyreStream::SpyreStream()
    : stream_(getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1, 0))
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
  if (id() == 0) {
    return 0;
  } else if (id() <= kStreamsPerDevice) {
    return 0;
  } else {
    return -1;
  }
}

bool SpyreStream::query() const {
  c10::DeviceGuard guard(stream_.device());

  DEBUGINFO("SpyreStream::query() - stream ", id(), " on device ",
            device().index());

  return true;  // Assume all work completed
}

void SpyreStream::synchronize() const {
  c10::DeviceGuard guard(stream_.device());

  DEBUGINFO("SpyreStream::synchronize() - stream ", id(), " on device ",
            device().index());

  // TODO(tmhoangt): add code
  // For now, do nothing (assumes synchronous execution)
}

SpyreStream::operator c10::Stream() const {
  return stream_;
}

c10::Stream SpyreStream::unwrap() const {
  return stream_;
}

void SpyreStream::copy_async(const at::Tensor& src,
                             const at::Tensor& dst) const {
  DEBUGINFO("src (", src.scalar_type(), ") is on:", src.device());
  DEBUGINFO("dst (", dst.scalar_type(), ") on:", dst.device());

  // TODO(tmhoangt): add type conversion node
  TORCH_CHECK(
      src.scalar_type() == dst.scalar_type(),
      "Spyre backend does not support type conversion yet during copy.");
  bool host2device = src.is_cpu() && dst.is_privateuseone();
  bool device2host = src.is_privateuseone() && dst.is_cpu();

  const at::Tensor* dev_tensor = host2device ? &dst : &src;
  const at::Tensor* cpu_tensor = host2device ? &src : &dst;
  at::Tensor tmp_tensor;
  if (cpu_tensor->dim() == 0) {
    tmp_tensor = cpu_tensor->unsqueeze(0);
    cpu_tensor = &tmp_tensor;
  }

  void* cpu_ptr = cpu_tensor->data_ptr();
  void* dev_ptr = dev_tensor->data_ptr();
  DataConversionInfo dci;

  SpyreTensorLayout stl = get_spyre_tensor_layout(*dev_tensor);
  auto* ctx = static_cast<SharedOwnerCtx*>(
      dev_tensor->storage().data_ptr().get_context());
  dci = generate_dci_(&src, stl, host2device);
  if (host2device || device2host) {
    copy_async_impl(cpu_ptr, ctx->owner, ctx->device_id, dci, host2device);
  } else if (src.is_privateuseone() && dst.is_privateuseone()) {
    // Device to device copy - for now, we can just do a device to host copy
    // followed by a host to device copy
    // TODO(tmhoangt): optimize this by doing a direct device to device copy
    // once flex runtime supports it
    const at::Tensor cpu_tensor = src.cpu();
    copy_async(cpu_tensor, dst);
  } else {
    TORCH_CHECK(false, "Unsupported copy types: src on ", src.device(),
                " dst on ", dst.device());
  }
}

flex::StreamHandle SpyreStream::get_flex_handle() const {}

void SpyreStream::copy_async_impl(
    void* cpu_ptr, flex::DeviceMemoryAllocationPtr& device_allocation,
    int device_id, const DataConversionInfo& dci, bool host2device) const {}

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

  // Mapping from c10::StreamId to flex::StreamHandle
  std::unordered_map<c10::StreamId, flex::StreamHandle> stream_handle_map;
};

StreamPool& getStreamPool() {
  static StreamPool pool;
  return pool;
}

thread_local std::unordered_map<c10::DeviceIndex, c10::StreamId>
    current_streams;

}  // anonymous namespace

void initializeStreamPool(c10::DeviceIndex device_index) {
  auto& pool = getStreamPool();
  std::lock_guard<std::mutex> lock(pool.mutex);

  if (pool.low_priority_streams.find(device_index) !=
      pool.low_priority_streams.end()) {
    return;  // Already initialized
  }

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

SpyreStream getDefaultStream(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, 0);
  }
  return SpyreStream(c10::Stream(c10::Stream::DEFAULT, device));
}

SpyreStream getCurrentStream(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, 0);
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
    device = c10::Device(c10::DeviceType::PrivateUse1, 0);
  }

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

  // Create corresponding flex stream handle (if not exists)
  if (pool.stream_handle_map.find(stream_id) == pool.stream_handle_map.end()) {
    auto runtime = GlobalRuntime::get();
    flex::StreamHandle flex_handle =
        runtime->create_stream(device.index(), priority);
    pool.stream_handle_map[stream_id] = flex_handle;
  }

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, stream_id));
}

void synchronizeDevice(c10::optional<c10::Device> device) {
  if (device.has_value()) {
    auto stream = getCurrentStream(device.value());
    stream.synchronize();
  } else {
    // Synchronize all devices
    for (int i = 0; i < device_count(); ++i) {
      auto dev = c10::Device(c10::DeviceType::PrivateUse1, i);
      auto stream = getCurrentStream(dev);
      stream.synchronize();
    }
  }
}

}  // namespace spyre
