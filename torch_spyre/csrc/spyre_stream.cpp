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

namespace spyre {

// Stream pool configuration
constexpr int kStreamsPerDevice = 32;

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
                             const at::Tensor& dst) const {}

flex::StreamHandle SpyreStream::get_flex_handle() const {}

void SpyreStream::copy_async_impl(void* cpu_ptr, void* device_owner,
                                  int device_id,
                                  const data_conversion_info& dci,
                                  bool host2device) const {}

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

SpyreStream getStreamFromPool(c10::Device device, int priority) {
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
}  // namespace spyre
