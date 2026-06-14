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

#include "spyre_guard.h"

#include <ATen/core/op_registration/adaption.h>

#include <flex/runtime_stream/runtime_event.hpp>

#include "module.h"
#include "spyre_allocator.h"
#include "spyre_device_enum.h"
#include "spyre_stream.h"

namespace spyre {

c10::DeviceType SpyreGuardImpl::type() const {
  return c10::DeviceType::PrivateUse1;
}

c10::Device SpyreGuardImpl::exchangeDevice(c10::Device d) const {
  auto old = getDevice();
  setDevice(d);
  return old;
}

c10::Device SpyreGuardImpl::getDevice() const {
  return {type(), tls_idx};
}

void SpyreGuardImpl::setDevice(c10::Device d) const {
  TORCH_INTERNAL_ASSERT(d.type() == type());
  // (optionally tell your runtime to switch)
  tls_idx = d.index();
}

void SpyreGuardImpl::uncheckedSetDevice(c10::Device) const noexcept {}

c10::DeviceIndex SpyreGuardImpl::deviceCount() const noexcept {
  return c10::DeviceIndex(getVisibleDeviceCount());
}

c10::Stream SpyreGuardImpl::getStream(c10::Device device) const {
  return getCurrentStream(device).unwrap();
}

c10::Stream SpyreGuardImpl::getDefaultStream(c10::Device device) const {
  return spyre::getDefaultStream(device).unwrap();
}

c10::Stream SpyreGuardImpl::getStreamFromGlobalPool(c10::Device device,
                                                    bool isHighPriority) const {
  int priority = isHighPriority ? -1 : 0;
  return getStreamFromPool(device, priority).unwrap();
}

c10::Stream SpyreGuardImpl::getNewStream(c10::Device device,
                                         int priority) const {
  return getStreamFromPool(device, priority).unwrap();
}

void SpyreGuardImpl::synchronizeStream(const c10::Stream& stream) const {
  TORCH_CHECK(stream.device().type() == this->type());
  SpyreStream(stream).synchronize();
}

bool SpyreGuardImpl::queryStream(const c10::Stream& stream) const {
  TORCH_CHECK(stream.device().type() == this->type());
  return SpyreStream(stream).query();
}

void SpyreGuardImpl::synchronizeDevice(c10::DeviceIndex device_index) const {
  c10::Device dev(c10::DeviceType::PrivateUse1, device_index);
  spyre::synchronizeDevice(dev);
}

c10::Stream SpyreGuardImpl::exchangeStream(c10::Stream stream) const {
  SpyreStream ss(stream);

  c10::Stream old = getCurrentStream(stream.device()).unwrap();

  // Set TLS current stream for THAT device index
  setCurrentStream(ss);

  return old;
}

void SpyreGuardImpl::recordDataPtrOnStream(const c10::DataPtr& data_ptr,
                                           const c10::Stream& stream) const {
  SpyreAllocator::instance().recordStream(data_ptr, stream);
}

// --- Event methods ---
// RuntimeEvent is non-copyable/non-movable, so we store as void* via
// new/delete. PyTorch's InlineEvent passes void* event_ to these methods.
//
// P0 limitation: record() calls stream->synchronize() on the host, and
// wait()/synchronize() block the host thread via shared_future. This means
// events provide correctness but NOT compute/comm overlap — the host
// serializes stream transitions. True device-side waiting (no host block)
// requires P2 firmware support (R5 SIGNAL/WAIT on HMLT counters).

void SpyreGuardImpl::destroyEvent(
    void* event, const c10::DeviceIndex device_index) const noexcept {
  delete static_cast<flex::RuntimeEvent*>(event);
}

void SpyreGuardImpl::record(void** event, const c10::Stream& stream,
                            const c10::DeviceIndex device_index,
                            const c10::EventFlag flag) const {
  if (!*event) {
    *event = new flex::RuntimeEvent();
  }
  auto* re = static_cast<flex::RuntimeEvent*>(*event);
  SpyreStream ss(stream);
  re->record(ss.getRuntimeHandle());
}

void SpyreGuardImpl::block(void* event, const c10::Stream& stream) const {
  if (!event) return;
  auto* re = static_cast<flex::RuntimeEvent*>(event);
  SpyreStream ss(stream);
  re->wait(ss.getRuntimeHandle());
}

bool SpyreGuardImpl::queryEvent(void* event) const {
  if (!event) return true;
  auto* re = static_cast<flex::RuntimeEvent*>(event);
  return re->query();
}

void SpyreGuardImpl::synchronizeEvent(void* event) const {
  if (!event) return;
  auto* re = static_cast<flex::RuntimeEvent*>(event);
  re->synchronize();
}

c10::DeviceCapability SpyreGuardImpl::getDeviceCapability(
    c10::Device /*unused*/) const {
  c10::DeviceCapability cap{};

  cap.capability_data.capability_bits =
      (1ULL << c10::kIndex_Float) | (1ULL << c10::kIndex_Half) |
      (1ULL << c10::kIndex_Bool) | (1ULL << c10::kIndex_Char) |
      (1ULL << c10::kIndex_Byte) | (1ULL << c10::kIndex_Short) |
      (1ULL << c10::kIndex_Int4) | (1ULL << c10::kIndex_BFloat16) |
      (1ULL << c10::kIndex_Float8_e4m3fn) |
      (1ULL << c10::kIndex_Float8_e5m2fnuz);

  return cap;
}

thread_local c10::DeviceIndex SpyreGuardImpl::tls_idx = 0;

// Registration — runs when _C.so is loaded.
// Loading _C.so does NOT trigger device initialization; that only
// happens when start_runtime() is called via _lazy_init().
C10_REGISTER_GUARD_IMPL(PrivateUse1, SpyreGuardImpl);

}  // namespace spyre
