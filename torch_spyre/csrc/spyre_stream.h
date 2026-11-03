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

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Stream.h>

#include <chrono>
#include <vector>

#include "module.h"
#include "spyre_kernel.h"

namespace spyre {
class SpyreStream {
 private:
  c10::Stream stream_;

 public:
  explicit SpyreStream(c10::Stream stream);
  SpyreStream();

  // Stream properties
  c10::StreamId id() const;
  c10::Device device() const;
  int priority() const;

  // Synchronization
  bool query() const;        // Check if work completed
  void synchronize() const;  // Block until work done
  // Bounded synchronize: poll query() until idle or ``timeout`` elapses. A
  // timeout raises a typed SpyreStreamSynchronizeTimeout error naming the
  // stream, device, and last-observed in-flight/queued operation counts —
  // a hung kernel becomes a diagnosable error instead of a blocked thread.
  void synchronize(std::chrono::milliseconds timeout) const;

  void copyAsync(const at::Tensor& src, const at::Tensor& dst) const;
  // Bit-preserving host<->device copy with NO DataConversionInfo attached
  // (dci=nullptr). The DMA transfers the full device allocation as raw bytes,
  // bypassing deeptools' ConvertData entirely. Sanctioned escape hatch for
  // dtypes with no same-format ConvertData arm on the device (e.g. SENINT16,
  // DtException sen_data_convert.cpp:2505); never value-converting. Enforced
  // preconditions: exactly one side CPU and the other spyre, matching scalar
  // types, contiguous host tensor, and host storage large enough to hold the
  // whole device allocation (the transfer is allocation-sized, not
  // numel-sized).
  void copyRawAsync(const at::Tensor& src, const at::Tensor& dst) const;
  // Copies exactly ``program_bytes`` from ``prog_cpu_ptr`` into the device
  // allocation. The byte count is explicit so the DMA never reads past the end
  // of the host program buffer when the allocation is rounded up beyond the
  // program size.
  void copyProgramAsync(void* prog_cpu_ptr,
                        const flex::CompositeAddress* device_address,
                        uint64_t program_bytes) const;
  void executeProgramAsync(const KernelArtifacts& arts,
                           const std::vector<at::Tensor>& args) const;

  // Device-side MEMORY_FILL DMA. Routes through flex::RuntimeStream::fillAsync,
  // which performs the value->pattern conversion internally (no host buffer or
  // H2D copy).
  void fillAsync(const flex::CompositeAddress* dst, double value,
                 DataFormats dtype, bool use_dmai = true) const;

  // Conversions
  c10::Stream unwrap() const;

 private:
  mutable flex::RuntimeStream* flex_handle_ = nullptr;

  flex::RuntimeStream* getRuntimeHandle() const;
  flex::RuntimeStream* resolveRuntimeHandle() const;
  // ``copy_bytes == 0`` transfers the full device allocation (the tensor-copy
  // default); a non-zero value transfers exactly that many bytes (the program
  // copy, bound to the program's own size rather than the rounded allocation).
  void copyAsyncImpl(void* cpu_ptr,
                     const flex::CompositeAddress* device_address,
                     const DataConversionInfo* dci, bool host2device,
                     uint64_t copy_bytes = 0) const;
};

/**
 * Get a stream from the global stream pool.
 * Streams are allocated round-robin from a pool maintained by torch-spyre.
 *
 * @param device Device to allocate stream on (default: current device)
 * @param priority Stream priority: 0 (normal) or -1 (high)
 * @return A SpyreStream from the pool
 */
SpyreStream getStreamFromPool(
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1),
    int priority = 0);

/**
 * Get the default stream for a device.
 * The default stream is stream ID 0 and is always available.
 *
 * @param device Device to get default stream for
 * @return The default SpyreStream (stream ID 0)
 */
SpyreStream getDefaultStream(
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1));

/**
 * Get the current stream for a device (thread-local).
 * Each thread maintains its own current stream per device.
 *
 * @param device Device to get current stream for
 * @return The current SpyreStream for this thread
 */
SpyreStream getCurrentStream(
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1));

/**
 * Set the current stream for a device (thread-local).
 *
 * @param stream Stream to set as current
 * @return The previous current stream
 */
SpyreStream setCurrentStream(SpyreStream stream);

void synchronizeDevice(c10::optional<c10::Device> device);

// Bounded device synchronize: every pooled stream must go idle within
// ``timeout`` or a typed SpyreStreamSynchronizeTimeout error names all stuck
// pool stream IDs (flex streams have no ids of their own). The streams are
// not reset — the caller decides whether to wait unbounded or reset the
// device.
void synchronizeDeviceBounded(c10::optional<c10::Device> device,
                              std::chrono::milliseconds timeout);

}  // namespace spyre
