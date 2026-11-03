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
#include <c10/util/intrusive_ptr.h>

#include "module.h"

namespace spyre {

at::Tensor spyre_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                               std::optional<c10::ScalarType> dtype_opt,
                               std::optional<c10::Layout> layout_opt,
                               std::optional<c10::Device> device_opt,
                               std::optional<bool> pin_memory_opt);

at::Tensor spyre_copy_from(const at::Tensor& self, const at::Tensor& dst,
                           bool non_blocking);

/**
 * Bit-preserving host<->device copy that bypasses the DCI/ConvertData path.
 *
 * Issues the DMA with dci=nullptr (the same raw-word mechanism
 * copyProgramAsync uses for init packets), so deeptools' ConvertData is never
 * involved. This is the sanctioned escape hatch for dtypes with no
 * same-format ConvertData arm on the device (e.g. torch.int16 ->
 * (SENINT16, SENINT16), which throws DtException at
 * sen_data_convert.cpp:2505 on-box in both H2D and D2H).
 *
 * Contract (enforced in SpyreStream::copyRawAsync):
 *   - exactly one of self/dst on CPU, the other on spyre;
 *   - identical scalar types (bit-preserving, never value-converting);
 *   - contiguous host tensor whose storage holds the full (stick-padded)
 *     device allocation.
 *
 * Unlike spyre_copy_from there is no D2H staging/view path: the copy is a raw
 * byte mirror of the physical allocation.
 *
 * @param self One of the tensors (host or spyre)
 * @param dst The other tensor (spyre or host)
 * @param non_blocking If false, synchronize the stream before returning
 * @return dst
 */
at::Tensor spyre_raw_copy(const at::Tensor& self, const at::Tensor& dst,
                          bool non_blocking);

/**
 * Fill a spyre tensor with a scalar value using device-side FillDMA.
 *
 * Uses flex::RuntimeStream::fillAsync() to perform the fill entirely
 * on-device without allocating a host buffer or performing an H2D DMA.
 *
 * @param self The spyre tensor to fill (in-place)
 * @param value The fill value as a double (converted to the correct hardware
 *              pattern based on the tensor's dtype)
 * @return The filled tensor (same as self)
 */
at::Tensor spyre_fill_tensor(const at::Tensor& self, double value);

class SpyreTensorLayout;
at::Tensor spyre_empty_with_layout(c10::IntArrayRef size,
                                   c10::IntArrayRef stride,
                                   c10::ScalarType dtype,
                                   SpyreTensorLayout device_layout);

at::Tensor empty_with_layout(
    c10::IntArrayRef size, SpyreTensorLayout device_layout,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

at::Tensor py_empty_with_layout(
    c10::IntArrayRef size, SpyreTensorLayout device_layout,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

auto generate_dci(const at::Tensor* cpu_tensor, const at::Tensor* dev_tensor,
                  SpyreTensorLayout stl, int64_t cpu_offset, bool host2device)
    -> DataConversionInfo;
}  // namespace spyre
