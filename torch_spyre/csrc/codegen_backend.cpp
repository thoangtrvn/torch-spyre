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

#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace spyre::codegen {
namespace {

enum class DataFormats : int {
  SEN169_FP16 = 0,
  IEEE_FP32 = 1,
  INVALID = 2,
  SEN143_FP8 = 3,
  SEN152_FP8 = 4,
  SEN153_FP9 = 5,
  SENINT2 = 6,
  SENINT4 = 7,
  SENINT8 = 8,
  SENINT16 = 9,
  SENINT24 = 10,
  IEEE_INT64 = 11,
  IEEE_INT32 = 12,
  SENUINT32 = 13,
  SENUINT2 = 14,
  IEEE_FP16 = 15,
  BOOL = 16,
  BFLOAT16 = 17,
  SEN18F_FP24 = 18,
};

thread_local c10::DeviceIndex current_device_index = 0;

c10::Device current_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, current_device_index);
}

const char* data_format_name(DataFormats df) {
  switch (df) {
    case DataFormats::SEN169_FP16:
      return "SEN169_FP16";
    case DataFormats::IEEE_FP32:
      return "IEEE_FP32";
    case DataFormats::INVALID:
      return "INVALID";
    case DataFormats::SEN143_FP8:
      return "SEN143_FP8";
    case DataFormats::SEN152_FP8:
      return "SEN152_FP8";
    case DataFormats::SEN153_FP9:
      return "SEN153_FP9";
    case DataFormats::SENINT2:
      return "SENINT2";
    case DataFormats::SENINT4:
      return "SENINT4";
    case DataFormats::SENINT8:
      return "SENINT8";
    case DataFormats::SENINT16:
      return "SENINT16";
    case DataFormats::SENINT24:
      return "SENINT24";
    case DataFormats::IEEE_INT64:
      return "IEEE_INT64";
    case DataFormats::IEEE_INT32:
      return "IEEE_INT32";
    case DataFormats::SENUINT32:
      return "SENUINT32";
    case DataFormats::SENUINT2:
      return "SENUINT2";
    case DataFormats::IEEE_FP16:
      return "IEEE_FP16";
    case DataFormats::BOOL:
      return "BOOL";
    case DataFormats::BFLOAT16:
      return "BFLOAT16";
    case DataFormats::SEN18F_FP24:
      return "SEN18F_FP24";
  }
  return "INVALID";
}

int64_t elems_per_stick_for_format(DataFormats df) {
  switch (df) {
    case DataFormats::SEN169_FP16:
    case DataFormats::IEEE_FP16:
    case DataFormats::BFLOAT16:
      return 64;
    case DataFormats::IEEE_FP32:
    case DataFormats::IEEE_INT32:
      return 32;
    case DataFormats::IEEE_INT64:
      return 16;
    case DataFormats::BOOL:
    case DataFormats::SEN143_FP8:
    case DataFormats::SEN152_FP8:
      return 128;
    default:
      TORCH_CHECK(false, "elements-per-stick is undefined for ",
                  data_format_name(df));
  }
}

DataFormats get_device_dtype(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Half:
      return DataFormats::SEN169_FP16;
    case c10::ScalarType::Float:
      return DataFormats::IEEE_FP32;
    case c10::ScalarType::BFloat16:
      return DataFormats::BFLOAT16;
    case c10::ScalarType::Long:
      return DataFormats::IEEE_INT64;
    case c10::ScalarType::Int:
      return DataFormats::IEEE_INT32;
    case c10::ScalarType::Bool:
      return DataFormats::BOOL;
    case c10::ScalarType::Float8_e4m3fn:
      return DataFormats::SEN143_FP8;
    case c10::ScalarType::Float8_e5m2:
    case c10::ScalarType::Float8_e5m2fnuz:
      return DataFormats::SEN152_FP8;
    default:
      TORCH_CHECK(false, "unsupported Spyre dtype in codegen-only mode: ", dtype);
  }
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  int64_t stride = 1;
  for (auto dim = static_cast<int64_t>(sizes.size()) - 1; dim >= 0; --dim) {
    strides[dim] = stride;
    stride *= sizes[dim];
  }
  return strides;
}

std::vector<int64_t> contiguous_strides(c10::IntArrayRef sizes) {
  return contiguous_strides(sizes.vec());
}

class SpyreTensorLayout {
 public:
  std::vector<int64_t> device_size;
  std::vector<int64_t> stride_map;
  DataFormats device_dtype;

  SpyreTensorLayout() : device_dtype(DataFormats::INVALID) {}

  SpyreTensorLayout(std::vector<int64_t> host_size, c10::ScalarType dtype)
      : device_dtype(get_device_dtype(dtype)) {
    init_from_host_size(std::move(host_size));
  }

  SpyreTensorLayout(std::vector<int64_t> host_size,
                    std::vector<int64_t> host_strides, c10::ScalarType dtype,
                    std::vector<int32_t> dim_order)
      : device_dtype(get_device_dtype(dtype)) {
    (void)host_strides;
    (void)dim_order;
    init_from_host_size(std::move(host_size));
  }

  SpyreTensorLayout(std::vector<int64_t> device_size,
                    std::vector<int64_t> stride_map, DataFormats device_dtype)
      : device_size(std::move(device_size)),
        stride_map(std::move(stride_map)),
        device_dtype(device_dtype) {}

  void init_from_host_size(std::vector<int64_t> host_size) {
    const auto eps = elems_per_stick_for_format(device_dtype);
    if (host_size.empty()) {
      device_size = {};
      stride_map = {};
      return;
    }
    if (host_size.size() == 1) {
      const auto n = host_size[0];
      device_size = {(n + eps - 1) / eps, eps};
    } else {
      device_size = std::move(host_size);
      device_size.back() = (device_size.back() + eps - 1) / eps;
      device_size.push_back(eps);
    }
    stride_map = contiguous_strides(device_size);
  }

  int64_t elems_per_stick() const {
    return codegen::elems_per_stick_for_format(device_dtype);
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "SpyreTensorLayout(device_size=[";
    for (size_t i = 0; i < device_size.size(); ++i) {
      ss << device_size[i];
      if (i + 1 < device_size.size()) ss << ", ";
    }
    ss << "], stride_map =[";
    for (size_t i = 0; i < stride_map.size(); ++i) {
      ss << stride_map[i];
      if (i + 1 < stride_map.size()) ss << ", ";
    }
    ss << "], device_dtype=DataFormats." << data_format_name(device_dtype)
       << ")";
    return ss.str();
  }

  bool operator==(const SpyreTensorLayout& other) const {
    return device_size == other.device_size && stride_map == other.stride_map &&
           device_dtype == other.device_dtype;
  }
};

class SpyreCodegenTensorImpl final : public at::TensorImpl {
 public:
  SpyreTensorLayout spyre_layout;
  std::vector<int64_t> dma_sizes;
  std::vector<int64_t> dma_strides;

  SpyreCodegenTensorImpl(c10::Storage&& storage, c10::DispatchKeySet key_set,
                         const caffe2::TypeMeta& dtype,
                         SpyreTensorLayout layout)
      : TensorImpl(std::move(storage), key_set, dtype),
        spyre_layout(std::move(layout)) {
    set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::Default);
  }

  c10::intrusive_ptr<at::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    return shallow_copy_and_detach_core(version_counter,
                                        allow_tensor_metadata_change);
  }

  c10::intrusive_ptr<at::TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override {
    return shallow_copy_and_detach_core(std::move(version_counter),
                                        allow_tensor_metadata_change);
  }

  void shallow_copy_from(const c10::intrusive_ptr<at::TensorImpl>& impl) override {
    auto* src = static_cast<SpyreCodegenTensorImpl*>(impl.get());
    TensorImpl::shallow_copy_from(impl);
    spyre_layout = src->spyre_layout;
    dma_sizes = src->dma_sizes;
    dma_strides = src->dma_strides;
  }

 private:
  template <typename VariableVersion>
  c10::intrusive_ptr<at::TensorImpl> shallow_copy_and_detach_core(
      VariableVersion&& version_counter, bool allow_tensor_metadata_change) const {
    auto impl = c10::make_intrusive<SpyreCodegenTensorImpl>(
        c10::Storage(storage_), key_set_, data_type_, spyre_layout);
    impl->dma_sizes = dma_sizes;
    impl->dma_strides = dma_strides;
    copy_tensor_metadata(/*src_impl=*/this, /*dest_impl=*/impl.get(),
                         /*version_counter=*/std::forward<VariableVersion>(
                             version_counter),
                         /*allow_tensor_metadata_change=*/
                             allow_tensor_metadata_change);
    return impl;
  }
};

SpyreCodegenTensorImpl* checked_spyre_impl(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "expected a Spyre codegen tensor, got ", tensor.device());
  return static_cast<SpyreCodegenTensorImpl*>(tensor.unsafeGetTensorImpl());
}

struct SpyreCodegenHooksArgs : public at::PrivateUse1HooksArgs {};

struct SpyreCodegenHooksInterface : public at::PrivateUse1HooksInterface {
  SpyreCodegenHooksInterface() = default;
  explicit SpyreCodegenHooksInterface(SpyreCodegenHooksArgs) {}
  ~SpyreCodegenHooksInterface() override = default;

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    return device_index == 0;
  }

  bool isAvailable() const override {
    return true;
  }
};

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, SpyreCodegenHooksInterface,
                       SpyreCodegenHooksArgs);
C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, SpyreCodegenHooksInterface,
                    SpyreCodegenHooksArgs);
C10_REGISTER_TYPED_CLASS(PrivateUse1HooksRegistry, "SpyreCodegenHooks",
                         SpyreCodegenHooksInterface);

struct SpyreCodegenGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  SpyreCodegenGuardImpl() = default;
  explicit SpyreCodegenGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == static_type);
  }

  c10::DeviceType type() const override {
    return static_type;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    auto old_device = current_device();
    current_device_index = d.index();
    return old_device;
  }

  c10::Device getDevice() const override {
    return current_device();
  }

  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    TORCH_CHECK(d.index() == 0,
                "codegen-only Spyre backend exposes only fake device 0");
    current_device_index = d.index();
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    current_device_index = d.index();
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::UNSAFE, current_device(), 0);
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::UNSAFE, current_device(), 0);
  }

  c10::Stream getStreamFromGlobalPool(c10::Device d,
                                      bool isHighPriority = false) const override {
    return c10::Stream(c10::Stream::UNSAFE, current_device(), 0);
  }

  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    return c10::Stream(c10::Stream::UNSAFE, current_device(), 0);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::UNSAFE, current_device(), 0);
  }

  void destroyEvent(void* event,
                    const c10::DeviceIndex device_index) const noexcept override {}

  void record(void** event, const c10::Stream& stream,
              const c10::DeviceIndex device_index,
              const c10::EventFlag flag) const override {}

  void block(void* event, const c10::Stream& stream) const override {}

  bool queryEvent(void* event) const override {
    return true;
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  bool queryStream(const c10::Stream& stream) const override {
    return true;
  }

  void synchronizeStream(const c10::Stream& stream) const override {}

  void synchronizeEvent(void* event) const override {}

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr,
                             const c10::Stream& stream) const override {}

  double elapsedTime(void* event1, void* event2,
                     const c10::DeviceIndex device_index) const override {
    return 0.0;
  }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, SpyreCodegenGuardImpl);

void delete_fake_allocation(void* ptr) {
  std::free(ptr);
}

c10::DataPtr make_fake_data_ptr(size_t logical_nbytes, c10::Device device) {
  if (logical_nbytes == 0) {
    return {nullptr, device};
  }
  void* token = std::malloc(1);
  TORCH_CHECK(token != nullptr, "failed to allocate fake Spyre codegen token");
  return {token, token, &delete_fake_allocation, device};
}

size_t storage_nbytes(c10::IntArrayRef size, c10::IntArrayRef stride,
                      c10::ScalarType dtype) {
  if (size.empty()) {
    return c10::elementSize(dtype);
  }
  int64_t max_offset = 0;
  for (const auto dim : c10::irange(size.size())) {
    TORCH_CHECK(size[dim] >= 0, "negative dimension not allowed: ", size[dim]);
    TORCH_CHECK(stride[dim] >= 0,
                "negative stride is not supported by Spyre codegen tensors");
    if (size[dim] == 0) {
      return 0;
    }
    max_offset += (size[dim] - 1) * stride[dim];
  }
  return static_cast<size_t>(max_offset + 1) * c10::elementSize(dtype);
}

at::Tensor make_fake_spyre_tensor(c10::IntArrayRef size, c10::IntArrayRef stride,
                                  c10::ScalarType dtype, c10::Device device,
                                  SpyreTensorLayout device_layout) {
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(device.index() <= 0,
              "codegen-only Spyre backend exposes only fake device 0");
  at::detail::check_size_nonnegative(size);
  TORCH_CHECK(size.size() == stride.size(),
              "size and stride rank mismatch for fake Spyre tensor");

  auto logical_nbytes = storage_nbytes(size, stride, dtype);
  auto data_ptr = make_fake_data_ptr(logical_nbytes, device);
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), logical_nbytes, std::move(data_ptr),
      /*allocator=*/nullptr, /*resizable=*/false);
  auto storage = c10::Storage(std::move(storage_impl));

  constexpr c10::DispatchKeySet key_set(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreCodegenTensorImpl>(
      std::move(storage), key_set, c10::scalarTypeToTypeMeta(dtype),
      std::move(device_layout));
  auto* impl = static_cast<SpyreCodegenTensorImpl*>(tensor.unsafeGetTensorImpl());
  impl->set_sizes_and_strides(size, stride);
  impl->dma_sizes = size.vec();
  impl->dma_strides = stride.vec();
  return tensor;
}

at::Tensor make_fake_spyre_tensor(c10::IntArrayRef size, c10::IntArrayRef stride,
                                  c10::ScalarType dtype, c10::Device device) {
  return make_fake_spyre_tensor(size, stride, dtype, device,
                                SpyreTensorLayout(size.vec(), dtype));
}

at::Tensor spyre_codegen_empty_strided(
    c10::IntArrayRef size, c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt) {
  const auto dtype = c10::dtype_or_default(dtype_opt);
  c10::Device device = device_opt.value_or(current_device());
  if (device.index() < 0) {
    device = current_device();
  }
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
              "Spyre codegen tensors only support strided layout");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt),
              "Pin memory can only be on CPU");
  return make_fake_spyre_tensor(size, stride, dtype, device);
}

at::Tensor spyre_codegen_empty(
    c10::IntArrayRef size, std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto dtype = c10::dtype_or_default(dtype_opt);
  c10::Device device = device_opt.value_or(current_device());
  if (device.index() < 0) {
    device = current_device();
  }
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
              "Spyre codegen tensors only support strided layout");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt),
              "Pin memory can only be on CPU");
  return make_fake_spyre_tensor(size, contiguous_strides(size.vec()), dtype, device);
}

at::Tensor spyre_codegen_empty_with_layout(
    c10::IntArrayRef size, c10::IntArrayRef stride, c10::ScalarType dtype,
    SpyreTensorLayout device_layout) {
  return make_fake_spyre_tensor(size, stride, dtype, current_device(),
                                std::move(device_layout));
}

at::Tensor py_empty_with_layout(
    c10::IntArrayRef size, SpyreTensorLayout device_layout,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto dtype = c10::dtype_or_default(dtype_opt);
  auto device = device_opt.value_or(current_device());
  if (device.index() < 0) {
    device = current_device();
  }
  return make_fake_spyre_tensor(size, contiguous_strides(size.vec()), dtype, device,
                                std::move(device_layout));
}

at::Tensor& spyre_codegen_set_storage(at::Tensor& result, at::Storage storage,
                                      int64_t storage_offset,
                                      c10::IntArrayRef size,
                                      c10::IntArrayRef stride) {
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  result.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return result;
}

at::Tensor& spyre_codegen_copy_(at::Tensor& self, const at::Tensor& src,
                                bool non_blocking) {
  TORCH_CHECK(self.is_privateuseone(),
              "Spyre codegen copy_ expects a Spyre destination tensor");
  TORCH_CHECK(self.scalar_type() == src.scalar_type(),
              "Spyre codegen copy_ does not perform dtype conversion");
  TORCH_CHECK(self.sizes() == src.sizes(),
              "Spyre codegen copy_ expects matching logical sizes");
  return self;
}

SpyreTensorLayout get_spyre_tensor_layout(const at::Tensor& tensor) {
  return checked_spyre_impl(tensor)->spyre_layout;
}

void set_spyre_tensor_layout(const at::Tensor& tensor,
                             const SpyreTensorLayout& layout) {
  checked_spyre_impl(tensor)->spyre_layout = layout;
}

void copy_tensor(const at::Tensor& src, const at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(dst.is_privateuseone() || src.is_privateuseone(),
              "Spyre codegen copy_tensor expects one Spyre tensor");
}

// Bit-level float->uint32 (deeptools BinaryConvert<uint32_t>).
static inline uint32_t float_to_bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return bits;
}

// Ported verbatim from deeptools/util/sen_data_convert.cpp
// (FloatToFp16BinInternal): sentient fp16 (DL16, 1-6-9) encoding. This
// feeds encode_constant(), which lands in generated device binaries —
// getting it wrong silently corrupts constants.
static inline uint16_t float_to_fp16_bin(float f) {
  const uint32_t f32 = float_to_bits(f);
  const int32_t exponent = ((f32 >> 23) & 0xFF) - 127;
  uint32_t fraction = f32 & 0x7FFFFF;
  // fp32: 1 (sign) + 8 (exp) + 23 (fra) -> sentient fp16: 1 + 6 + 9

  uint32_t rounding = 0;
  if (((fraction >> (23 - 9 - 1)) & 1) && ((fraction >> (23 - 9)) != 0x1FF)) {
    rounding = 1;
  }
  fraction = (fraction >> (23 - 9)) + rounding;
  uint32_t f16;
  if (exponent > 31) {
    f16 = (f32 >> 16) | 0x7FFF;  // INF with sign bit

    // Typical DL use-case is mask operation where saturating does not
    // impact accuracy and avoids potential NaN values during computation
    if (f < 0) {
      // for large negative values, cap saturation at 0xEFFF
      f16 = f16 & 0xEFFF;
    } else {
      // for large positive values, cap saturation at 0x6FFF
      f16 = f16 & 0x6FFF;
    }
  } else if (exponent < -30) {
    f16 = (f32 >> 16) & 0x8000;  // Zero with sign bit
  } else {
    f16 = ((f32 >> 16) & 0x8000) |
          static_cast<uint32_t>(exponent + 31) << 9u | fraction;
  }
  return static_cast<uint16_t>(f16);
}

// Mirrors spyre::encodeConstant in csrc/module.cpp (the full runtime
// build's torch_spyre._C.encode_constant).
uint32_t encode_constant(float torch_const, DataFormats df) {
  if (df == DataFormats::IEEE_FP32) {
    return float_to_bits(torch_const);
  }
  return float_to_fp16_bin(torch_const);
}


TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(spyre_codegen_empty));
  m.impl("empty_strided", TORCH_FN(spyre_codegen_empty_strided));
  m.impl("copy_", TORCH_FN(spyre_codegen_copy_));
  m.impl("set_.source_Storage_storage_offset",
         TORCH_FN(spyre_codegen_set_storage));
}

}  // namespace
}  // namespace spyre::codegen

PYBIND11_MODULE(_codegen_backend, m) {
  static auto* hooks = new spyre::codegen::SpyreCodegenHooksInterface();
  at::RegisterPrivateUse1HooksInterface(hooks);

  m.doc() = "Codegen-only Spyre PrivateUse1 backend registration";

  py::enum_<spyre::codegen::DataFormats>(m, "DataFormats")
      .value("SEN169_FP16", spyre::codegen::DataFormats::SEN169_FP16)
      .value("IEEE_FP32", spyre::codegen::DataFormats::IEEE_FP32)
      .value("INVALID", spyre::codegen::DataFormats::INVALID)
      .value("SEN143_FP8", spyre::codegen::DataFormats::SEN143_FP8)
      .value("SEN152_FP8", spyre::codegen::DataFormats::SEN152_FP8)
      .value("SEN153_FP9", spyre::codegen::DataFormats::SEN153_FP9)
      .value("SENINT2", spyre::codegen::DataFormats::SENINT2)
      .value("SENINT4", spyre::codegen::DataFormats::SENINT4)
      .value("SENINT8", spyre::codegen::DataFormats::SENINT8)
      .value("SENINT16", spyre::codegen::DataFormats::SENINT16)
      .value("SENINT24", spyre::codegen::DataFormats::SENINT24)
      .value("IEEE_INT64", spyre::codegen::DataFormats::IEEE_INT64)
      .value("IEEE_INT32", spyre::codegen::DataFormats::IEEE_INT32)
      .value("SENUINT32", spyre::codegen::DataFormats::SENUINT32)
      .value("SENUINT2", spyre::codegen::DataFormats::SENUINT2)
      .value("IEEE_FP16", spyre::codegen::DataFormats::IEEE_FP16)
      .value("BOOL", spyre::codegen::DataFormats::BOOL)
      .value("BFLOAT16", spyre::codegen::DataFormats::BFLOAT16)
      .value("SEN18F_FP24", spyre::codegen::DataFormats::SEN18F_FP24)
      .def("elems_per_stick",
           [](const spyre::codegen::DataFormats& df) {
             return spyre::codegen::elems_per_stick_for_format(df);
           });

  py::class_<spyre::codegen::SpyreTensorLayout>(m, "SpyreTensorLayout")
      .def_readonly("device_size", &spyre::codegen::SpyreTensorLayout::device_size)
      .def_readonly("stride_map", &spyre::codegen::SpyreTensorLayout::stride_map)
      .def_readonly("device_dtype", &spyre::codegen::SpyreTensorLayout::device_dtype)
      .def(py::init<std::vector<int64_t>, c10::ScalarType>(),
           py::arg("host_size"), py::arg("dtype"))
      .def(py::init<std::vector<int64_t>, std::vector<int64_t>, c10::ScalarType,
                    std::vector<int32_t>>(),
           py::arg("host_size"), py::arg("host_strides"), py::arg("dtype"),
           py::arg("dim_order"))
      .def(py::init<std::vector<int64_t>, std::vector<int64_t>,
                    spyre::codegen::DataFormats>(),
           py::arg("device_size"), py::arg("stride_map"),
           py::arg("device_dtype"))
      .def("elems_per_stick", &spyre::codegen::SpyreTensorLayout::elems_per_stick)
      .def("__str__",
           [](const spyre::codegen::SpyreTensorLayout& layout) {
             return layout.toString();
           })
      .def("__repr__",
           [](const spyre::codegen::SpyreTensorLayout& layout) {
             return layout.toString();
           })
      .def(py::self == py::self)
      .def("__hash__", [](const spyre::codegen::SpyreTensorLayout& layout) {
        size_t seed = 0;
        for (auto v : layout.device_size) {
          seed = c10::hash_combine(seed, std::hash<int64_t>{}(v));
        }
        for (auto v : layout.stride_map) {
          seed = c10::hash_combine(seed, std::hash<int64_t>{}(v));
        }
        return c10::hash_combine(
            seed, std::hash<int>{}(static_cast<int>(layout.device_dtype)));
      });

  m.def("device_count", []() { return 1; });
  m.def("current_device", []() { return spyre::codegen::current_device_index; });
  m.def("set_device", [](int idx) {
    TORCH_CHECK(idx == 0, "codegen-only Spyre backend exposes only fake device 0");
    spyre::codegen::current_device_index =
        static_cast<c10::DeviceIndex>(idx);
  });
  m.def("get_device_dtype", &spyre::codegen::get_device_dtype);
  m.def("get_elem_in_stick", [](c10::ScalarType dtype) {
    return spyre::codegen::elems_per_stick_for_format(
        spyre::codegen::get_device_dtype(dtype));
  });
  m.def("spyre_empty_with_layout", &spyre::codegen::spyre_codegen_empty_with_layout);
  m.def("empty_with_layout", &spyre::codegen::py_empty_with_layout);
  m.def("get_spyre_tensor_layout", &spyre::codegen::get_spyre_tensor_layout);
  m.def("set_spyre_tensor_layout", &spyre::codegen::set_spyre_tensor_layout);
  m.def("copy_tensor", &spyre::codegen::copy_tensor, py::arg("self"),
        py::arg("dst"), py::arg("non_blocking") = false);
  m.def("encode_constant", &spyre::codegen::encode_constant,
        py::arg("value"), py::arg("dtype"));
}
