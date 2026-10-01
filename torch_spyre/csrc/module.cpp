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

#include "module.h"

#include <c10/core/ScalarType.h>
#include <pybind11/native_enum.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <util/sen_data_convert.h>
#include <util/sendefs.h>

#include <cstdint>
#include <cstdlib>     // std::getenv
#include <filesystem>  // NOLINT(build/c++17)
#include <flex/flex.hpp>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "launch_identity.h"
#include "logging.h"
#include "spyre_allocator.h"
#include "spyre_device_enum.h"
#include "spyre_guard.h"
#include "spyre_kernel.h"
#include "spyre_mem.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"
#include "spyre_views.h"
#include "types_mapping.h"

namespace fs = std::filesystem;

namespace spyre {

static constexpr int32_t kSpyreTensorLayoutPickleVersion = 2;

std::atomic<bool> g_downcast_warn_enabled{true};

bool get_downcast_warn_enabled() {
  return g_downcast_warn_enabled.load(std::memory_order_relaxed);
}

void set_downcast_warn_enabled(bool enabled) {
  g_downcast_warn_enabled.store(enabled, std::memory_order_relaxed);
}

// Optional: initialize from env at module init
static void init_from_env() {
  if (const char* v = std::getenv(SPYRE_DOWNCAST_ENV)) {
    // Accept 0/1, true/false, on/off
    std::string s(v);
    for (auto& c : s) c = std::tolower(c);
    bool enable = !(s == "0" || s == "false" || s == "off");
    g_downcast_warn_enabled.store(enable, std::memory_order_relaxed);
  }
}

void _startRuntime() {
  DEBUGINFO("starting runtime");
  // Determine logical device index with priority:
  //   1. tls_idx (non-zero) — set via explicit set_device() call
  //   2. LOCAL_RANK env var — set by torchrun per process
  //   3. 0 — single-device / non-torchrun default
  int logical_device_id = 0;
  int tls_idx = static_cast<int>(SpyreGuardImpl::tls_idx);
  if (tls_idx != 0) {
    logical_device_id = tls_idx;
  } else if (const char* lr = std::getenv("LOCAL_RANK")) {
    logical_device_id = std::atoi(lr);
  }

  const int num_devices = getVisibleDeviceCount();
  TORCH_CHECK(logical_device_id < num_devices,
              "Device index out of bounds. logical_device_id=",
              logical_device_id, ", number of visible devices=", num_devices);

  std::shared_ptr<Runtime> runtime;
  auto s = flex::initializeRuntime(&runtime, logical_device_id);
  init_from_env();
  if (runtime) {
    GlobalRuntime::set(runtime);
    DEBUGINFO(s);
    DEBUGINFO("runtime started with logical_device_id ", logical_device_id);
  } else {
    DEBUGINFO("runtime FAILED TO START.");
    throw std::runtime_error("Failed to initialize Spyre runtime. ");
  }
}
void startRuntime() {
  static std::once_flag flag;
  std::call_once(flag, _startRuntime);
}

void freeRuntime() { shutdownRuntime(); }

uint32_t encodeConstant(float torch_const, DataFormats df) {
  uint32_t sen_const;

  if (df == DataFormats::IEEE_FP32) {
    sen_const =
        deeptools::BinaryConvert<uint32_t>(static_cast<float>(torch_const));
  } else {
    sen_const = deeptools::FloatToFp16Bin(torch_const);
  }
  return sen_const;
}

int64_t get_elem_in_stick(c10::ScalarType torch_dtype) {
  auto str_type = torchScalarToString[torch_dtype];
  const auto [sen_dtype_cpu, sen_dtype_dev] =
      stringToDTDataFormatPair(str_type);
  return elems_per_stick(sen_dtype_dev);
}

DataFormats get_device_dtype(c10::ScalarType torch_dtype) {
  auto str_type = torchScalarToString[torch_dtype];
  const auto [sen_dtype_cpu, sen_dtype_dev] =
      stringToDTDataFormatPair(str_type);
  return sen_dtype_dev;
}

bool is_supported_dtype(c10::ScalarType dtype) {
  // TODO(kmehant,yoheiueda): Replace this heuristic with a reliable method to
  // determine supported dtypes. Using elems_per_stick can miss certain
  // unsupported dtypes. See #950
  DataFormats sen_dtype_dev = get_device_dtype(dtype);
  return sen_dtype_dev != DataFormats::INVALID &&
         elems_per_stick(sen_dtype_dev) > 0;
}
int device_count() {
  return getVisibleDeviceCount();
}

}  // namespace spyre

namespace py = pybind11;
namespace {

py::object jsonToPyObject(const nlohmann::json& value) {
  if (value.is_object()) {
    py::dict result;
    for (auto it = value.begin(); it != value.end(); ++it) {
      result[py::str(it.key())] = jsonToPyObject(it.value());
    }
    return std::move(result);
  }
  if (value.is_array()) {
    py::list result;
    for (const auto& item : value) {
      result.append(jsonToPyObject(item));
    }
    return std::move(result);
  }
  if (value.is_string()) {
    return py::str(value.get<std::string>());
  }
  if (value.is_boolean()) {
    return py::bool_(value.get<bool>());
  }
  if (value.is_number_unsigned()) {
    return py::int_(value.get<uint64_t>());
  }
  if (value.is_number_integer()) {
    return py::int_(value.get<int64_t>());
  }
  if (value.is_number_float()) {
    return py::float_(value.get<double>());
  }
  if (value.is_null()) {
    return py::none();
  }
  throw std::runtime_error("Unsupported nlohmann::json value type");
}

class RuntimeLaunchObservation final {
 public:
  explicit RuntimeLaunchObservation(nlohmann::json observation)
      : observation_(std::move(observation)) {}

  py::dict consumeForRuntimeAudit() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (consumed_) {
      throw std::runtime_error(
          "RuntimeLaunchObservation has already been consumed by a runtime "
          "identity audit");
    }
    consumed_ = true;
    return jsonToPyObject(observation_).cast<py::dict>();
  }

  // Native-only single-use consumption that returns the raw observation JSON for
  // in-extension validation. Shares the single-use flag with
  // consumeForRuntimeAudit() so an observation can be consumed exactly once by
  // either path, never both and never twice.
  nlohmann::json consumeRawForNativeValidation() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (consumed_) {
      throw std::runtime_error(
          "RuntimeLaunchObservation has already been consumed by a runtime "
          "identity audit");
    }
    consumed_ = true;
    return std::move(observation_);
  }

 private:
  std::mutex mutex_;
  nlohmann::json observation_;
  bool consumed_ = false;
};

// Extension-owned verdict of validating a runtime launch observation against
// trusted expected identity. There is no Python constructor: only the native
// validator below can hand one to Python, so Python cannot manufacture a
// verdict that reports origin_verified. Trust is rooted in that construction
// path (a genuine, pybind-registered observation single-use-consumed in C++),
// never in a mutable module attribute comparison.
class RuntimeLaunchIdentityVerdict final {
 public:
  RuntimeLaunchIdentityVerdict(bool origin_verified,
                               std::vector<std::string> mismatches,
                               nlohmann::json observation)
      : origin_verified_(origin_verified),
        mismatches_(std::move(mismatches)),
        observation_(std::move(observation)) {}

  bool originVerified() const { return origin_verified_; }

  py::list mismatches() const {
    py::list out;
    for (const auto& mismatch : mismatches_) {
      out.append(py::str(mismatch));
    }
    return out;
  }

  py::dict observation() const {
    return jsonToPyObject(observation_).cast<py::dict>();
  }

 private:
  bool origin_verified_;
  std::vector<std::string> mismatches_;
  nlohmann::json observation_;
};

[[noreturn]] py::object rejectRuntimeLaunchPickle() {
  // Pickling is reconstruction from a byte stream: it would let a forged
  // observation/verdict be minted offline, outside the native launch path that
  // is the sole root of trust. copy/deepcopy are intentionally NOT routed here
  // -- each type routes them to its own compatibility binding below (the
  // observation consumes itself into a plain dict; the verdict aliases), so a
  // caller that copies a launch result keeps working. Only reconstruction from
  // a byte stream is refused.
  throw py::type_error(
      "this native runtime object is single-use and cannot be reconstructed or "
      "pickled; use it in-process");
}

}  // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() = "Spyre C++ bindings";
  m.def("start_runtime", &spyre::startRuntime);
  m.def("free_runtime", &spyre::freeRuntime);
  m.def("launch_kernel", &spyre::launchKernel);
  py::class_<RuntimeLaunchObservation, std::shared_ptr<RuntimeLaunchObservation>>(
      m, "RuntimeLaunchObservation")
      .def("consume_for_runtime_audit",
           &RuntimeLaunchObservation::consumeForRuntimeAudit)
      // copy.copy / copy.deepcopy single-use-CONSUME the opaque observation into
      // a plain diagnostic dict (the observation's own JSON fields). The Triton
      // launcher deep-copies the launch result and then indexes it as a mapping
      // (observation["launched_binary_sha256"] = ...; observation.get(...)); the
      // native object has no mapping protocol, so returning the native object or
      // an alias would still break that caller. Returning the consumed dict
      // preserves it, and any subsequent deepcopy of that dict is an ordinary
      // dict copy.
      //
      // This CANNOT authenticate provenance. The result is a plain mapping, not
      // the pybind-registered native type, so it can never reach the native
      // validator; and because the copy consumed the single-use original, any
      // later native validation of the original fails closed. A genuine verdict
      // is therefore obtainable only by validating the native original BEFORE it
      // is copied. consumeForRuntimeAudit() shares the single-use flag with the
      // native validator's consume, so exactly one of the two can ever run.
      // Signatures match CPython's calls: __copy__(self); __deepcopy__(self, memo).
      .def("__copy__",
           [](RuntimeLaunchObservation& self) {
             return self.consumeForRuntimeAudit();
           })
      .def("__deepcopy__",
           [](RuntimeLaunchObservation& self, py::object) {
             return self.consumeForRuntimeAudit();
           })
      // Pickling stays rejected, with signatures matching CPython's calls
      // (__reduce__(self); __reduce_ex__(self, protocol)) so it fails as a clean
      // TypeError rather than an arity error.
      .def("__reduce__", [](py::object) { return rejectRuntimeLaunchPickle(); })
      .def("__reduce_ex__",
           [](py::object, int) { return rejectRuntimeLaunchPickle(); });
  py::class_<RuntimeLaunchIdentityVerdict,
             std::shared_ptr<RuntimeLaunchIdentityVerdict>>(
      m, "RuntimeLaunchIdentityVerdict")
      .def_property_readonly("origin_verified",
                             &RuntimeLaunchIdentityVerdict::originVerified)
      .def("mismatches", &RuntimeLaunchIdentityVerdict::mismatches)
      .def("observation", &RuntimeLaunchIdentityVerdict::observation)
      // copy.copy / copy.deepcopy alias the same extension-owned verdict, with
      // correct __copy__(self) / __deepcopy__(self, memo) signatures so a caller
      // that deep-copies a verdict gets the same object rather than a TypeError.
      // Unlike the observation (whose copy consumes it into a plain dict), the
      // verdict carries no single-use state, so aliasing is a safe compatibility
      // choice; it still cannot be Python-manufactured. Pickling stays rejected.
      .def("__copy__",
           [](std::shared_ptr<RuntimeLaunchIdentityVerdict> self) {
             return self;
           })
      .def("__deepcopy__",
           [](std::shared_ptr<RuntimeLaunchIdentityVerdict> self, py::object) {
             return self;
           })
      .def("__reduce__", [](py::object) { return rejectRuntimeLaunchPickle(); })
      .def("__reduce_ex__",
           [](py::object, int) { return rejectRuntimeLaunchPickle(); });
  m.def(
      "validate_runtime_launch_observation",
      [](std::shared_ptr<RuntimeLaunchObservation> observation,
         const std::string& expected_emitted_sha256,
         const std::string& expected_target,
         const std::string& expected_generation,
         const std::string& expected_cache_identity) {
        // The std::shared_ptr<RuntimeLaunchObservation> parameter is the
        // unforgeable native gate: pybind casts the argument against the genuine
        // registered type, so a monkeypatched _C.RuntimeLaunchObservation
        // attribute or a pure-Python look-alike cannot reach this body. The
        // object is single-use-consumed natively before validation.
        //
        // Fail closed on a null holder. pybind can convert Python None to a null
        // std::shared_ptr even with .none(false) on some code paths/holder
        // conversions, and dereferencing it below would call a method on a null
        // object (the first act of which is a mutex lock) and crash the process.
        // Reject it explicitly so the boundary raises a catchable Python
        // exception instead of segfaulting.
        if (!observation) {
          throw py::type_error(
              "validate_runtime_launch_observation requires a native "
              "RuntimeLaunchObservation produced by launch_kernel_from_bytes, "
              "not None");
        }
        nlohmann::json consumed =
            observation->consumeRawForNativeValidation();
        spyre::LaunchIdentityVerdict verdict =
            spyre::validate_launch_identity_observation(
                consumed, expected_emitted_sha256, expected_target,
                expected_generation, expected_cache_identity);
        return std::make_shared<RuntimeLaunchIdentityVerdict>(
            verdict.origin_verified, std::move(verdict.mismatches),
            std::move(consumed));
      },
      py::arg("observation").none(false), py::arg("expected_emitted_sha256"),
      py::arg("expected_target"), py::arg("expected_generation"),
      py::arg("expected_cache_identity"),
      "Natively single-use-consume and validate an opaque runtime launch "
      "observation against trusted expected identity. Returns an extension-owned "
      "verdict that Python cannot manufacture; origin_verified is true only when "
      "every identity leg matches.");
  m.def(
      "launch_kernel_from_bytes",
      [](py::bytes binary_data, py::list args, const std::string& target,
         const std::string& generation) {
        // Convert py::bytes -> std::vector<uint8_t>
        std::string raw = binary_data;
        std::vector<uint8_t> binary(raw.begin(), raw.end());

        // Convert py::list of tensors -> std::vector<at::Tensor>
        std::vector<at::Tensor> tensor_args;
        tensor_args.reserve(args.size());
        for (auto& item : args) {
          tensor_args.push_back(item.cast<at::Tensor>());
        }

        return std::make_shared<RuntimeLaunchObservation>(
            spyre::launchKernelFromBytes(binary, tensor_args, target, generation));
      },
      py::arg("binary"), py::arg("args"),
      py::arg("target") = std::string(spyre::kDefaultTarget),
      py::arg("generation") = std::string(spyre::kDefaultGeneration),
      "Launch a kernel from raw init packet bytes and return its single-use "
      "native runtime observation.");
  m.def("get_last_launch_observation", &spyre::getLastLaunchObservation);
  m.def("encode_constant", &spyre::encodeConstant);

  py::class_<spyre::SpyreTensorLayout> dci_cls(m, "SpyreTensorLayout");

  dci_cls.def_readonly("device_size", &spyre::SpyreTensorLayout::device_size)
      .def_readonly("stride_map", &spyre::SpyreTensorLayout::stride_map)
      .def_readonly("device_dtype", &spyre::SpyreTensorLayout::device_dtype)
      .def("__str__",
           [](const spyre::SpyreTensorLayout& c) { return c.toString(); })
      .def("__repr__",
           [](const spyre::SpyreTensorLayout& c) { return c.toString(); })
      .def("elems_per_stick", &spyre::SpyreTensorLayout::elems_per_stick)
      .def(py::self == py::self)
      .def("__hash__",
           [](const spyre::SpyreTensorLayout& c) {
             return std::hash<spyre::SpyreTensorLayout>{}(c);
           })
      .def(py::init<std::vector<int64_t>, c10::ScalarType>(),
           py::arg("host_size"), py::arg("dtype"))
      .def(py::init<std::vector<int64_t>, std::vector<int64_t>, c10::ScalarType,
                    std::vector<int32_t>>(),
           py::arg("host_size"), py::arg("host_strides"), py::arg("dtype"),
           py::arg("dim_order"))
      .def(py::init<std::vector<int64_t>, std::vector<int64_t>, DataFormats>(),
           py::arg("device_size"), py::arg("stride_map"),
           py::arg("device_dtype"))
      .def(py::pickle(
          [](const spyre::SpyreTensorLayout& p) {  // __getstate__
            // Return a tuple that fully encodes the state of the object
            // If the pickle format changes, then update
            // kSpyreTensorLayoutPickleVersion but keep the tuple as the
            // returned object and the first element to be the
            // kSpyreTensorLayoutPickleVersion
            return py::make_tuple(spyre::kSpyreTensorLayoutPickleVersion,
                                  p.device_size, p.stride_map, p.device_dtype);
          },
          [](py::tuple t) {  // __setstate__
            int32_t version = t[0].cast<int32_t>();
            if (version == 1) {
              // Version 1 had: (version, device_size, dim_map, stride_map,
              // device_dtype) — discard dim_map
              if (t.size() != 5) {
                throw py::value_error(
                    "Invalid SpyreTensorLayout pickle v1: wrong tuple size");
              }
              return spyre::SpyreTensorLayout(t[1].cast<std::vector<int64_t>>(),
                                              t[3].cast<std::vector<int64_t>>(),
                                              t[4].cast<DataFormats>());
            } else if (version == 2) {
              // Version 2: (version, device_size, stride_map, device_dtype)
              if (t.size() != 4) {
                throw py::value_error(
                    "Invalid SpyreTensorLayout pickle v2: wrong tuple size");
              }
              return spyre::SpyreTensorLayout(t[1].cast<std::vector<int64_t>>(),
                                              t[2].cast<std::vector<int64_t>>(),
                                              t[3].cast<DataFormats>());
            } else {
              throw py::value_error(
                  "Unsupported SpyreTensorLayout pickle version: " +
                  std::to_string(version));
            }
          }));

  m.def("spyre_empty_with_layout", &spyre::spyre_empty_with_layout);
  m.def("empty_with_layout", &spyre::py_empty_with_layout);
  m.def("as_strided_with_layout", &spyre::as_strided_with_layout);
  m.def("reinterpret_tensor", &spyre::reinterpret_tensor);
  m.def("reinterpret_tensor_with_layout",
        &spyre::reinterpret_tensor_with_layout);

  py::enum_<DataFormats>(m, "DataFormats")
      .value("SEN169_FP16", DataFormats::SEN169_FP16)
      .value("IEEE_FP32", DataFormats::IEEE_FP32)
      .value("INVALID", DataFormats::INVALID)
      .value("SEN143_FP8", DataFormats::SEN143_FP8)
      .value("SEN152_FP8", DataFormats::SEN152_FP8)
      .value("SEN153_FP9", DataFormats::SEN153_FP9)
      .value("SENINT2", DataFormats::SENINT2)
      .value("SENINT4", DataFormats::SENINT4)
      .value("SENINT8", DataFormats::SENINT8)
      .value("SENINT16", DataFormats::SENINT16)
      .value("SENINT24", DataFormats::SENINT24)
      .value("IEEE_INT64", DataFormats::IEEE_INT64)
      .value("IEEE_INT32", DataFormats::IEEE_INT32)
      .value("SENUINT32", DataFormats::SENUINT32)
      .value("SENUINT2", DataFormats::SENUINT2)
      .value("IEEE_FP16", DataFormats::IEEE_FP16)
      .value("BOOL", DataFormats::BOOL)
      .value("BFLOAT16", DataFormats::BFLOAT16)
      .value("SEN18F_FP24", DataFormats::SEN18F_FP24)
      .def("elems_per_stick",
           [](const DataFormats& df) { return spyre::elems_per_stick(df); });

  m.def("get_spyre_tensor_layout", &spyre::get_spyre_tensor_layout);
  m.def("set_spyre_tensor_layout", &spyre::set_spyre_tensor_layout);
  m.def("get_downcast_warning", &spyre::get_downcast_warn_enabled,
        "Return whether downcast warnings are enabled.");
  m.def("set_downcast_warning", &spyre::set_downcast_warn_enabled,
        "Enable/disable downcast warnings for this process.");
  m.def("get_elem_in_stick", &spyre::get_elem_in_stick);
  m.def("get_device_dtype", &spyre::get_device_dtype);

  // Memory copy function
  m.def("copy_tensor", &spyre::spyre_copy_from,
        "Copy tensor between host and device using DMA", py::arg("self"),
        py::arg("dst"), py::arg("non_blocking") = false);

  // Device-side fill using FillDMA (no host buffer or H2D copy)
  m.def("fill_tensor", &spyre::spyre_fill_tensor,
        "Fill a spyre tensor with a scalar value using device-side FillDMA",
        py::arg("self"), py::arg("value"));

  // Stream management functions
  m.def("get_stream_from_pool", &spyre::getStreamFromPool, py::arg("device"),
        py::arg("priority") = 0,
        "Get a stream from the pool with specified device and priority");

  m.def("current_stream", &spyre::getCurrentStream, py::arg("device"),
        "Get the current stream for a device");

  m.def("set_current_stream", &spyre::setCurrentStream, py::arg("stream"),
        "Set the current stream and return the previous one");

  m.def("default_stream", &spyre::getDefaultStream, py::arg("device"),
        "Get the default stream for a device");

  m.def("synchronize", &spyre::synchronizeDevice,
        py::arg("device") = py::none(), "Synchronize a device or all devices");

  // Expose SpyreStream class to Python
  py::class_<spyre::SpyreStream>(m, "_SpyreStreamBase")
      .def("synchronize", &spyre::SpyreStream::synchronize,
           "Wait for all operations on this stream to complete")
      .def("query", &spyre::SpyreStream::query,
           "Check if all operations on this stream have completed")
      .def("device", &spyre::SpyreStream::device,
           "Get the device associated with this stream")
      .def("id", &spyre::SpyreStream::id, "Get the stream ID")
      .def("priority", &spyre::SpyreStream::priority, "Get the stream priority")
      .def("__repr__", [](const spyre::SpyreStream& stream) {
        return "<torch_spyre.Stream device=" +
               std::to_string(stream.device().index()) +
               " id=" + std::to_string(stream.id()) + ">";
      });
  m.def("set_device", [](int idx) {
    int count = spyre::device_count();
    TORCH_CHECK(idx >= 0 && idx < count, "Device index ", idx,
                " out of range [0, ", count, ")");
    c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
        ->setDevice(c10::Device(c10::DeviceType::PrivateUse1,
                                static_cast<c10::DeviceIndex>(idx)));
  });
  m.def("current_device", []() {
    return c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
        ->getDevice()
        .index();
  });
  m.def("device_count", &spyre::device_count);
}
