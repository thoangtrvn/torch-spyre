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

#include "spyre_kernel.h"

#include <c10/util/Exception.h>

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "key_mutex_registry.h"
#include "launch_identity.h"
#include "logging.h"
#include "spyre_allocator.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"

namespace fs = std::filesystem;

namespace {

std::mutex g_last_launch_observation_mtx;
json g_last_launch_observation;

constexpr uint64_t kSegmentSizeBytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr uint64_t kSegmentOffsetMask = kSegmentSizeBytes - 1ULL;
constexpr uint64_t kReadWriteXlatMode = 3ULL;
constexpr uint64_t kDeviceAlignmentBytes = 128ULL;

std::string hex64(uint64_t value) {
  std::ostringstream os;
  os << std::hex << std::setw(16) << std::setfill('0') << value;
  return os.str();
}

std::string fnv1a256(const std::vector<uint8_t>& bytes) {
  // Non-cryptographic placeholder with SHA-256-shaped output. It is diagnostic
  // only until torch-spyre links a real SHA-256 implementation.
  uint64_t h0 = 0xcbf29ce484222325ULL;
  uint64_t h1 = 0x84222325cbf29ce4ULL;
  uint64_t h2 = 0x9e3779b97f4a7c15ULL;
  uint64_t h3 = 0x100000001b3ULL;
  for (uint8_t b : bytes) {
    h0 ^= static_cast<uint64_t>(b);
    h0 *= 0x100000001b3ULL;
    h1 ^= static_cast<uint64_t>(b) + (h0 << 6) + (h0 >> 2);
    h1 *= 0x100000001b3ULL;
    h2 ^= static_cast<uint64_t>(b) + (h1 << 7) + (h1 >> 3);
    h2 *= 0x100000001b3ULL;
    h3 ^= static_cast<uint64_t>(b) + (h2 << 8) + (h2 >> 4);
    h3 *= 0x100000001b3ULL;
  }
  return "fnv256:" + hex64(h0) + hex64(h1) + hex64(h2) + hex64(h3);
}

std::string contractDeviceDtype(const DataFormats& dtype) {
  switch (dtype) {
    case DataFormats::SEN169_FP16:
      return "DL16";
    case DataFormats::IEEE_FP32:
      return "Float";
    case DataFormats::BFLOAT16:
      return "BF16";
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
      return "Int64";
    case DataFormats::IEEE_INT32:
      return "Int";
    case DataFormats::SENUINT32:
      return "UInt32";
    case DataFormats::SENUINT2:
      return "SENUINT2";
    case DataFormats::IEEE_FP16:
      return "Half";
    case DataFormats::BOOL:
      return "Bool";
    case DataFormats::SEN18F_FP24:
      return "SEN18F_FP24";
    case DataFormats::INVALID:
      return "INVALID";
  }
  return "UNKNOWN";
}

json tensorLaunchObservation(const at::Tensor& tensor, size_t index) {
  json observation;
  observation["arg_index"] = index;
  observation["selected_segment_index"] = index;
  observation["storage_offset_bytes"] = tensor.storage_offset() * tensor.element_size();
  observation["device_bytes"] = tensor.nbytes();
  observation["torch_scalar_type"] = c10::toString(tensor.scalar_type());
  observation["device_dtype"] = c10::toString(tensor.scalar_type());
  observation["alignment_bytes"] = kDeviceAlignmentBytes;
  observation["allocation_base_observed"] = false;
  observation["effective_pointer_alignment_proven"] = false;

  // Validate the argument is a Spyre (private-use) tensor BEFORE interpreting its
  // storage deleter context as a SharedOwnerCtx. A CPU / non-Spyre tensor carries
  // a different (or null) deleter context; casting it to SharedOwnerCtx* and
  // dereferencing composite_addr below would be undefined behavior, and the
  // nullptr check alone does not cover a non-null non-Spyre context. Fail closed
  // with a typed status before any such interpretation.
  if (!tensor.is_privateuseone()) {
    observation["allocation_observation_status"] =
        "non_spyre_tensor_argument_unobserved";
    return observation;
  }

  const auto layout = spyre::get_spyre_tensor_layout(tensor);
  observation["device_encoding_dtype"] = contractDeviceDtype(layout.device_dtype);
  observation["device_encoding_data_format"] =
      static_cast<int64_t>(layout.device_dtype);
  observation["device_encoding_proof_source"] =
      "torch_spyre.spyre_tensor_allocation";
  observation["device_encoding_layout"] = {
      {"device_size", layout.device_size},
      {"stride_map", layout.stride_map},
  };

  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      tensor.storage().data_ptr().get_context());
  if (ctx == nullptr || !ctx->composite_addr.is_single_chunk()) {
    observation["allocation_observation_status"] =
        ctx == nullptr ? "missing_spyre_allocation_context"
                       : "multi_chunk_allocation_unobserved";
    return observation;
  }

  const auto& chunk = ctx->composite_addr.chunks().at(0);
  const uint64_t allocation_base = chunk.addr.region_id + chunk.addr.offset;
  const uint64_t allocation_end = allocation_base + chunk.size;
  const uint64_t storage_offset_bytes = observation["storage_offset_bytes"];
  const uint64_t effective_pointer = allocation_base + storage_offset_bytes;

  observation["allocation_base"] = allocation_base;
  observation["allocation_end"] = allocation_end;
  observation["allocation_region_id"] = chunk.addr.region_id;
  observation["allocation_region_offset"] = chunk.addr.offset;
  observation["allocation_chunk_size"] = chunk.size;
  observation["allocation_domain_id"] = chunk.domain_id;
  observation["allocation_total_size"] = ctx->composite_addr.total_size();
  observation["allocation_base_observed"] = true;
  observation["effective_pointer"] = effective_pointer;
  observation["effective_pointer_alignment_proven"] =
      effective_pointer % kDeviceAlignmentBytes == 0;
  observation["xlat_xa"] = allocation_base;
  observation["xlat_xl"] = kSegmentSizeBytes;
  observation["xlat_xm"] = index;
  observation["xlat_slot_index"] = index;
  observation["xlat"] = {
      {"XA", allocation_base},
      {"XL", kSegmentSizeBytes},
      {"XM", index},
      {"mode", kReadWriteXlatMode},
  };
  observation["xlat_realization_source"] =
      "flex_graph_free_compute_tensor_paddrs";
  observation["xlat_xm_semantics"] = "runtime_translation_slot_index";
  observation["xlat_length_semantics"] = "flex_graph_free_segment_size";
  observation["xlat_base_uses_storage_offset"] = false;
  observation["xlat_segment_size_bytes"] = kSegmentSizeBytes;
  observation["xlat_segment_offset"] = allocation_base & kSegmentOffsetMask;
  return observation;
}

// This global is a diagnostic-only mirror. The direct invocation return value is
// the trusted in-process provenance for a byte launch.
void recordLastLaunchObservation(json observation) {
  std::lock_guard<std::mutex> lock(g_last_launch_observation_mtx);
  g_last_launch_observation = std::move(observation);
}

}  // namespace

namespace spyre {
std::ostream& operator<<(std::ostream& os, const KernelArtifacts& k) {
  os << "KernelArtifacts {\n";
  os << "  init_bin.size       = " << k.init_bin.size() << " bytes\n";
  os << "  program_size        = " << k.program_size << " bytes\n";
  os << "  bundle_mlir_path    = \"" << k.bundle_mlir_path << "\"\n";
  os << "}";
  return os;
}

std::vector<uint8_t> readHexEncodedFile(const std::string& filepath) {
  // Slurp entire file into memory in one I/O call
  std::ifstream inpFile(filepath, std::ios::in | std::ios::binary);
  if (!inpFile.is_open()) {
    throw std::runtime_error("Failed to open file: " + filepath);
  }
  std::vector<char> buf;
  inpFile.seekg(0, std::ios::end);
  buf.resize(static_cast<size_t>(inpFile.tellg()));
  inpFile.seekg(0, std::ios::beg);
  inpFile.read(buf.data(), static_cast<std::streamsize>(buf.size()));

  auto c2u = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return 0;
  };

  std::vector<uint8_t> binary_data;
  binary_data.reserve(buf.size() / 257 * 128);

  const char* p = buf.data();
  const char* end = p + buf.size();

  while (p < end) {
    // Skip newlines
    if (*p == '\n' || *p == '\r') {
      p++;
      continue;
    }
    // Skip comment lines
    if (*p == '#') {
      while (p < end && *p != '\n') p++;
      continue;
    }

    // We're at the start of a hex line — it must be exactly 256 chars
    const char* line_start = p;
    const char* line_end = p;
    while (line_end < end && *line_end != '\n' && *line_end != '\r') {
      line_end++;
    }
    if (line_end - line_start != 256) {
      throw std::runtime_error(
          "In readHexEncodedFile, line is not 256 chars in " + filepath);
    }

    auto rit = std::make_reverse_iterator(line_end);
    auto rend = std::make_reverse_iterator(line_start);
    while (rit != rend) {
      uint8_t byte = static_cast<uint8_t>(c2u(*rit++) + (c2u(*rit++) << 4));
      binary_data.push_back(byte);
    }

    p = line_end;
  }

  if (binary_data.empty()) {
    throw std::runtime_error("No data decoded from file: " + filepath);
  }

  return binary_data;
}

std::vector<uint8_t> readBinaryFile(const std::string& filepath) {
  // Read a raw binary file in hardware-native byte order.
  // No byte swapping needed — the file layout matches device memory exactly.
  std::ifstream inpFile(filepath, std::ios::in | std::ios::binary);
  if (!inpFile.is_open()) {
    throw std::runtime_error("Failed to open binary file: " + filepath);
  }
  inpFile.seekg(0, std::ios::end);
  auto size = static_cast<size_t>(inpFile.tellg());
  inpFile.seekg(0, std::ios::beg);

  if (size == 0) {
    throw std::runtime_error("Empty binary file: " + filepath);
  }
  if (size % 128 != 0) {
    throw std::runtime_error(
        "Binary file size must be multiple of 128 bytes (row granularity), got " +
        std::to_string(size) + " bytes in " + filepath);
  }

  std::vector<uint8_t> data(size);
  inpFile.read(reinterpret_cast<char*>(data.data()),
               static_cast<std::streamsize>(size));
  return data;
}

std::string get_init_path(const std::string& code_dir) {
  fs::path dir(code_dir);
  std::string kernel_name = dir.filename().string();

  std::string program_dir =
      "loadprogram_to_device/" + kernel_name + "-SenProgSend";

  return (dir / program_dir).string();
}

std::string get_pagi_path(const std::string& code_dir) {
  fs::path dir(code_dir);
  std::string kernel_name = dir.filename().string();

  std::string program_dir = "execute/" + kernel_name;

  return (dir / program_dir).string();
}

// Process-wide runtime launch/teardown lifecycle gate.
//
// Every kernel launch (launchKernel, launchKernelFromBytes) holds this in SHARED
// mode across its entire body — cache lookup, program copy, and the
// executeProgramAsync enqueue. Teardown (clearArtifactCache, shutdownRuntime)
// takes it in EXCLUSIVE mode. The exclusive lock cannot be granted until every
// in-flight launch has returned and released its shared lease, so once teardown
// owns the gate no launch is mid-enqueue; teardown then drains the active Spyre
// device's streams (synchronizeDevice(nullopt) synchronizes the default stream
// plus the TLS-current device's priority stream pools) so no enqueued DMA/compute
// still references a program allocation before it frees any. This relies on
// torch-spyre's single-active-device-per-process model (SpyreGuardImpl::tls_idx);
// it is NOT a process-wide every-device drain, and the drain scope must be
// revisited if a multi-device runtime is ever supported. shutdownRuntime holds
// the same exclusion across GlobalRuntime::reset() — which only drops torch-spyre's
// shared_ptr handle to the flex::RuntimeContext (it does not reinitialize or reset
// the Flex singleton itself) — so no launch can enter between the cache clear and
// that handle release. This is a conservative synchronous barrier built only from
// the existing SpyreStream::synchronize()/synchronizeDevice() primitives; it does
// not depend on a per-op completion event.
std::shared_mutex g_runtime_lifecycle_gate;

// Cache: code_dir -> artifacts (loaded once). The value is a shared_ptr so a
// launch holds the artifact (and its owning device allocation) alive through
// executeProgramAsync via a shared lease, rather than a bare reference into the
// map that a concurrent clear could invalidate.
std::unordered_map<std::string, std::shared_ptr<KernelArtifacts>>
    g_artifact_cache;
std::shared_mutex g_artifact_cache_mtx;  // protects g_artifact_cache
// Lifetime-safe per-key mutexes: a loading thread holds a shared_ptr handle for
// its whole critical section, so a concurrent cache clear cannot destroy the
// mutex mid-load.
KeyMutexRegistry g_key_registry;

std::shared_ptr<KernelArtifacts> getOrLoadArtifacts(const std::string& code_dir,
                                                    const SpyreStream& stream) {
  // Fast path: return a cached entry (as a shared lease) without creating a
  // per-key mutex.
  {
    std::shared_lock<std::shared_mutex> lock(g_artifact_cache_mtx);
    auto it = g_artifact_cache.find(code_dir);
    if (it != g_artifact_cache.end()) {
      return it->second;
    }
  }
  // Per-key lock: only one thread loads a given key. The returned shared_ptr
  // keeps the mutex alive for the whole critical section even if a concurrent
  // cache clear erases the registry entry.
  std::shared_ptr<std::mutex> key_mtx = g_key_registry.acquire(code_dir);
  std::lock_guard<std::mutex> key_lock(*key_mtx);

  // Double-check after acquiring per-key lock
  {
    std::shared_lock<std::shared_mutex> lock(g_artifact_cache_mtx);
    auto it = g_artifact_cache.find(code_dir);
    if (it != g_artifact_cache.end()) {
      return it->second;
    }
  }

  auto arts = std::make_shared<KernelArtifacts>();

  fs::path dir(code_dir);
  std::string bundle_path = (dir / "bundle.mlir").string();
  // Store bundle.mlir path for future JIT compilation
  arts->bundle_mlir_path = bundle_path;
  TORCH_CHECK(std::filesystem::exists(bundle_path),
              "Bundle not found: ", bundle_path);

  // Read init packet binary — prefer raw binary (.bin) over hex (.txt)
  std::string init_dir = get_init_path(code_dir);
  std::string init_bin_path = init_dir + "/init.bin";
  std::string init_txt_path = init_dir + "/init.txt";
  if (std::filesystem::exists(init_bin_path)) {
    arts->init_bin = readBinaryFile(init_bin_path);
  } else {
    arts->init_bin = readHexEncodedFile(init_txt_path);
  }

  arts->program_size = arts->init_bin.size();
  auto& allocator = SpyreAllocator::instance();
  flex::AllocationDirective directive(flex::PlacementPolicy::Bind, {0},
                                      std::nullopt, flex::MemoryType::Program);
  arts->device_alloc =
      std::move(allocator.allocate(arts->program_size, directive));
  auto* ctx = static_cast<SharedOwnerCtx*>(arts->device_alloc.get_context());
  TORCH_CHECK(arts->program_size <= ctx->composite_addr.total_size(),
              "Program size (", arts->program_size,
              ") exceeds allocated device memory (",
              ctx->composite_addr.total_size(), ")");
  stream.copyProgramAsync(arts->init_bin.data(), &ctx->composite_addr,
                          arts->program_size);
  // Drain the asynchronous program copy BEFORE the entry is published, so a
  // concurrent cache hit on another stream cannot select and execute this
  // program before its device bytes are written. If synchronize() throws, the
  // launch fails without ever inserting a not-yet-ready entry.
  stream.synchronize();

  std::unique_lock<std::shared_mutex> lock(g_artifact_cache_mtx);
  auto [it, inserted] = g_artifact_cache.emplace(code_dir, std::move(arts));
  if (inserted) {
    DEBUGINFO(*it->second);
  }
  return it->second;
}

void launchKernel(const std::string& code_dir,
                  const std::vector<at::Tensor>& args) {
  // Hold the runtime lifecycle gate shared for the whole launch so teardown
  // cannot drain/free program allocations while this launch is in flight.
  std::shared_lock<std::shared_mutex> launch_lease(g_runtime_lifecycle_gate);
  auto stream = getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1, -1));

  // The shared_ptr lease keeps the artifact's device allocation alive across the
  // asynchronous executeProgramAsync enqueue.
  std::shared_ptr<KernelArtifacts> arts = getOrLoadArtifacts(code_dir, stream);

  stream.executeProgramAsync(*arts, args);
}

// Cache for launchKernelFromBytes: full artifact-aware cache identity -> loaded
// artifacts. The key is the full-SHA-256 "sha256:<64 hex>" canonical identity
// over target/generation/binary (never a truncated 64-bit derivative), so two
// distinct artifacts cannot alias on a shortened key. The exact-byte guard
// before every submission remains the authoritative equality check.
// Avoids re-allocating and copying device memory on repeated launches
// of the same kernel (e.g., training loops).
//
// The value is a shared_ptr: a cache hit copies the shared_ptr out under the map
// lock and holds it through executeProgramAsync, so the artifact (and its owning
// device allocation) outlives the enqueue even if the map entry is dropped. A
// miss drains the program copy with SpyreStream::synchronize() BEFORE inserting
// the entry, so a concurrent cache hit on another stream can never select a
// program whose device bytes are not yet written. The process-wide runtime
// lifecycle gate (g_runtime_lifecycle_gate) provides the complementary teardown
// guarantee: clear/shutdown drains the active device's streams under exclusive
// ownership before freeing any program allocation.
static std::unordered_map<std::string, std::shared_ptr<KernelArtifacts>>
    g_bytes_artifact_cache;
static std::shared_mutex g_bytes_cache_mtx;  // protects g_bytes_artifact_cache
// Lifetime-safe per-key mutexes serialize who loads a given binary (one
// copyProgramAsync per key). A loading thread holds a shared_ptr handle for its
// whole critical section, so a concurrent cache clear cannot destroy the mutex
// out from under it.
static KeyMutexRegistry g_bytes_key_registry;

json addCachedBinaryObservation(const VerifiedByteIdentity& verified,
                                const std::string& target,
                                const std::string& generation,
                                bool program_copied_this_launch) {
  return build_launch_identity_observation(verified, target, generation,
                                           fnv1a256(verified.bytes()),
                                           program_copied_this_launch);
}

void addBaseLaunchObservation(json& observation,
                              const std::vector<uint8_t>& binary,
                              const std::vector<at::Tensor>& args,
                              const std::string& cache_status) {
  observation["schema_version"] = "spyre-launch-observation-v2";
  observation["source"] = "torch_spyre.launch_kernel_from_bytes";
  observation["launched_binary_size"] = binary.size();
  observation["launched_binary_fingerprint"] = fnv1a256(binary);
  observation["launched_binary_fingerprint_algorithm"] =
      "fnv1a256_noncryptographic";
  observation["launched_binary_fingerprint_proof_scope"] =
      "diagnostic_only_not_sha256_identity_proof";
  observation["tensor_arg_count"] = args.size();
  observation["launch_observations"] = json::array();
  for (size_t i = 0; i < args.size(); ++i) {
    observation["launch_observations"].push_back(
        tensorLaunchObservation(args[i], i));
  }
  observation["bytes_cache_status"] = cache_status;
}

json launchKernelFromBytes(const std::vector<uint8_t>& binary,
                           const std::vector<at::Tensor>& args,
                           const std::string& target,
                           const std::string& generation) {
  // Hold the runtime lifecycle gate shared for the whole launch (see
  // g_runtime_lifecycle_gate): teardown cannot drain or free program allocations
  // until this launch returns and releases the lease.
  std::shared_lock<std::shared_mutex> launch_lease(g_runtime_lifecycle_gate);

  // Validate the explicit launch context rather than silently trusting a
  // hardcoded label. RCUDD1A / AIU 1.0 is the only shipped byte-launch
  // generation; anything else is refused fail-closed.
  TORCH_CHECK(target == kDefaultTarget && generation == kDefaultGeneration,
              "Unsupported byte-launch target/generation: got target='", target,
              "', generation='", generation, "'; only target='", kDefaultTarget,
              "' / generation='", kDefaultGeneration,
              "' is a shipped byte-launch generation");

  // Full artifact-aware cache identity (full SHA-256 over
  // target/generation/binary), used as the cache key with no truncation.
  const std::string key = cache_identity_prefixed(binary, target, generation);

  // Asynchronous byte-launch correctness rests on two cooperating mechanisms,
  // both built only from existing SpyreStream primitives (no invented API):
  //   1. Readiness. A cache miss drains its program copy with
  //      SpyreStream::synchronize() BEFORE inserting the cache entry, so a
  //      concurrent cache hit on another stream can never select a program whose
  //      device bytes are not yet written.
  //   2. Lifetime. A cache hit copies the entry's shared_ptr out under the map
  //      lock and holds it through executeProgramAsync, keeping the artifact's
  //      device allocation alive across the asynchronous enqueue. Teardown
  //      (clearArtifactCache/shutdownRuntime) takes the lifecycle gate
  //      exclusively — which cannot be granted while any launch holds its shared
  //      lease — then drains the active device's streams (synchronizeDevice) before
  //      freeing any allocation, so no enqueued compute ever references freed
  //      program memory.
  // Source-level wiring alone does not establish runtime identity or hardware
  // authority: runtime identity requires the native validator on a genuine
  // observation, and hardware authority requires a device KAT.

  // Execute a resident cache entry. The shared_ptr lease keeps the artifact (and
  // its owning device allocation) alive across the asynchronous enqueue.
  auto run_cached_launch =
      [&](const std::shared_ptr<KernelArtifacts>& arts,
          const char* cache_status) -> json {
    if (!exact_bytes_equal(arts->init_bin, binary)) {
      TORCH_CHECK(false,
                  "Refusing byte launch: cached artifact bytes differ from "
                  "requested binary before executeProgramAsync");
    }
    // Cache hit: the device program was copied (and drained) by the original
    // miss; this launch selects the resident program and performs no fresh
    // copyProgramAsync, so program_copied_this_launch is false.
    VerifiedByteIdentity verified =
        VerifiedByteIdentity::verify(binary, arts->init_bin, arts->init_bin);
    auto stream = getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1, -1));
    json observation = addCachedBinaryObservation(
        verified, target, generation, /*program_copied_this_launch=*/false);
    addBaseLaunchObservation(observation, binary, args, cache_status);
    stream.executeProgramAsync(*arts, args);
    recordLastLaunchObservation(observation);
    return observation;
  };

  // Fast path: select a resident entry under the map lock and hold it alive for
  // this launch via the copied shared_ptr lease.
  {
    std::shared_lock<std::shared_mutex> lock(g_bytes_cache_mtx);
    auto it = g_bytes_artifact_cache.find(key);
    if (it != g_bytes_artifact_cache.end()) {
      std::shared_ptr<KernelArtifacts> arts = it->second;
      lock.unlock();
      return run_cached_launch(arts, "bytes_cache_hit");
    }
  }

  // Acquire the per-key mutex. The returned shared_ptr keeps the mutex alive for
  // the whole load even if a concurrent cache clear erases the registry entry.
  // Only one thread loads a given binary.
  std::shared_ptr<std::mutex> key_mtx = g_bytes_key_registry.acquire(key);
  std::lock_guard<std::mutex> key_lock(*key_mtx);

  // Double-check after acquiring the per-key lock: another thread may have loaded
  // this key while we waited.
  {
    std::shared_lock<std::shared_mutex> lock(g_bytes_cache_mtx);
    auto it = g_bytes_artifact_cache.find(key);
    if (it != g_bytes_artifact_cache.end()) {
      std::shared_ptr<KernelArtifacts> arts = it->second;
      lock.unlock();
      return run_cached_launch(arts, "bytes_cache_hit_after_wait");
    }
  }

  // Sanity checks before device allocation — a malformed binary can crash
  // or hang the C++ runtime. Init packets must be 32-byte aligned (flit
  // granularity) and at least one flit (64 bytes = header + SPR flit).
  TORCH_CHECK(binary.size() % 32 == 0,
              "Init packet size must be 32-byte aligned, got ", binary.size(),
              " bytes");
  TORCH_CHECK(binary.size() >= 64, "Init packet too small: ", binary.size(),
              " bytes (minimum 64 = header + SPR flit)");

  // Refuse the two-layer kernel container at the device boundary. A container
  // starts with the ASCII magic "SPYR" (first byte 'S', 0x53) where a raw
  // launchable init packet starts with the QGInit magic byte 0xDD, so the
  // discrimination is unambiguous. This boundary consumes RAW device payloads
  // only: the runtime unwrapper that extracts the Layer-2 payload (and reserves
  // the container's declared off-chip footprint) is not implemented yet and
  // lands atomically with the first consumer (kernel-container-format.md,
  // "Refusal surface"). The size checks above are NOT a reliable backstop —
  // whether a container passes %32 and >=64 depends on its total length mod 32
  // (a real wrapped matmul is caught by %32; other shapes are not). The
  // refusal is on the magic, not the size: without it, any container whose
  // length happens to be 32-aligned would be copied to the device verbatim,
  // launching 40 bytes of header as if it were a QGInit packet — silent
  // device corruption with zero diagnostic.
  TORCH_CHECK(binary.size() < 4 || binary[0] != 'S' || binary[1] != 'P' ||
                  binary[2] != 'Y' || binary[3] != 'R',
              "Refusing byte launch: binary starts with the kernel container "
              "magic 'SPYR'. This boundary takes RAW init packets (starting "
              "with the QGInit magic byte 0xDD), not two-layer containers. "
              "Extract the payload with kernel_container.extract_payload() "
              "before launching; the runtime unwrapper that does this "
              "automatically is not implemented yet.");

  // Load artifacts — no global map lock held during device allocation/copy. The
  // artifact is owned by a shared_ptr so a cache hit can hold it alive across its
  // asynchronous enqueue, and so teardown's shared/exclusive gate governs its
  // lifetime coherently.
  auto arts = std::make_shared<KernelArtifacts>();
  arts->init_bin = binary;
  arts->program_size = binary.size();

  auto stream = getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1, -1));
  auto& allocator = SpyreAllocator::instance();
  flex::AllocationDirective directive(flex::PlacementPolicy::Bind, {0},
                                      std::nullopt, flex::MemoryType::Program);
  arts->device_alloc =
      std::move(allocator.allocate(arts->program_size, directive));
  auto* ctx = static_cast<SharedOwnerCtx*>(arts->device_alloc.get_context());
  TORCH_CHECK(arts->program_size <= ctx->composite_addr.total_size(),
              "Program size (", arts->program_size,
              ") exceeds allocated device memory (",
              ctx->composite_addr.total_size(), ")");
  // Copy exactly the program's own byte count at the copyProgramAsync boundary,
  // never the (possibly rounded-up) allocation total size. The source buffer
  // (arts->init_bin) stays alive here and across the synchronize() below.
  stream.copyProgramAsync(arts->init_bin.data(), &ctx->composite_addr,
                          arts->program_size);

  // Block until the asynchronous program copy has completed BEFORE the entry is
  // made visible, so no cache hit on another stream can ever select a program
  // whose device bytes are not yet written. If synchronize() throws, the launch
  // fails without inserting a not-yet-ready entry.
  stream.synchronize();

  // Publish the (now fully copied) entry. Only this thread publishes this key:
  // the insertion happens under the per-key lock after the double-check found the
  // key absent, and cache hits never insert.
  {
    std::unique_lock<std::shared_mutex> lock(g_bytes_cache_mtx);
    g_bytes_artifact_cache.emplace(key, arts);
  }

  // Guard before executeProgramAsync on the cache-miss path. The local shared_ptr
  // lease keeps the device allocation alive across the asynchronous enqueue.
  if (!exact_bytes_equal(arts->init_bin, binary)) {
    TORCH_CHECK(false,
                "Refusing byte launch: cached artifact bytes differ from "
                "requested binary before executeProgramAsync");
  }
  // Cache miss: the exact program bytes were just copied (and drained) at the
  // copyProgramAsync boundary this launch, so program_copied_this_launch is true
  // and the submitted-byte proof scope is the program-copy boundary.
  VerifiedByteIdentity verified =
      VerifiedByteIdentity::verify(binary, arts->init_bin, arts->init_bin);
  json observation = addCachedBinaryObservation(
      verified, target, generation, /*program_copied_this_launch=*/true);
  addBaseLaunchObservation(observation, binary, args, "bytes_cache_miss");
  stream.executeProgramAsync(*arts, args);
  recordLastLaunchObservation(observation);
  return observation;
}

std::string getLastLaunchObservation() {
  std::lock_guard<std::mutex> lock(g_last_launch_observation_mtx);
  return g_last_launch_observation.dump();
}

// Clear both artifact caches. The runtime lifecycle gate must already be held
// EXCLUSIVELY by the caller, and every device stream must already be drained, so
// no launch is in flight and no enqueued compute still references an allocation
// freed here. This never re-acquires the (non-recursive) lifecycle gate, so the
// public entry points can compose drain + clear (+ reset) under one exclusive
// hold without self-deadlock.
void clearArtifactCacheLocked() {
  {
    std::unique_lock<std::shared_mutex> lock(g_artifact_cache_mtx);
    g_artifact_cache.clear();
  }
  g_key_registry.clear();
  {
    std::unique_lock<std::shared_mutex> lock(g_bytes_cache_mtx);
    g_bytes_artifact_cache.clear();
  }
  g_bytes_key_registry.clear();
}

void clearArtifactCache() {
  // Exclusive teardown: block until every in-flight launch has returned and
  // released its shared lease, then drain the active device's streams so no
  // enqueued DMA/compute still references a program allocation, and only then free
  // the cached allocations. The drain covers the TLS-current device's streams
  // (single-active-device model), not every device in the process.
  std::unique_lock<std::shared_mutex> gate(g_runtime_lifecycle_gate);
  if (GlobalRuntime::get()) {
    synchronizeDevice(c10::optional<c10::Device>());
  }
  clearArtifactCacheLocked();
}

void shutdownRuntime() {
  // Hold the exclusive lifecycle gate across drain + cache clear + runtime-handle
  // release so that (a) no launch can enter between the clear and the release, and
  // (b) no allocation is freed and torch-spyre's runtime handle is not dropped while
  // any queued work is still in flight. GlobalRuntime::reset() only nulls
  // torch-spyre's shared_ptr to the flex::RuntimeContext; it does not reset the Flex
  // singleton. The gate is non-recursive, so this calls the *Locked helper rather
  // than the public clearArtifactCache() that would re-acquire it.
  std::unique_lock<std::shared_mutex> gate(g_runtime_lifecycle_gate);
  if (GlobalRuntime::get()) {
    synchronizeDevice(c10::optional<c10::Device>());
  }
  clearArtifactCacheLocked();
  GlobalRuntime::reset();
}

}  // namespace spyre
