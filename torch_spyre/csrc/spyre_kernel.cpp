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
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      tensor.storage().data_ptr().get_context());
  json observation;
  observation["arg_index"] = index;
  observation["selected_segment_index"] = index;
  observation["storage_offset_bytes"] = tensor.storage_offset() * tensor.element_size();
  observation["device_bytes"] = tensor.nbytes();
  observation["torch_scalar_type"] = c10::toString(tensor.scalar_type());
  observation["device_dtype"] = c10::toString(tensor.scalar_type());
  if (tensor.is_privateuseone()) {
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
  }
  observation["alignment_bytes"] = kDeviceAlignmentBytes;
  observation["allocation_base_observed"] = false;
  observation["effective_pointer_alignment_proven"] = false;
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

// Cache: code_dir -> artifacts (loaded once)
std::unordered_map<std::string, KernelArtifacts> g_artifact_cache;
std::shared_mutex
    g_artifact_cache_mtx;  // protects g_artifact_cache and g_key_mtxs
std::unordered_map<std::string, std::unique_ptr<std::mutex>> g_key_mtxs;

KernelArtifacts& getOrLoadArtifacts(const std::string& code_dir,
                                    const SpyreStream& stream) {
  {
    std::unique_lock<std::shared_mutex> lock(g_artifact_cache_mtx);
    auto it = g_artifact_cache.find(code_dir);
    if (it != g_artifact_cache.end()) {
      return it->second;
    }
    auto& key_mtx = g_key_mtxs[code_dir];
    if (!key_mtx) {
      key_mtx = std::make_unique<std::mutex>();
    }
  }

  // Per-key lock: only one thread loads a given key
  std::mutex* key_mtx = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(g_artifact_cache_mtx);
    key_mtx = g_key_mtxs[code_dir].get();
  }
  std::lock_guard<std::mutex> key_lock(*key_mtx);

  // Double-check after acquiring per-key lock
  {
    std::shared_lock<std::shared_mutex> lock(g_artifact_cache_mtx);
    auto it = g_artifact_cache.find(code_dir);
    if (it != g_artifact_cache.end()) {
      return it->second;
    }
  }

  KernelArtifacts arts;

  fs::path dir(code_dir);
  std::string bundle_path = (dir / "bundle.mlir").string();
  // Store bundle.mlir path for future JIT compilation
  arts.bundle_mlir_path = bundle_path;
  TORCH_CHECK(std::filesystem::exists(bundle_path),
              "Bundle not found: ", bundle_path);

  // Read init packet binary — prefer raw binary (.bin) over hex (.txt)
  std::string init_dir = get_init_path(code_dir);
  std::string init_bin_path = init_dir + "/init.bin";
  std::string init_txt_path = init_dir + "/init.txt";
  if (std::filesystem::exists(init_bin_path)) {
    arts.init_bin = readBinaryFile(init_bin_path);
  } else {
    arts.init_bin = readHexEncodedFile(init_txt_path);
  }

  arts.program_size = arts.init_bin.size();
  auto& allocator = SpyreAllocator::instance();
  flex::AllocationDirective directive(flex::PlacementPolicy::Bind, {0},
                                      std::nullopt, flex::MemoryType::Program);
  arts.device_alloc =
      std::move(allocator.allocate(arts.program_size, directive));
  auto* ctx = static_cast<SharedOwnerCtx*>(arts.device_alloc.get_context());
  TORCH_CHECK(arts.program_size <= ctx->composite_addr.total_size(),
              "Program size (", arts.program_size,
              ") exceeds allocated device memory (",
              ctx->composite_addr.total_size(), ")");
  stream.copyProgramAsync(arts.init_bin.data(), &ctx->composite_addr);

  std::unique_lock<std::shared_mutex> lock(g_artifact_cache_mtx);
  auto [it, inserted] = g_artifact_cache.emplace(code_dir, std::move(arts));
  if (inserted) {
    DEBUGINFO(it->second);
  }
  return it->second;
}

void launchKernel(const std::string& code_dir,
                  const std::vector<at::Tensor>& args) {
  auto stream = getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1, -1));

  auto& arts = getOrLoadArtifacts(code_dir, stream);

  stream.executeProgramAsync(arts, args);
}

// Cache for launchKernelFromBytes: hash of binary -> loaded artifacts
// Avoids re-allocating and copying device memory on repeated launches
// of the same kernel (e.g., training loops).
static std::unordered_map<size_t, KernelArtifacts> g_bytes_artifact_cache;
static std::shared_mutex g_bytes_cache_mtx;
// Per-key mutexes to prevent concurrent loads of the same binary.
static std::unordered_map<size_t, std::unique_ptr<std::mutex>> g_bytes_key_mtxs;

void addCachedBinaryObservation(json& observation,
                                const std::vector<uint8_t>& launched_binary,
                                const std::vector<uint8_t>& cached_binary) {
  observation["cached_binary_size"] = cached_binary.size();
  observation["cached_binary_matches_launched_bytes"] =
      cached_binary == launched_binary;
  observation["cached_binary_match_method"] = "exact_byte_vector_equality";
  observation["cached_binary_fingerprint"] = fnv1a256(cached_binary);
  observation["cached_binary_fingerprint_algorithm"] =
      "fnv1a256_noncryptographic";
  observation["cached_binary_fingerprint_proof_scope"] =
      "diagnostic_only_not_sha256_identity_proof";
}

std::string launchKernelFromBytes(const std::vector<uint8_t>& binary,
                                  const std::vector<at::Tensor>& args) {
  json observation;
  observation["schema_version"] = "spyre-launch-observation-v1";
  observation["source"] = "torch_spyre.launch_kernel_from_bytes";
  observation["launched_binary_size"] = binary.size();
  observation["launched_binary_fingerprint"] = fnv1a256(binary);
  observation["launched_binary_fingerprint_algorithm"] = "fnv1a256_noncryptographic";
  observation["launched_binary_fingerprint_proof_scope"] =
      "diagnostic_only_not_sha256_identity_proof";
  observation["tensor_arg_count"] = args.size();
  observation["launch_observations"] = json::array();
  for (size_t i = 0; i < args.size(); ++i) {
    observation["launch_observations"].push_back(
        tensorLaunchObservation(args[i], i));
  }

  // Compute a hash of the binary for cache lookup.
  // Uses FNV-1a over the raw bytes. Collisions are possible but
  // extremely unlikely in practice (<1 MB init packets, small number
  // of distinct kernels per session). A collision would cause wrong-code
  // execution — if this becomes a concern, replace with SHA-256 truncated
  // to 64 bits.
  size_t key = 0xcbf29ce484222325ULL;  // FNV offset basis
  for (uint8_t b : binary) {
    key ^= static_cast<size_t>(b);
    key *= 0x100000001b3ULL;  // FNV prime
  }

  // Fast path: check cache under shared lock
  {
    std::shared_lock<std::shared_mutex> lock(g_bytes_cache_mtx);
    auto it = g_bytes_artifact_cache.find(key);
    if (it != g_bytes_artifact_cache.end()) {
      auto stream =
          getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1, -1));
      observation["bytes_cache_status"] = "bytes_cache_hit";
      addCachedBinaryObservation(observation, binary, it->second.init_bin);
      stream.executeProgramAsync(it->second, args);
      recordLastLaunchObservation(observation);
      return observation.dump();
    }
  }

  // Ensure a per-key mutex exists (under unique lock on the cache map)
  {
    std::unique_lock<std::shared_mutex> lock(g_bytes_cache_mtx);
    auto& key_mtx = g_bytes_key_mtxs[key];
    if (!key_mtx) {
      key_mtx = std::make_unique<std::mutex>();
    }
  }

  // Acquire per-key mutex — only one thread loads a given binary
  std::mutex* key_mtx = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(g_bytes_cache_mtx);
    key_mtx = g_bytes_key_mtxs[key].get();
  }
  std::lock_guard<std::mutex> key_lock(*key_mtx);

  // Double-check after acquiring per-key lock
  {
    std::shared_lock<std::shared_mutex> lock(g_bytes_cache_mtx);
    auto it = g_bytes_artifact_cache.find(key);
    if (it != g_bytes_artifact_cache.end()) {
      auto stream =
          getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1, -1));
      observation["bytes_cache_status"] = "bytes_cache_hit_after_wait";
      addCachedBinaryObservation(observation, binary, it->second.init_bin);
      stream.executeProgramAsync(it->second, args);
      recordLastLaunchObservation(observation);
      return observation.dump();
    }
  }

  // Sanity checks before device allocation — a malformed binary can crash
  // or hang the C++ runtime. Init packets must be 32-byte aligned (flit
  // granularity) and at least one flit (64 bytes = header + SPR flit).
  TORCH_CHECK(binary.size() % 32 == 0,
              "Init packet size must be 32-byte aligned, got ",
              binary.size(), " bytes");
  TORCH_CHECK(binary.size() >= 64,
              "Init packet too small: ", binary.size(),
              " bytes (minimum 64 = header + SPR flit)");

  // Load artifacts — no global lock held during device allocation/copy
  KernelArtifacts arts;
  arts.init_bin = binary;
  arts.program_size = binary.size();

  auto stream = getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1, -1));
  auto& allocator = SpyreAllocator::instance();
  flex::AllocationDirective directive(flex::PlacementPolicy::Bind, {0},
                                      std::nullopt, flex::MemoryType::Program);
  arts.device_alloc =
      std::move(allocator.allocate(arts.program_size, directive));
  auto* ctx = static_cast<SharedOwnerCtx*>(arts.device_alloc.get_context());
  TORCH_CHECK(arts.program_size <= ctx->composite_addr.total_size(),
              "Program size (", arts.program_size,
              ") exceeds allocated device memory (",
              ctx->composite_addr.total_size(), ")");
  stream.copyProgramAsync(arts.init_bin.data(), &ctx->composite_addr);

  // Cache under unique lock
  {
    std::unique_lock<std::shared_mutex> lock(g_bytes_cache_mtx);
    g_bytes_artifact_cache.emplace(key, std::move(arts));
  }

  // Execute outside all locks — same pattern as launchKernel
  {
    std::shared_lock<std::shared_mutex> lock(g_bytes_cache_mtx);
    auto it = g_bytes_artifact_cache.find(key);
    TORCH_CHECK(it != g_bytes_artifact_cache.end(),
                "Cached artifact not found after insertion");
    observation["bytes_cache_status"] = "bytes_cache_miss";
    addCachedBinaryObservation(observation, binary, it->second.init_bin);
    stream.executeProgramAsync(it->second, args);
  }
  recordLastLaunchObservation(observation);
  return observation.dump();
}

std::string getLastLaunchObservation() {
  std::lock_guard<std::mutex> lock(g_last_launch_observation_mtx);
  return g_last_launch_observation.dump();
}

void clearArtifactCache() {
  {
    std::unique_lock<std::shared_mutex> lock(g_artifact_cache_mtx);
    g_artifact_cache.clear();
    g_key_mtxs.clear();
  }
  {
    std::unique_lock<std::shared_mutex> lock(g_bytes_cache_mtx);
    g_bytes_artifact_cache.clear();
    g_bytes_key_mtxs.clear();
  }
}

}  // namespace spyre
