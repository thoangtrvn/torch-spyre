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

#include "launch_identity.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "sha256.h"

namespace spyre {
namespace {

// Append an 8-byte little-endian length prefix followed by the field bytes.
// A length-delimited (rather than NUL-separated) framing keeps the canonical
// encoding injective even if a field could contain a NUL byte: the digest over
// (target, generation, binary) can only collide if the fields are identical.
void AppendLengthDelimited(std::vector<uint8_t>& context, const uint8_t* data,
                           size_t size) {
  uint64_t length = static_cast<uint64_t>(size);
  for (int shift = 0; shift < 64; shift += 8) {
    context.push_back(static_cast<uint8_t>((length >> shift) & 0xffU));
  }
  context.insert(context.end(), data, data + size);
}

// Canonical length-delimited cache-identity context. This byte encoding is the
// shared contract recomputed independently by the Python runtime audit; the two
// implementations must stay identical or the cross-check fails closed. Fields
// are framed in the fixed order target, generation, binary, each as an 8-byte
// little-endian length followed by its bytes.
std::vector<uint8_t> CacheIdentityContext(const std::vector<uint8_t>& binary,
                                          const std::string& target,
                                          const std::string& generation) {
  std::vector<uint8_t> context;
  context.reserve(target.size() + generation.size() + binary.size() + 24);
  AppendLengthDelimited(context,
                        reinterpret_cast<const uint8_t*>(target.data()),
                        target.size());
  AppendLengthDelimited(context,
                        reinterpret_cast<const uint8_t*>(generation.data()),
                        generation.size());
  AppendLengthDelimited(context, binary.data(), binary.size());
  return context;
}

}  // namespace

std::string cache_identity_prefixed(const std::vector<uint8_t>& binary,
                                    const std::string& target,
                                    const std::string& generation) {
  return sha256_prefixed(CacheIdentityContext(binary, target, generation));
}

std::string legacy_fnv64_hex(const std::vector<uint8_t>& binary) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (uint8_t byte : binary) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= 0x100000001b3ULL;
  }
  std::ostringstream output;
  output << std::hex << std::setw(16) << std::setfill('0') << hash;
  return output.str();
}

bool exact_bytes_equal(const std::vector<uint8_t>& a,
                       const std::vector<uint8_t>& b) {
  return a == b;
}

VerifiedByteIdentity VerifiedByteIdentity::verify(
    const std::vector<uint8_t>& requested, const std::vector<uint8_t>& cached,
    const std::vector<uint8_t>& submitted) {
  if (requested.empty()) {
    throw std::invalid_argument(
        "verified byte identity requires a non-empty binary");
  }
  if (!(requested == cached)) {
    throw std::invalid_argument(
        "verified byte identity refused: cached bytes differ from requested "
        "bytes before submission");
  }
  if (!(requested == submitted)) {
    throw std::invalid_argument(
        "verified byte identity refused: submitted bytes differ from requested "
        "bytes at the copy boundary");
  }
  return VerifiedByteIdentity(requested);
}

nlohmann::json build_launch_identity_observation(
    const VerifiedByteIdentity& verified, const std::string& target,
    const std::string& generation, const std::string& cached_binary_fnv256,
    bool program_copied_this_launch) {
  // ``verified`` certifies requested==cached==submitted over this one vector,
  // so every SHA-256 leg is this digest and every equality boolean is true.
  // There is no path to assert guard enforcement over bytes that were not
  // proven identical: the mismatch is refused in
  // VerifiedByteIdentity::verify().
  const std::vector<uint8_t>& bytes = verified.bytes();
  const std::string digest = sha256_prefixed(bytes);

  nlohmann::json observation;
  observation["runtime_provenance"] = "in_process_launch_return";
  observation["launch_target"] = target;
  observation["launch_generation"] = generation;
  observation["launch_target_generation_status"] =
      "validated_shipped_byte_launch_generation";
  observation["launched_binary_sha256"] = digest;
  observation["launched_binary_sha256_algorithm"] = "sha256_fips180_4";
  observation["launched_binary_sha256_proof_scope"] =
      "cryptographic_binary_identity";
  observation["cached_binary_sha256"] = digest;
  observation["cached_binary_sha256_algorithm"] = "sha256_fips180_4";
  observation["cached_binary_sha256_proof_scope"] =
      "cryptographic_binary_identity";
  observation["submitted_binary_size"] = bytes.size();
  observation["submitted_binary_sha256"] = digest;
  observation["submitted_binary_sha256_algorithm"] = "sha256_fips180_4";
  // The submitted-byte proof scope depends on whether this launch actually
  // performed the host->device program copy. Only a cache miss copies the exact
  // program bytes at the copyProgramAsync boundary this launch; a cache hit
  // selects a device program copied by an earlier miss and must not claim a
  // fresh program-copy boundary. ``program_copied_this_launch`` makes the
  // distinction explicit so the Python audit keys the expected labels on it.
  observation["program_copied_this_launch"] = program_copied_this_launch;
  if (program_copied_this_launch) {
    observation["submitted_binary_sha256_proof_scope"] =
        "program_copy_boundary_cryptographic_identity";
    observation["submitted_binary_boundary"] = "copyProgramAsync";
  } else {
    observation["submitted_binary_sha256_proof_scope"] =
        "selected_cached_host_artifact_identity_not_recopied";
    observation["submitted_binary_boundary"] = "cached_device_program_selected";
  }
  observation["cache_identity"] =
      cache_identity_prefixed(bytes, target, generation);
  observation["cache_identity_algorithm"] =
      "sha256_artifact_aware_cache_identity";
  observation["cache_identity_scope"] =
      "sha256_binary_plus_target_generation_context";
  observation["cache_identity_legacy_fnv1a64"] = legacy_fnv64_hex(bytes);
  observation["cache_identity_legacy_fnv1a64_algorithm"] =
      "fnv1a64_noncryptographic";
  observation["cache_identity_legacy_fnv1a64_proof_scope"] =
      "diagnostic_only_not_cache_identity_proof";
  observation["pre_execute_identity_guard_status"] =
      "enforced_cached_equals_requested_before_submit";
  observation["identity_checked_before_submit"] = true;
  observation["observation_order"] = "identity_checked_pre_submit";
  observation["submitted_binary_matches_cached"] = true;
  observation["submitted_binary_matches_launched"] = true;
  observation["cached_binary_size"] = bytes.size();
  observation["cached_binary_matches_launched_bytes"] = true;
  observation["cached_binary_match_method"] = "exact_byte_vector_equality";
  observation["cached_binary_fingerprint"] = cached_binary_fnv256;
  observation["cached_binary_fingerprint_algorithm"] =
      "fnv1a256_noncryptographic";
  observation["cached_binary_fingerprint_proof_scope"] =
      "diagnostic_only_not_sha256_identity_proof";
  return observation;
}

namespace {

// Read a required string field. Records a specific mismatch and returns false if
// the key is absent or not a string; otherwise copies it into ``out``.
bool ReadString(const nlohmann::json& observation, const char* key,
                std::string& out, std::vector<std::string>& mismatches) {
  auto it = observation.find(key);
  if (it == observation.end()) {
    mismatches.emplace_back(std::string("missing field: ") + key);
    return false;
  }
  if (!it->is_string()) {
    mismatches.emplace_back(std::string("non-string field: ") + key);
    return false;
  }
  out = it->get<std::string>();
  return true;
}

// Require a string field to equal ``expected``.
void RequireStringEquals(const nlohmann::json& observation, const char* key,
                         const std::string& expected,
                         std::vector<std::string>& mismatches) {
  std::string value;
  if (!ReadString(observation, key, value, mismatches)) {
    return;
  }
  if (value != expected) {
    mismatches.emplace_back(std::string(key) + " mismatch: expected '" +
                            expected + "' got '" + value + "'");
  }
}

// Require a boolean field to be present and true.
void RequireTrue(const nlohmann::json& observation, const char* key,
                 std::vector<std::string>& mismatches) {
  auto it = observation.find(key);
  if (it == observation.end()) {
    mismatches.emplace_back(std::string("missing field: ") + key);
    return;
  }
  if (!it->is_boolean()) {
    mismatches.emplace_back(std::string("non-boolean field: ") + key);
    return;
  }
  if (!it->get<bool>()) {
    mismatches.emplace_back(std::string(key) + " is false, expected true");
  }
}

// Read a required unsigned-integer field.
bool ReadUint(const nlohmann::json& observation, const char* key, uint64_t& out,
              std::vector<std::string>& mismatches) {
  auto it = observation.find(key);
  if (it == observation.end()) {
    mismatches.emplace_back(std::string("missing field: ") + key);
    return false;
  }
  if (!it->is_number_unsigned()) {
    mismatches.emplace_back(std::string("non-unsigned-integer field: ") + key);
    return false;
  }
  out = it->get<uint64_t>();
  return true;
}

}  // namespace

LaunchIdentityVerdict validate_launch_identity_observation(
    const nlohmann::json& observation,
    const std::string& expected_emitted_sha256,
    const std::string& expected_target, const std::string& expected_generation,
    const std::string& expected_cache_identity) {
  LaunchIdentityVerdict verdict;
  std::vector<std::string>& mismatches = verdict.mismatches;

  if (!observation.is_object()) {
    mismatches.emplace_back("observation is not a JSON object");
    verdict.origin_verified = false;
    return verdict;
  }

  // Schema version this validator understands.
  RequireStringEquals(observation, "schema_version",
                      "spyre-launch-observation-v2", mismatches);

  // Trusted launch context.
  RequireStringEquals(observation, "launch_target", expected_target,
                      mismatches);
  RequireStringEquals(observation, "launch_generation", expected_generation,
                      mismatches);

  // Every SHA-256 leg must equal the trusted expected emitted digest. Checking
  // each leg against the expected value also guarantees the legs equal one
  // another, so a divergent submitted/cached leg is caught.
  RequireStringEquals(observation, "launched_binary_sha256",
                      expected_emitted_sha256, mismatches);
  RequireStringEquals(observation, "cached_binary_sha256",
                      expected_emitted_sha256, mismatches);
  RequireStringEquals(observation, "submitted_binary_sha256",
                      expected_emitted_sha256, mismatches);

  // Artifact-aware cache identity.
  RequireStringEquals(observation, "cache_identity", expected_cache_identity,
                      mismatches);

  // Pre-execute guard and consistency flags.
  RequireStringEquals(observation, "pre_execute_identity_guard_status",
                      "enforced_cached_equals_requested_before_submit",
                      mismatches);
  RequireTrue(observation, "identity_checked_before_submit", mismatches);
  RequireTrue(observation, "submitted_binary_matches_cached", mismatches);
  RequireTrue(observation, "cached_binary_matches_launched_bytes", mismatches);

  // Sizes must be present and consistent.
  uint64_t submitted_size = 0;
  uint64_t cached_size = 0;
  bool have_submitted = ReadUint(observation, "submitted_binary_size",
                                 submitted_size, mismatches);
  bool have_cached =
      ReadUint(observation, "cached_binary_size", cached_size, mismatches);
  if (have_submitted && have_cached && submitted_size != cached_size) {
    mismatches.emplace_back(
        "submitted_binary_size and cached_binary_size differ");
  }

  verdict.origin_verified = mismatches.empty();
  return verdict;
}

}  // namespace spyre
