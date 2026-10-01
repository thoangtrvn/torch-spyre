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

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace spyre {

inline constexpr char kDefaultTarget[] = "rcudd1a";
inline constexpr char kDefaultGeneration[] = "AIU 1.0";

// Canonical artifact-aware cache identity: the full SHA-256 over a
// length-delimited encoding of (target, generation, binary), rendered as
// "sha256:<64 lowercase hex>". Each field is framed as an 8-byte little-endian
// length followed by its bytes, in that fixed order; this byte encoding is the
// shared contract that the Python runtime audit recomputes independently to
// reject arbitrary syntactically valid digests, so the two implementations must
// stay identical. This full-digest string is used BOTH as the byte-cache map
// key and as the observation ``cache_identity`` value; it is never reduced to a
// truncated 64-bit derivative. The target/generation context prevents
// cross-artifact aliasing of otherwise-identical byte vectors across
// generations.
std::string cache_identity_prefixed(const std::vector<uint8_t>& binary,
                                    const std::string& target,
                                    const std::string& generation);

// Backward-compatible non-cryptographic FNV-1a 64-bit diagnostic (16 hex
// characters). Retained for diagnostics only; never a cache identity or proof.
std::string legacy_fnv64_hex(const std::vector<uint8_t>& binary);

bool exact_bytes_equal(const std::vector<uint8_t>& a,
                       const std::vector<uint8_t>& b);

// Capability object proving that the pre-execute cached==requested==submitted
// byte-identity guard actually ran and passed over one concrete byte vector.
// It can only be produced by ``verify``, which refuses (throws
// std::invalid_argument) on any mismatch or on an empty binary. A launch
// observation that claims guard enforcement therefore cannot be constructed for
// unequal inputs: the observation builder consumes this token rather than three
// independent vectors, so there is no code path that asserts enforcement over
// bytes that were not proven identical.
class VerifiedByteIdentity {
 public:
  static VerifiedByteIdentity verify(const std::vector<uint8_t>& requested,
                                     const std::vector<uint8_t>& cached,
                                     const std::vector<uint8_t>& submitted);

  const std::vector<uint8_t>& bytes() const {
    return bytes_;
  }

 private:
  explicit VerifiedByteIdentity(std::vector<uint8_t> bytes)
      : bytes_(std::move(bytes)) {}

  std::vector<uint8_t> bytes_;
};

// Build the versioned launch-identity observation from a proven byte-identity
// token. Because ``verified`` certifies requested==cached==submitted over a
// single byte vector, every launched/cached/submitted SHA-256 leg and every
// equality boolean is consistent by construction. ``target``/``generation`` are
// the explicit, caller-validated launch context echoed into the observation;
// ``cached_binary_fnv256`` is the retained non-cryptographic diagnostic.
//
// ``program_copied_this_launch`` records whether this launch performed the
// host->device program copy. It is true only on a cache miss, where the exact
// program bytes are freshly copied at the ``copyProgramAsync`` boundary; on a
// cache hit the device program was copied by an earlier miss and is merely
// selected for ``executeProgramAsync``. The submitted-byte boundary and proof
// scope are set honestly from this flag: a cache hit never claims a fresh
// program-copy boundary it did not perform.
nlohmann::json build_launch_identity_observation(
    const VerifiedByteIdentity& verified, const std::string& target,
    const std::string& generation, const std::string& cached_binary_fnv256,
    bool program_copied_this_launch);

// Verdict of validating a launch observation against trusted expected identity.
// ``origin_verified`` is true only when every check passes; otherwise
// ``mismatches`` names each specific failure. This is a plain value object; it
// carries no trust by itself. Trust comes from WHERE it is produced: only the
// native extension (which single-use-consumes a genuine, pybind-registered
// observation object before calling the validator) can hand Python a verdict,
// and Python cannot manufacture one.
struct LaunchIdentityVerdict {
  bool origin_verified = false;
  std::vector<std::string> mismatches;
};

// Validate that an observation emitted by build_launch_identity_observation /
// launchKernelFromBytes matches the trusted expected identity. This is pure
// logic over an already-consumed observation: it checks the observation's
// internal consistency (schema version, equal SHA-256 legs, enforced pre-execute
// guard flags, consistent sizes) AND equality to the trusted expected emitted
// SHA-256, target, generation, and artifact-aware cache identity. It never
// trusts a field it is not given an expected value for. Every missing or
// wrong-typed key is recorded as a mismatch rather than raising, so a malformed
// observation fails closed. ``origin_verified`` is true iff ``mismatches`` is
// empty. Native single-use consumption and the pybind registered-type gate are
// enforced by the caller (module.cpp), not here.
LaunchIdentityVerdict validate_launch_identity_observation(
    const nlohmann::json& observation,
    const std::string& expected_emitted_sha256,
    const std::string& expected_target, const std::string& expected_generation,
    const std::string& expected_cache_identity);

}  // namespace spyre
