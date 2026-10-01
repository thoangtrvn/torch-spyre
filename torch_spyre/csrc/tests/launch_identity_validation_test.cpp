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

// Pure-logic unit test for validate_launch_identity_observation(). This exercises
// the native cross-check that an observation matches trusted expected identity.
// It is NOT a runtime/native-provenance test: it proves the validation logic, not
// that any genuine runtime produced the observation.

#include <cstdint>
#include <iostream>
#include <string>

#include "launch_identity.h"

namespace {

const char kDigest[] =
    "sha256:0000000000000000000000000000000000000000000000000000000000000001";
const char kCacheIdentity[] =
    "sha256:1111111111111111111111111111111111111111111111111111111111111111";
const char kTarget[] = "rcudd1a";
const char kGeneration[] = "AIU 1.0";

// A well-formed observation that must validate against the matching expected
// values below.
nlohmann::json WellFormedObservation() {
  nlohmann::json observation;
  observation["schema_version"] = "spyre-launch-observation-v2";
  observation["launch_target"] = kTarget;
  observation["launch_generation"] = kGeneration;
  observation["launched_binary_sha256"] = kDigest;
  observation["cached_binary_sha256"] = kDigest;
  observation["submitted_binary_sha256"] = kDigest;
  observation["cache_identity"] = kCacheIdentity;
  observation["pre_execute_identity_guard_status"] =
      "enforced_cached_equals_requested_before_submit";
  observation["identity_checked_before_submit"] = true;
  observation["submitted_binary_matches_cached"] = true;
  observation["cached_binary_matches_launched_bytes"] = true;
  observation["submitted_binary_size"] = static_cast<uint64_t>(65536);
  observation["cached_binary_size"] = static_cast<uint64_t>(65536);
  return observation;
}

spyre::LaunchIdentityVerdict Validate(const nlohmann::json& observation) {
  return spyre::validate_launch_identity_observation(
      observation, kDigest, kTarget, kGeneration, kCacheIdentity);
}

int g_failures = 0;

void ExpectVerified(const std::string& name,
                    const spyre::LaunchIdentityVerdict& verdict) {
  if (!verdict.origin_verified || !verdict.mismatches.empty()) {
    std::cerr << "FAIL (" << name << "): expected origin_verified with no "
              << "mismatches, got verified=" << verdict.origin_verified
              << " mismatches=" << verdict.mismatches.size() << "\n";
    for (const auto& m : verdict.mismatches) {
      std::cerr << "    - " << m << "\n";
    }
    ++g_failures;
  }
}

void ExpectRejected(const std::string& name,
                    const spyre::LaunchIdentityVerdict& verdict) {
  if (verdict.origin_verified || verdict.mismatches.empty()) {
    std::cerr << "FAIL (" << name << "): expected rejection with a specific "
              << "mismatch, got verified=" << verdict.origin_verified
              << " mismatches=" << verdict.mismatches.size() << "\n";
    ++g_failures;
  }
}

}  // namespace

int main() {
  // Positive: a well-formed observation with matching expected identity.
  ExpectVerified("well_formed", Validate(WellFormedObservation()));

  {  // Wrong target.
    auto o = WellFormedObservation();
    o["launch_target"] = "sen1p5";
    ExpectRejected("wrong_target", Validate(o));
  }
  {  // Wrong generation.
    auto o = WellFormedObservation();
    o["launch_generation"] = "AIU 1.5";
    ExpectRejected("wrong_generation", Validate(o));
  }
  {  // Wrong emitted SHA (all legs diverge from expected).
    auto o = WellFormedObservation();
    const char* other =
        "sha256:00000000000000000000000000000000000000000000000000000000000000"
        "02";
    o["launched_binary_sha256"] = other;
    o["cached_binary_sha256"] = other;
    o["submitted_binary_sha256"] = other;
    ExpectRejected("wrong_emitted_sha", Validate(o));
  }
  {  // Mismatched cache identity.
    auto o = WellFormedObservation();
    o["cache_identity"] =
        "sha256:22222222222222222222222222222222222222222222222222222222222222"
        "22";
    ExpectRejected("wrong_cache_identity", Validate(o));
  }
  {  // Divergent submitted SHA leg only.
    auto o = WellFormedObservation();
    o["submitted_binary_sha256"] =
        "sha256:33333333333333333333333333333333333333333333333333333333333333"
        "33";
    ExpectRejected("divergent_submitted_leg", Validate(o));
  }
  {  // Guard flag false.
    auto o = WellFormedObservation();
    o["identity_checked_before_submit"] = false;
    ExpectRejected("guard_flag_false", Validate(o));
  }
  {  // Missing required key.
    auto o = WellFormedObservation();
    o.erase("cache_identity");
    ExpectRejected("missing_key", Validate(o));
  }
  {  // Wrong schema version.
    auto o = WellFormedObservation();
    o["schema_version"] = "spyre-launch-observation-v1";
    ExpectRejected("wrong_schema_version", Validate(o));
  }
  {  // Size inconsistency.
    auto o = WellFormedObservation();
    o["cached_binary_size"] = static_cast<uint64_t>(32768);
    ExpectRejected("size_mismatch", Validate(o));
  }
  {  // Non-object observation fails closed.
    nlohmann::json not_object = "a string";
    ExpectRejected("non_object", Validate(not_object));
  }

  if (g_failures != 0) {
    std::cerr << "launch_identity_validation_test: " << g_failures
              << " failure(s)\n";
    return 1;
  }
  std::cout << "launch_identity_validation_test: all validation checks passed\n";
  return 0;
}
