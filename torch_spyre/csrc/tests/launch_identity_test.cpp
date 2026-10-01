#include "launch_identity.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "sha256.h"

namespace {

void AssertNewV2FieldSet(const nlohmann::json& observation) {
  const std::vector<std::string> fields = {
      "runtime_provenance",
      "launch_target",
      "launch_generation",
      "launch_target_generation_status",
      "launched_binary_sha256",
      "launched_binary_sha256_algorithm",
      "launched_binary_sha256_proof_scope",
      "cached_binary_sha256",
      "cached_binary_sha256_algorithm",
      "cached_binary_sha256_proof_scope",
      "submitted_binary_size",
      "submitted_binary_sha256",
      "submitted_binary_sha256_algorithm",
      "submitted_binary_sha256_proof_scope",
      "submitted_binary_boundary",
      "cache_identity",
      "cache_identity_algorithm",
      "cache_identity_scope",
      "cache_identity_legacy_fnv1a64",
      "cache_identity_legacy_fnv1a64_algorithm",
      "cache_identity_legacy_fnv1a64_proof_scope",
      "pre_execute_identity_guard_status",
      "identity_checked_before_submit",
      "observation_order",
      "submitted_binary_matches_cached",
      "submitted_binary_matches_launched",
      "program_copied_this_launch",
      "cached_binary_size",
      "cached_binary_matches_launched_bytes",
      "cached_binary_match_method",
      "cached_binary_fingerprint",
      "cached_binary_fingerprint_algorithm",
      "cached_binary_fingerprint_proof_scope",
  };
  for (const std::string& field : fields) {
    assert(observation.contains(field));
  }
}

bool VerifyThrows(const std::vector<uint8_t>& requested,
                  const std::vector<uint8_t>& cached,
                  const std::vector<uint8_t>& submitted) {
  try {
    spyre::VerifiedByteIdentity::verify(requested, cached, submitted);
  }
  catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  const std::vector<uint8_t> launched = {0x10, 0x20, 0x30, 0x40};
  const std::vector<uint8_t> different = {0x10, 0x20, 0x30, 0x41};
  const std::string target = spyre::kDefaultTarget;
  const std::string generation = spyre::kDefaultGeneration;

  // Full-digest cache identity: a full 256-bit (64 hex) SHA-256, context-bound
  // to target/generation/binary, never a truncated 64-bit (16 hex) value.
  const std::string cache_identity =
      spyre::cache_identity_prefixed(launched, target, generation);
  assert(std::regex_match(cache_identity, std::regex("sha256:[0-9a-f]{64}")));
  assert(cache_identity.substr(std::string("sha256:").size()).size() == 64);
  assert(cache_identity != spyre::cache_identity_prefixed(
                               launched, "alternate-target", generation));
  assert(cache_identity != spyre::cache_identity_prefixed(
                               launched, target, "alternate-generation"));
  assert(cache_identity !=
         spyre::cache_identity_prefixed(different, target, generation));
  assert(spyre::exact_bytes_equal(launched, launched));
  assert(!spyre::exact_bytes_equal(launched, different));

  // Pinned cross-language golden: the canonical length-delimited (target,
  // generation, binary) encoding and its SHA-256 are a shared contract with the
  // Python runtime audit (_runtime_cache_identity_prefixed in
  // runtime_launch_abi_audit.py). Both sides assert this exact literal digest for
  // the reference input so any future drift between the two implementations is
  // caught; a test that only re-calls its own helper cannot catch that. Explicit
  // literals keep the vector independent of the kDefault* constants.
  const std::vector<uint8_t> abc = {'a', 'b', 'c'};
  assert(spyre::cache_identity_prefixed(abc, "rcudd1a", "AIU 1.0") ==
         "sha256:00ab0237ad3e83c52098f9601aa034d42101cf00e421af5af77e00ae4c82d647");
  // The length framing keeps the empty binary distinct from any non-empty one.
  assert(spyre::cache_identity_prefixed({}, "rcudd1a", "AIU 1.0") ==
         "sha256:e18f2f279b9f6be7d0876438e960e7fe5060c5e2b4ee3f07199bb4438511b727");

  // Verified byte identity is the only route to an "enforced" observation. It
  // refuses any mismatch and the empty binary.
  assert(VerifyThrows(launched, different, launched));
  assert(VerifyThrows(launched, launched, different));
  assert(VerifyThrows({}, {}, {}));

  const spyre::VerifiedByteIdentity verified =
      spyre::VerifiedByteIdentity::verify(launched, launched, launched);
  assert(verified.bytes() == launched);

  // program_copied_this_launch=true models the cache-miss path, where the exact
  // program bytes are freshly copied at the copyProgramAsync boundary.
  const nlohmann::json observation = spyre::build_launch_identity_observation(
      verified, target, generation, "fnv256:cached-diagnostic",
      /*program_copied_this_launch=*/true);
  AssertNewV2FieldSet(observation);
  assert(observation.at("runtime_provenance") == "in_process_launch_return");
  assert(observation.at("launch_target") == target);
  assert(observation.at("launch_generation") == generation);
  assert(observation.at("launch_target_generation_status") ==
         "validated_shipped_byte_launch_generation");
  assert(observation.at("launched_binary_sha256") ==
         spyre::sha256_prefixed(launched));
  assert(observation.at("launched_binary_sha256_algorithm") ==
         "sha256_fips180_4");
  assert(observation.at("launched_binary_sha256_proof_scope") ==
         "cryptographic_binary_identity");
  assert(observation.at("cached_binary_sha256") ==
         spyre::sha256_prefixed(launched));
  assert(observation.at("cached_binary_sha256_algorithm") ==
         "sha256_fips180_4");
  assert(observation.at("cached_binary_sha256_proof_scope") ==
         "cryptographic_binary_identity");
  assert(observation.at("submitted_binary_sha256") ==
         spyre::sha256_prefixed(launched));
  assert(observation.at("submitted_binary_size") == launched.size());
  assert(observation.at("submitted_binary_sha256_algorithm") ==
         "sha256_fips180_4");
  assert(observation.at("submitted_binary_sha256_proof_scope") ==
         "program_copy_boundary_cryptographic_identity");
  assert(observation.at("submitted_binary_boundary") == "copyProgramAsync");
  assert(observation.at("program_copied_this_launch") == true);
  assert(observation.at("cache_identity") == cache_identity);
  assert(observation.at("cache_identity_algorithm") ==
         "sha256_artifact_aware_cache_identity");
  assert(observation.at("cache_identity_scope") ==
         "sha256_binary_plus_target_generation_context");
  assert(observation.at("cache_identity_legacy_fnv1a64") ==
         spyre::legacy_fnv64_hex(launched));
  assert(observation.at("cache_identity_legacy_fnv1a64_algorithm") ==
         "fnv1a64_noncryptographic");
  assert(observation.at("cache_identity_legacy_fnv1a64_proof_scope") ==
         "diagnostic_only_not_cache_identity_proof");
  assert(observation.at("pre_execute_identity_guard_status") ==
         "enforced_cached_equals_requested_before_submit");
  assert(observation.at("identity_checked_before_submit") == true);
  assert(observation.at("observation_order") == "identity_checked_pre_submit");
  assert(observation.at("submitted_binary_matches_cached") == true);
  assert(observation.at("submitted_binary_matches_launched") == true);
  assert(observation.at("cached_binary_size") == launched.size());
  assert(observation.at("cached_binary_matches_launched_bytes") == true);
  assert(observation.at("cached_binary_match_method") ==
         "exact_byte_vector_equality");
  assert(observation.at("cached_binary_fingerprint") ==
         "fnv256:cached-diagnostic");
  assert(observation.at("cached_binary_fingerprint_algorithm") ==
         "fnv1a256_noncryptographic");
  assert(observation.at("cached_binary_fingerprint_proof_scope") ==
         "diagnostic_only_not_sha256_identity_proof");

  // The cache identity echoed in the observation is bound to the launch
  // context.
  assert(observation.at("cache_identity") ==
         spyre::cache_identity_prefixed(launched, target, generation));

  // program_copied_this_launch=false models the cache-hit path: no fresh
  // copyProgramAsync this launch, so the submitted bytes identify the selected
  // cached host artifact rather than a program-copy boundary. The honest label
  // must never claim copyProgramAsync for a launch that performed no copy.
  const nlohmann::json cache_hit_observation =
      spyre::build_launch_identity_observation(
          verified, target, generation, "fnv256:cached-diagnostic",
          /*program_copied_this_launch=*/false);
  AssertNewV2FieldSet(cache_hit_observation);
  assert(cache_hit_observation.at("program_copied_this_launch") == false);
  assert(cache_hit_observation.at("submitted_binary_boundary") ==
         "cached_device_program_selected");
  assert(cache_hit_observation.at("submitted_binary_sha256_proof_scope") ==
         "selected_cached_host_artifact_identity_not_recopied");
  // The submitted SHA still identifies the exact cached host bytes.
  assert(cache_hit_observation.at("submitted_binary_sha256") ==
         spyre::sha256_prefixed(launched));

  std::cout << "launch_identity_test: artifact-aware identity checks passed\n";
  return 0;
}
