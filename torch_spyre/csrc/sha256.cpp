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

#include "sha256.h"

#include <array>
#include <iomanip>
#include <sstream>

namespace spyre {
namespace {

constexpr std::array<uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

uint32_t RotateRight(uint32_t value, uint32_t count) {
  return (value >> count) | (value << (32U - count));
}

void Transform(const uint8_t* block, std::array<uint32_t, 8>* state) {
  std::array<uint32_t, 64> words{};
  for (size_t i = 0; i < 16; ++i) {
    words[i] = (static_cast<uint32_t>(block[i * 4]) << 24U) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16U) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8U) |
               static_cast<uint32_t>(block[i * 4 + 3]);
  }
  for (size_t i = 16; i < words.size(); ++i) {
    const uint32_t s0 = RotateRight(words[i - 15], 7) ^
                        RotateRight(words[i - 15], 18) ^ (words[i - 15] >> 3U);
    const uint32_t s1 = RotateRight(words[i - 2], 17) ^
                        RotateRight(words[i - 2], 19) ^ (words[i - 2] >> 10U);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }

  uint32_t a = (*state)[0];
  uint32_t b = (*state)[1];
  uint32_t c = (*state)[2];
  uint32_t d = (*state)[3];
  uint32_t e = (*state)[4];
  uint32_t f = (*state)[5];
  uint32_t g = (*state)[6];
  uint32_t h = (*state)[7];
  for (size_t i = 0; i < words.size(); ++i) {
    const uint32_t s1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
    const uint32_t choice = (e & f) ^ ((~e) & g);
    const uint32_t temp1 = h + s1 + choice + kRoundConstants[i] + words[i];
    const uint32_t s0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = s0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  (*state)[0] += a;
  (*state)[1] += b;
  (*state)[2] += c;
  (*state)[3] += d;
  (*state)[4] += e;
  (*state)[5] += f;
  (*state)[6] += g;
  (*state)[7] += h;
}

}  // namespace

std::string sha256_hex(const std::vector<uint8_t>& bytes) {
  std::array<uint32_t, 8> state = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  };

  size_t offset = 0;
  while (offset + 64 <= bytes.size()) {
    Transform(bytes.data() + offset, &state);
    offset += 64;
  }

  std::array<uint8_t, 128> final_block{};
  const size_t remainder = bytes.size() - offset;
  for (size_t i = 0; i < remainder; ++i) {
    final_block[i] = bytes[offset + i];
  }
  final_block[remainder] = 0x80U;
  const size_t final_block_count = remainder >= 56 ? 2 : 1;
  const uint64_t bit_length = static_cast<uint64_t>(bytes.size()) * 8U;
  const size_t length_offset = final_block_count * 64 - 8;
  for (size_t i = 0; i < 8; ++i) {
    final_block[length_offset + i] =
        static_cast<uint8_t>(bit_length >> ((7 - i) * 8U));
  }
  Transform(final_block.data(), &state);
  if (final_block_count == 2) {
    Transform(final_block.data() + 64, &state);
  }

  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (uint32_t word : state) {
    output << std::setw(8) << word;
  }
  return output.str();
}

std::string sha256_prefixed(const std::vector<uint8_t>& bytes) {
  return "sha256:" + sha256_hex(bytes);
}

}  // namespace spyre
