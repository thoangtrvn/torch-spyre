#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "sha256.h"

namespace {

std::vector<uint8_t> Bytes(const std::string& text) {
  return std::vector<uint8_t>(text.begin(), text.end());
}

}  // namespace

int main() {
  assert(spyre::sha256_hex(Bytes("")) ==
         "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  assert(spyre::sha256_hex(Bytes("abc")) ==
         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  assert(spyre::sha256_hex(Bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  std::vector<uint8_t> multi_block(1000000, static_cast<uint8_t>('a'));
  assert(spyre::sha256_hex(multi_block) ==
         "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  assert(spyre::sha256_prefixed(Bytes("abc")) ==
         "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  std::cout << "sha256_test: all standard vectors passed\n";
  return 0;
}
