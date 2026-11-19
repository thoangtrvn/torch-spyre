# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Target resolution for the device RNG build.

The device RNG kernel is 1p0-only: the witnessed binary is a rcudd1a emission.
``build_rng_kat_binary`` must resolve the target from the generation registry and
thread it through the compile (so the target and the stick geometry share one
generation), and MUST refuse a target with no witnessed kernel rather than emit a
wrong-generation binary. This pins both: the witnessed byte-match is preserved
for rcudd1a, and a non-1p0 target refuses typed.
"""

import hashlib

import pytest

pytest.importorskip("sentient_codegen")

from torch_spyre.rng import SpyreRngError, build_rng_kat_binary  # noqa: E402


def _rng_kat_registered() -> bool:
    """True only when the paired codegen carries the ``rng_kat`` pattern.

    ``build_rng_kat_binary`` emits through the codegen kernel, so it cannot
    produce the witnessed binary on a codegen tree that has not landed the
    registration (e.g. an older KB tip). Gate the module on that capability so a
    codegen pairing without ``rng_kat`` SKIPS with a named reason instead of
    reddening on a backend feature the framework legitimately depends on.
    """
    try:
        from sentient_codegen.scheduler.matmul import PATTERN_REGISTRY

        return "rng_kat" in PATTERN_REGISTRY
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _rng_kat_registered(),
    reason="needs a codegen tree with rng_kat registered (staging tip); "
    "build_rng_kat_binary cannot emit the witnessed binary without it",
)

_E697 = "e697b657d14269f219626f9e7418b29c223332e2e8eb03a61c3e091f4d263d9d"


def test_default_target_still_byte_matches_the_witnessed_binary():
    binary, _ = build_rng_kat_binary()
    assert hashlib.sha256(binary).hexdigest() == _E697


def test_explicit_rcudd1a_byte_matches_the_witnessed_binary():
    binary, _ = build_rng_kat_binary(target="rcudd1a")
    assert hashlib.sha256(binary).hexdigest() == _E697


def test_non_1p0_target_refuses_typed():
    """A target with no witnessed RNG kernel must refuse, not emit 1p0 bytes."""
    with pytest.raises(SpyreRngError):
        build_rng_kat_binary(target="sen1p5")
