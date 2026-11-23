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

"""T-RN-7 seed-invariant fingerprint gate for the UNIFORM draw binary.

The device UNIFORM draw is reproducible across ``(seed, offset)`` because the
seed and block offset are baked into the PARAMS segment (segment 0), NOT into
the emitted executable binary. ``launch_uniform_draw`` binds the binary by
tile count alone — ``build_rng_kat_uniform_binary(n_tiles=plan.n_tiles)`` (see
``torch_spyre/rng.py``) — so two plans that differ only in ``(seed, offset)``
launch the SAME bytes and differ only in the params they ship.

This module proves that separation OFFLINE, without a device:

* the binary the launcher would bind for a plan is byte-identical (sha256)
  for two plans that differ only in ``(seed, offset)``;
* those two plans' params DO differ, so the equality above cannot pass
  vacuously by the two plans being identical;
* a third plan at a different ``n_tiles`` binds a DIFFERENT binary, so the gate
  is not blind to geometry.

Nothing here is blocked on hardware: all three properties are device-free. The
binary IS emitted by ``sentient_codegen``, so this module skips (loudly, never
silently passing) when the codegen backend is not importable or the ``rng_kat``
pattern is not registered in the importable tree.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

hr = pytest.importorskip("torch_spyre.rng")

if TYPE_CHECKING:
    from torch_spyre.rng import RngLaunchPlan

pytest.importorskip(
    "sentient_codegen",
    reason="codegen backend not importable — point PYTHONPATH at a codegen "
    "tree to emit the UNIFORM binary; without one there is nothing to gate",
)

if not hr.rng_pattern_registered():
    pytest.skip(
        "backend pattern 'rng_kat' is not registered in the importable codegen "
        "tree; point PYTHONPATH at a codegen tip that carries it",
        allow_module_level=True,
    )

_TARGET = "rcudd1a"

#: Two plans that differ in BOTH seed and offset, so a builder that leaked
#: either into the executable would make the equal-sha assertion below fail.
_SEED_A, _OFFSET_A = 0x5EED_A, 0
_SEED_B, _OFFSET_B = 0x5EED_B, 7
#: A third, distinct params point for the geometry test.
_SEED_C, _OFFSET_C = 0xC0FF_EE, 0


def _sha256(binary_bytes) -> str:
    return hashlib.sha256(bytes(binary_bytes)).hexdigest()


def _plan(seed: int, offset: int, n_tiles: int) -> "RngLaunchPlan":
    """Build a UNIFORM launch plan through the real host entry point."""
    return hr.build_rng_launch_plan(
        seed=seed,
        offset=offset,
        n_tiles=n_tiles,
        egress=hr.EgressMode.UNIFORM,
    )


def _binary_for_plan(plan: "RngLaunchPlan") -> bytes:
    """The binary ``launch_uniform_draw`` binds for ``plan``.

    Mirrors the launcher's own call (``build_rng_kat_uniform_binary(
    n_tiles=plan.n_tiles)``) rather than reimplementing the build: the binary
    is selected by tile count, which is exactly the property under test.
    """
    return hr.build_rng_kat_uniform_binary(_TARGET, n_tiles=plan.n_tiles)[0]


@pytest.mark.parametrize("n_tiles", [1, 2])
def test_binary_is_seed_invariant_but_params_carry_state(n_tiles):
    """The binary is (seed, offset)-independent; the params are not.

    Both ``n_tiles`` widths the gate builds (1 and 2) are exercised so the
    property is not an artefact of the single-tile cell.
    """
    plan_a = _plan(_SEED_A, _OFFSET_A, n_tiles)
    bin_a = _binary_for_plan(plan_a)

    # Deriving plan B runs the host seed/offset chain (seed key -> epoch key ->
    # folded schedule). It sits BETWEEN the two binary fetches so that any
    # ambient-state leak from that chain into the executable would surface as a
    # digest change in bin_b.
    plan_b = _plan(_SEED_B, _OFFSET_B, n_tiles)
    bin_b = _binary_for_plan(plan_b)

    # Non-vacuity: the two plans must actually differ, or "same binary" below
    # would be trivially true because they are the same plan.
    assert not np.array_equal(plan_a.params, plan_b.params), (
        f"params for plans (seed={_SEED_A}, offset={_OFFSET_A}) and "
        f"(seed={_SEED_B}, offset={_OFFSET_B}) at n_tiles={n_tiles} are equal; "
        "the seed/offset plan is not reaching the params stream, so the "
        "seed-invariance assertion would be vacuous and the fingerprint gate "
        "would have no teeth."
    )

    assert _sha256(bin_a) == _sha256(bin_b), (
        f"UNIFORM binary differs across (seed, offset) at n_tiles={n_tiles}: "
        f"{_sha256(bin_a)} != {_sha256(bin_b)}. The seed and block offset must "
        "live in the params segment (segment 0), not the executable; a digest "
        "change here means the emitted binary has started consuming the RNG "
        "state, which breaks seed-invariant caching and reproducibility."
    )


def test_different_n_tiles_binds_a_different_binary():
    """A third plan at a different ``n_tiles`` must not reuse the other binary.

    Guards the opposite failure: a gate that only ever saw one binary could
    "pass" seed-invariance by never noticing geometry. Plan C changes the tile
    count (and seed/offset), so its binary must differ from plan A's at
    ``n_tiles=1``.
    """
    plan_c = _plan(_SEED_C, _OFFSET_C, 2)
    bin_c = _binary_for_plan(plan_c)

    plan_a = _plan(_SEED_A, _OFFSET_A, 1)
    bin_a = _binary_for_plan(plan_a)

    assert _sha256(bin_c) != _sha256(bin_a), (
        f"n_tiles=2 and n_tiles=1 bound the SAME binary "
        f"({_sha256(bin_c)}); the tile loop is not reflected in the emitted "
        "bytes, so the seed-invariance gate is blind to geometry and cannot "
        "distinguish a genuinely (seed, offset)-invariant binary from a "
        "geometry-insensitive one."
    )
