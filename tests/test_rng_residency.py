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
"""Residency + partition manager (Option 1 shared construction).

All OFFLINE: no device, and NO dependence on the parallel codegen lane's
``patch_offset_words``. The patch path is exercised with a REAL local patch
function that recomputes only the three offset-dependent sticks
(COUNTER, X0_BASE_LO, X0_BASE_HI), and its output is checked BYTE-IDENTICAL to
a full ``build_band_lxlu_stream`` rebuild from codegen.

What is pinned here:

* invalidation contract — offset advance (same seed+epoch) patches once and
  does NOT rebuild; a seed change (``manual_seed``/``set_rng_state``) or an
  epoch rollover does a FULL rebuild, never a bare patch;
* ``ensure_params`` over a ``(seed, offset)`` sweep is byte-identical to a full
  codegen rebuild;
* the partition allocator hands out pairwise-disjoint, epoch-rounded counter
  ranges that never cross 2^30 and never contain a block index >= 2^23;
* determinism — the same ``(seed, offset)`` yields identical params and
  nothing advances on a repeated request.
"""

from __future__ import annotations

import numpy as np
import pytest

import torch_spyre.rng as hr

rm = pytest.importorskip(
    "sentient_codegen.emulator.threefry_recipe_model",
    reason="needs the codegen tree (prepend codegen to PYTHONPATH)",
)

_LANES = 32
_ROUNDS = 20
_EPOCH = 1 << 30
_EXACT = 1 << 23


def _spec(seed: int, offset: int) -> hr.RngDrawSpec:
    return hr.RngDrawSpec.from_seed_offset(seed, offset)


def _reference_params(seed: int, offset: int, *, lanes: int = _LANES) -> np.ndarray:
    """Full params array from a direct codegen ``build_band_lxlu_stream`` rebuild."""
    seed_lo, seed_hi = hr.seed_to_key(seed)
    epoch = offset >> 30
    base = offset % _EPOCH
    k_prime = hr.epoch_key_2x32((seed_lo, seed_hi), epoch, device=0, stream=0)
    x0_base, x1 = hr.fold_key_schedule_2x32(k_prime, base)
    injections = hr.injection_words_2x32(k_prime, _ROUNDS)
    rotations = [hr.ROTATIONS_2X32[r % 8] for r in range(_ROUNDS)]
    blocks = np.arange(lanes, dtype=np.uint32)
    ref = rm.build_band_lxlu_stream(
        blocks,
        x0_base=x0_base,
        x1=x1,
        rotations=rotations,
        injections=injections,
        rounds=_ROUNDS,
        egress=rm.EgressMode.RAW_STATE,
    )
    return np.stack([np.asarray(w.words, dtype=np.uint32) for w in ref], axis=0)


def _real_patch_fn(stream, *, blocks, x0_base):
    """Local real patch: recompute ONLY COUNTER + X0_BASE_{LO,HI}.

    Byte-identical to a full rebuild for the same ``(blocks, x0_base)`` and the
    original key-derived words. Mirrors the codegen owner's contract.
    """
    blocks = np.asarray(blocks)
    lanes = blocks.size

    def splat(value: float) -> np.ndarray:
        return np.full(
            lanes, rm.encode_rfp32_values(np.array([value]))[0], dtype=np.uint32
        )

    lo = float(x0_base & 0xFFFF)
    hi = float(x0_base >> 16)
    out = list(stream)
    out[4] = hr.StreamWord(
        stream[4].slot, rm.encode_rfp32_values(blocks.astype(np.float64))
    )
    out[5] = hr.StreamWord(stream[5].slot, splat(lo))
    out[6] = hr.StreamWord(stream[6].slot, splat(hi - 128))
    return tuple(out)


# ---------------------------------------------------------------------------
# Invalidation contract — recording fakes.
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.builds: list[tuple] = []
        self.patches: list[dict] = []

    def build(self, spec, device_index):
        self.builds.append((spec, device_index))
        return hr._default_build_stream(
            spec, device_index, rounds=_ROUNDS, egress=hr.EgressMode.RAW_STATE, stream=0
        )

    def patch(self, stream, *, blocks, x0_base):
        self.patches.append({"blocks": np.array(blocks), "x0_base": int(x0_base)})
        return _real_patch_fn(stream, blocks=blocks, x0_base=x0_base)


def test_offset_advance_patches_once_without_rebuild():
    rec = _Recorder()
    res = hr.SpyreRngResources(build_fn=rec.build, patch_fn=rec.patch)

    res.ensure_params(_spec(1234, 0), 0)
    assert len(rec.builds) == 1 and len(rec.patches) == 0

    res.ensure_params(_spec(1234, 64), 0)
    assert len(rec.builds) == 1, "an offset advance must NOT re-run the full build"
    assert len(rec.patches) == 1, "an offset advance must patch exactly once"
    # The patch got the correct (blocks, x0_base).
    k_prime = hr.epoch_key_2x32(hr.seed_to_key(1234), 0, device=0, stream=0)
    expected_x0_base, _ = hr.fold_key_schedule_2x32(k_prime, 64)
    assert rec.patches[0]["x0_base"] == expected_x0_base
    np.testing.assert_array_equal(
        rec.patches[0]["blocks"], np.arange(_LANES, dtype=np.uint32)
    )


def test_seed_change_forces_full_rebuild_not_patch():
    rec = _Recorder()
    res = hr.SpyreRngResources(build_fn=rec.build, patch_fn=rec.patch)

    res.ensure_params(_spec(11, 0), 0)
    res.ensure_params(_spec(11, 64), 0)  # offset advance -> patch
    assert len(rec.builds) == 1 and len(rec.patches) == 1

    # A new seed (what manual_seed / set_rng_state produces) must rebuild.
    res.ensure_params(_spec(99, 64), 0)
    assert len(rec.builds) == 2, "a seed change must do a FULL rebuild"
    assert len(rec.patches) == 1, "a seed change must NOT be served by a patch"


def test_epoch_rollover_forces_full_rebuild():
    rec = _Recorder()
    res = hr.SpyreRngResources(build_fn=rec.build, patch_fn=rec.patch)

    res.ensure_params(_spec(7, _EPOCH - 64), 0)  # epoch 0
    res.ensure_params(_spec(7, _EPOCH + 0), 0)  # epoch 1, same seed -> re-key
    assert len(rec.builds) == 2
    assert len(rec.patches) == 0


def test_set_rng_state_maps_to_a_seed_change_and_rebuilds():
    """``SpyreGeneratorState.set_state`` changing the seed drives a rebuild."""
    rec = _Recorder()
    res = hr.SpyreRngResources(build_fn=rec.build, patch_fn=rec.patch)
    st = hr.SpyreGeneratorState()
    st.manual_seed(2024)
    res.ensure_params(_spec(st.seed, st.offset), 0)

    # get/set round-trip into a DIFFERENT seed, as set_rng_state would apply.
    other = hr.SpyreGeneratorState()
    other.manual_seed(555)
    blob = other.to_tensor()
    st.set_state(blob)
    res.ensure_params(_spec(st.seed, st.offset), 0)
    assert len(rec.builds) == 2
    assert len(rec.patches) == 0


# ---------------------------------------------------------------------------
# Byte-identity vs a full codegen rebuild, across a (seed, offset) sweep.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 67280421310721, 2**40 + 7])
@pytest.mark.parametrize("offset", [0, 1, 31, 64, 4096, _EPOCH + 128])
def test_fresh_build_is_byte_identical_to_codegen_rebuild(seed, offset):
    res = hr.SpyreRngResources(patch_fn=_real_patch_fn)
    got = res.ensure_params(_spec(seed, offset), 0)
    want = _reference_params(seed, offset)
    assert got.shape == want.shape == (251, _LANES)
    np.testing.assert_array_equal(got, want)


def test_patched_advance_is_byte_identical_to_codegen_rebuild():
    """Sequential offset advances (the patch path) match a full rebuild exactly."""
    seed = 0xC0FFEE
    res = hr.SpyreRngResources(patch_fn=_real_patch_fn)
    res.ensure_params(_spec(seed, 0), 0)  # prime: full build
    for offset in (1, 32, 33, 64, 200, 4096):
        got = res.ensure_params(_spec(seed, offset), 0)  # patched
        np.testing.assert_array_equal(
            got, _reference_params(seed, offset), err_msg=f"offset={offset}"
        )


def test_without_a_patch_fn_offset_advance_falls_back_to_rebuild():
    """No codegen patch available -> a full rebuild, byte-identical (correct)."""
    rec = _Recorder()
    res = hr.SpyreRngResources(build_fn=rec.build, patch_fn=None)
    res.ensure_params(_spec(5, 0), 0)
    got = res.ensure_params(_spec(5, 64), 0)
    assert len(rec.builds) == 2  # fell back to a full rebuild
    np.testing.assert_array_equal(got, _reference_params(5, 64))


# ---------------------------------------------------------------------------
# Determinism — same (seed, offset) is replay-safe; nothing advances on repeat.
# ---------------------------------------------------------------------------


def test_same_spec_is_deterministic_and_does_not_advance():
    res = hr.SpyreRngResources(patch_fn=_real_patch_fn)
    a = res.ensure_params(_spec(42, 128), 0)
    b = res.ensure_params(_spec(42, 128), 0)  # repeat, no change
    np.testing.assert_array_equal(a, b)
    assert a is b, "a repeated request must return the cached resident buffer"


def test_separate_devices_never_alias():
    res = hr.SpyreRngResources(patch_fn=_real_patch_fn)
    a = res.ensure_params(_spec(9, 0), 0)
    b = res.ensure_params(_spec(9, 0), 1)  # different device index
    # Same seed/offset folds the same key for device 0 vs 1 only via the epoch
    # key's device bits, so the buffers must be independent objects.
    assert a is not b
    assert res.reflected(0) is not None and res.reflected(1) is not None


# ---------------------------------------------------------------------------
# Partition allocator — disjoint, epoch-rounded, bounded.
# ---------------------------------------------------------------------------


def test_allocator_ranges_are_pairwise_disjoint_and_bounded():
    res = hr.SpyreRngResources()
    ranges = []
    for n in (1, 31, 32, 64, 7, 128, 3):
        epoch, base = res.reserve(n)
        assert 0 <= base < _EXACT, "block indices must stay below 2^23"
        assert base + n <= _EPOCH, "a range must never cross the 2^30 epoch"
        ranges.append((epoch, base, base + n))
    # Same-epoch ranges never overlap (a single monotonic cursor guarantees it).
    by_epoch: dict[int, list[tuple[int, int]]] = {}
    for epoch, lo, hi in ranges:
        by_epoch.setdefault(epoch, []).append((lo, hi))
    for spans in by_epoch.values():
        spans.sort()
        for (a_lo, a_hi), (b_lo, b_hi) in zip(spans, spans[1:]):
            assert a_hi <= b_lo, "reservation ranges must be disjoint"


def test_allocator_rolls_epoch_at_the_exactness_bound():
    res = hr.SpyreRngResources()
    epoch0, base0 = res.reserve(_EXACT - 8)
    assert epoch0 == 0 and base0 == 0
    # The next range cannot fit below 2^23 -> a fresh epoch, base 0.
    epoch1, base1 = res.reserve(64)
    assert epoch1 == 1 and base1 == 0


def test_allocator_refuses_a_scan_above_the_exactness_bound():
    res = hr.SpyreRngResources()
    with pytest.raises(hr.SpyreRngError):
        res.reserve(_EXACT + 1)
