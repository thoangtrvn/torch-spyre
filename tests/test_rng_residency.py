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

All OFFLINE: no device. The patch path is exercised with a REAL local patch
function that recomputes only the three offset-dependent sticks
(COUNTER, X0_BASE_LO, X0_BASE_HI), and its output is checked BYTE-IDENTICAL to
a full ``build_band_lxlu_stream`` rebuild from codegen. The DEVICE half is
exercised with a fake device bridge that COUNTS every host->device row, so the
data-movement claim (251 rows -> 3 rows) is measured, not asserted in prose.

What is pinned here:

* invalidation contract — offset advance (same seed+epoch) patches once and
  does NOT rebuild; a seed change (``manual_seed``/``set_rng_state``) or an
  epoch rollover does a FULL rebuild, never a bare patch;
* ``ensure_params`` over a ``(seed, offset)`` sweep is byte-identical to a full
  codegen rebuild;
* DEVICE residency — the full 251-stick (32 KB) params upload happens ONCE; a
  same-seed offset advance writes EXACTLY the 3 offset-dependent rows (384 B),
  and the resident device buffer is byte-identical to a fresh full build at the
  final ``(seed, offset)`` including across an epoch rollover;
* the default patch owner resolves to the codegen ``patch_offset_words`` and is
  the one used on an offset advance; the no-owner fallback is a full (correct)
  rebuild, visibly marked — never silent;
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
# Fake device bridge — COUNTS every host->device row (the accounting gate).
# ---------------------------------------------------------------------------

_WORDS_PER_STICK = _LANES  # 32 lanes per stick, one uint32 word each
_BYTES_PER_WORD = 4
_FULL_STICKS = 251  # RAW_STATE params stream size
_OFFSET_ROWS = (4, 5, 6)  # COUNTER, X0_BASE_LO, X0_BASE_HI


class _FakeDeviceTensor:
    """A device-tensor stand-in: a mirrored word buffer plus its shape."""

    def __init__(self, words: np.ndarray):
        self.words = np.array(words, dtype=np.uint32, copy=True)

    @property
    def shape(self) -> tuple:
        return tuple(self.words.shape)

    @property
    def device(self):
        return "fake-device"


class _FakeDeviceBridge:
    """Device bridge that models the transport and counts the bytes moved."""

    def __init__(self):
        self.upload_rows = 0
        self.written_rows = 0
        self.uploads: list[int] = []  # row count of each full upload
        self.writes: list[tuple] = []  # row indices of each partial write

    def upload(self, host_words, device):
        arr = np.array(host_words, dtype=np.uint32, copy=True)
        self.upload_rows += int(arr.shape[0])
        self.uploads.append(int(arr.shape[0]))
        return _FakeDeviceTensor(arr)

    def write_rows(self, resident, row_indices, host_rows):
        rows = tuple(int(r) for r in row_indices)
        resident.words[list(rows)] = np.asarray(host_rows, dtype=np.uint32)
        self.written_rows += len(rows)
        self.writes.append(rows)

    def to_host(self, resident):
        return np.array(resident.words, dtype=np.uint32, copy=True)

    @staticmethod
    def rows_to_bytes(rows: int) -> int:
        return int(rows) * _WORDS_PER_STICK * _BYTES_PER_WORD


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


# ---------------------------------------------------------------------------
# DEVICE residency — the headline: 251-row upload ONCE, 3-row writes after.
# ---------------------------------------------------------------------------


def test_offset_dependent_rows_come_from_codegen():
    """The rows written are the codegen owner's, not a hardcoded triple."""
    res = hr.SpyreRngResources()
    rows = res.offset_dependent_rows()
    assert rows == _OFFSET_ROWS, (
        "offset_dependent_indices must drive the written rows "
        f"(expected {_OFFSET_ROWS}, got {rows})"
    )
    # Same rows the codegen owner reports for the default (RAW_STATE) egress.
    assert rows == tuple(
        rm.offset_dependent_indices(rounds=_ROUNDS, egress=rm.EgressMode.RAW_STATE)
    )


def test_offset_advance_writes_only_three_rows_device_side():
    """THE data-movement gate: per offset advance, 3 rows (384 B) not 251."""
    bridge = _FakeDeviceBridge()
    res = hr.SpyreRngResources(device_bridge=bridge)
    seed = 0xC0FFEE

    res.ensure_device_params(_spec(seed, 0), 0)
    assert bridge.uploads == [_FULL_STICKS], "first draw uploads the full params"
    assert bridge.written_rows == 0
    assert res.last_device_update(0) == "upload"

    offsets = [1, 32, 33, 64, 200, 4096]
    for offset in offsets:
        res.ensure_device_params(_spec(seed, offset), 0)
        assert res.last_device_update(0) == "patch"

    # Every advance wrote EXACTLY the 3 offset-dependent rows.
    assert bridge.writes == [_OFFSET_ROWS] * len(offsets)
    assert bridge.written_rows == 3 * len(offsets)
    # The full params tensor was NEVER re-uploaded after the first build.
    assert bridge.uploads == [_FULL_STICKS]

    before_bytes = _FakeDeviceBridge.rows_to_bytes(_FULL_STICKS)
    after_bytes = _FakeDeviceBridge.rows_to_bytes(3)
    print(
        f"\n[upload accounting] before: {_FULL_STICKS} rows = {before_bytes} B "
        f"per draw -> after: 3 rows = {after_bytes} B per offset advance "
        f"({before_bytes // after_bytes}x fewer bytes)"
    )
    assert before_bytes == 32128 and after_bytes == 384
    # Per-advance host->device write is exactly 384 B, not 32 KB.
    for offset in offsets:
        res.ensure_device_params(_spec(seed, offset + 1), 0)
    assert bridge.written_rows == 3 * len(offsets) + 3 * len(offsets)


def test_resident_device_buffer_byte_identical_across_sweep_with_epoch_rollover():
    """The device mirror equals a fresh full build at every (seed, offset)."""
    bridge = _FakeDeviceBridge()
    res = hr.SpyreRngResources(device_bridge=bridge)
    seed = 0xABCDEF
    sweep = [
        0,
        1,
        32,
        64,
        1000,
        _EPOCH - 64,  # epoch 0 tail
        _EPOCH,  # epoch rollover -> FULL upload
        _EPOCH + 1,  # patch in the new epoch
        _EPOCH + 64,
    ]
    for offset in sweep:
        tensor = res.ensure_device_params(_spec(seed, offset), 0)
        got = np.asarray(bridge.to_host(tensor))
        want = _reference_params(seed, offset)
        assert got.shape == want.shape == (_FULL_STICKS, _LANES)
        np.testing.assert_array_equal(got, want, err_msg=f"offset={offset}")
        # The 3 offset-dependent rows equal the corresponding reference rows.
        rows = res.offset_dependent_rows()
        np.testing.assert_array_equal(got[list(rows)], want[list(rows)])
    # First build + epoch rollover => exactly two full uploads; every other
    # advance in the same epoch was a 3-row write.
    assert bridge.uploads == [_FULL_STICKS, _FULL_STICKS]
    assert bridge.writes == [_OFFSET_ROWS] * (len(sweep) - 2)


def test_seed_change_reuploads_the_device_buffer():
    bridge = _FakeDeviceBridge()
    res = hr.SpyreRngResources(device_bridge=bridge)
    res.ensure_device_params(_spec(11, 0), 0)  # first upload
    res.ensure_device_params(_spec(11, 64), 0)  # patch
    res.ensure_device_params(_spec(99, 64), 0)  # seed change -> full upload
    assert bridge.uploads == [_FULL_STICKS, _FULL_STICKS]
    assert res.last_device_update(0) == "upload"
    got = np.asarray(bridge.to_host(res.ensure_device_params(_spec(99, 64), 0)))
    np.testing.assert_array_equal(got, _reference_params(99, 64))


def test_repeated_device_spec_does_not_write():
    bridge = _FakeDeviceBridge()
    res = hr.SpyreRngResources(device_bridge=bridge)
    t1 = res.ensure_device_params(_spec(42, 128), 0)
    moved = bridge.upload_rows + bridge.written_rows
    t2 = res.ensure_device_params(_spec(42, 128), 0)
    assert t1 is t2
    assert res.last_device_update(0) == "cached"
    assert bridge.upload_rows + bridge.written_rows == moved, (
        "a repeated spec must not touch device memory"
    )


def test_device_residents_per_device_are_independent():
    bridge = _FakeDeviceBridge()
    res = hr.SpyreRngResources(device_bridge=bridge)
    a = res.ensure_device_params(_spec(9, 0), 0)
    b = res.ensure_device_params(_spec(9, 0), 1)
    assert a is not b
    assert bridge.uploads == [_FULL_STICKS, _FULL_STICKS]


# ---------------------------------------------------------------------------
# Real-patch integration — the codegen owner is the default and is used.
# ---------------------------------------------------------------------------


def test_default_patch_resolves_to_the_codegen_owner(monkeypatch):
    calls = []
    real = rm.patch_offset_words

    def spy(stream, *, blocks, x0_base):
        calls.append(1)
        return real(stream, blocks=blocks, x0_base=x0_base)

    monkeypatch.setattr(rm, "patch_offset_words", spy)

    res = hr.SpyreRngResources()
    res.ensure_params(_spec(1, 0), 0)
    res.ensure_params(_spec(1, 64), 0)  # offset advance
    assert len(calls) == 1, "the codegen patch owner must be used on an advance"
    assert res.patch_source(0) == "codegen"
    # And it is byte-identical to the codegen rebuild.
    np.testing.assert_array_equal(
        res.ensure_params(_spec(1, 64), 0), _reference_params(1, 64)
    )


def test_without_patch_owner_device_advance_uploads_and_is_marked(monkeypatch):
    """No codegen owner -> full (correct) upload, visibly marked, never silent."""
    monkeypatch.setattr(hr, "_load_codegen_patch_fn", lambda: None)
    monkeypatch.setattr(hr, "_load_codegen_offset_indices", lambda: None)

    bridge = _FakeDeviceBridge()
    res = hr.SpyreRngResources(device_bridge=bridge)
    res.ensure_device_params(_spec(5, 0), 0)
    tensor = res.ensure_device_params(_spec(5, 64), 0)

    assert res.patch_source(0) is None, "no codegen patch -> no patch-source claim"
    assert res.last_device_update(0) == "upload", "fallback must be a full upload"
    assert bridge.writes == [], "no partial write when the owner is absent"
    assert bridge.uploads == [_FULL_STICKS, _FULL_STICKS]
    np.testing.assert_array_equal(
        np.asarray(bridge.to_host(tensor)), _reference_params(5, 64)
    )
