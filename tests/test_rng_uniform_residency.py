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
"""A9: UNIFORM multitile params residency + incremental offset patch.

All OFFLINE: no device. The device half runs on a fake bridge that COUNTS
every uploaded row and every patched row, so the data-movement claim (one
full ``(257·n, 32)`` upload, then ``2n`` rows per draw) is measured, not
asserted in prose. Bit-exactness is checked against
``draw_params_stream``-equivalent full rebuilds across an offset sweep that
INCLUDES the per-tile ``0x10000`` carry and its neighbours.

Derivation of record: ``codegen/docs/rng-a9-uniform-params-residency.md``
(lane worktree). What is pinned here:

* EQUIVALENCE — the resident device buffer is byte-identical to a fresh
  full multitile build at every swept offset, including across the carry;
* the per-draw device write is EXACTLY ``2n`` rows (every tile's
  ``X0_BASE_LO`` + ``X0_BASE_HI``; the non-carrying HI rows are rewritten
  with identical bytes);
* the invalidation contract — same seed+epoch advance patches; seed change,
  epoch rollover and width change rebuild; a repeated spec writes nothing;
  a cold cache uploads once; a plan's already-built params SEED the resident
  (no double host build) and the NEXT draw still patches;
* the no-owner fallback is a full rebuild, visibly marked;
* a warm UNIFORM plan of the right width skips the params build; a cold or
  wrong-width plan builds in full (the pre-A9 bytes).
"""

from __future__ import annotations

import numpy as np
import pytest

import torch_spyre.rng as hr

rm = pytest.importorskip(
    "sentient_codegen.emulator.threefry_recipe_model",
    reason="needs the codegen tree (prepend codegen to PYTHONPATH)",
)

_HAS_CODEGEN_PATCH = hasattr(rm, "patch_offset_words") and hasattr(
    rm, "offset_dependent_indices"
)
_needs_codegen_patch = pytest.mark.skipif(
    not _HAS_CODEGEN_PATCH,
    reason="needs the codegen offset-patch owner "
    "(patch_offset_words/offset_dependent_indices)",
)

_LANES = 32
_ROUNDS = 20
_EPOCH = 1 << 30
_STRIDE = 257  # band_lxlu_pop_count(20, UNIFORM)


def _spec(seed: int, offset: int) -> hr.RngDrawSpec:
    return hr.RngDrawSpec.from_seed_offset(seed, offset)


def _full(seed: int, offset: int, n_tiles: int) -> np.ndarray:
    """Full multitile params array — the pre-A9 per-draw build (the oracle)."""
    return hr.stream_to_sticks(
        hr.draw_params_stream(
            seed=seed, offset=offset, egress=hr.EgressMode.UNIFORM, n_tiles=n_tiles
        )
    )


def _carry_offset(seed: int, n_tiles: int) -> int:
    """An offset whose NEXT advance carries tile ``n_tiles - 1``'s 16-bit half.

    ``x0_base = (base_counter + ks0) mod 2^32``; tile ``t`` carries when
    ``(ks0 & 0xFFFF) + t + delta >= 0x10000``. The returned offset is 0 so the
    sweep's delta from 0 to this value crosses the carry exactly at the last
    step (mirrors the A9 derivation probe).
    """
    ks0 = hr.key_schedule_2x32(
        hr.epoch_key_2x32(hr.seed_to_key(seed), 0, device=0, stream=0)
    )[0]
    return 0x10000 - (ks0 & 0xFFFF) - (n_tiles - 1)


class _FakeDeviceTensor:
    def __init__(self, words: np.ndarray):
        self.words = np.array(words, dtype=np.uint32, copy=True)

    @property
    def shape(self) -> tuple:
        return tuple(self.words.shape)


class _FakeDeviceBridge:
    """Counts every host->device row (the accounting gate)."""

    def __init__(self):
        self.uploads: list[int] = []
        self.writes: list[tuple] = []

    def upload(self, host_words, device):
        arr = np.array(host_words, dtype=np.uint32, copy=True)
        self.uploads.append(int(arr.shape[0]))
        return _FakeDeviceTensor(arr)

    def write_rows(self, resident, row_indices, host_rows):
        rows = tuple(int(r) for r in row_indices)
        resident.words[list(rows)] = np.asarray(host_rows, dtype=np.uint32)
        self.writes.append(rows)

    def to_host(self, resident):
        return np.array(resident.words, dtype=np.uint32, copy=True)


def _resources():
    return hr.SpyreRngResources(device_bridge=_FakeDeviceBridge())


# ---------------------------------------------------------------------------
# EQUIVALENCE — resident bytes == full-build bytes, across the carry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_tiles", [1, 4])
@pytest.mark.parametrize("seed", [12345, 0xC0FFEE])
def test_resident_uniform_bytes_match_full_build_across_carry(seed, n_tiles):
    """The resident tensor equals a fresh full build at every swept offset.

    The sweep includes the offset whose advance from the previous entry
    carries the LAST tile's 16-bit half (and its neighbours either side), so
    the per-tile ``X0_BASE_HI`` carry path is exercised, not just LO.
    """
    res = _resources()
    bridge: _FakeDeviceBridge = res._device_bridge
    carry = _carry_offset(seed, n_tiles)
    sweep = [0, 64, 256, carry - 1, carry, carry + 1, 4032, _EPOCH, _EPOCH + 64]
    for off in sweep:
        tensor = res.ensure_uniform_device_params(_spec(seed, off), n_tiles, 0)
        np.testing.assert_array_equal(
            bridge.to_host(tensor),
            _full(seed, off, n_tiles),
            err_msg=f"seed={seed} n_tiles={n_tiles} offset={off}",
        )
    # Exactly one full upload: every later draw (same seed) patched, and the
    # epoch rollover rebuilt via a fresh upload — so 2 uploads total.
    assert bridge.uploads == [_STRIDE * n_tiles] * 2, bridge.uploads
    # Every patch wrote exactly 2n rows.
    for rows in bridge.writes:
        assert len(rows) == 2 * n_tiles, rows
    # The patched rows are exactly the per-tile X0_BASE_{LO,HI} positions.
    for rows in bridge.writes:
        expected = tuple(
            t * _STRIDE + local for t in range(n_tiles) for local in (5, 6)
        )
        assert rows == expected


@_needs_codegen_patch
def test_vectorized_patch_rows_match_patch_offset_words_per_tile():
    """The vectorized fast patch (encode_rfp32_values owner) must equal the
    per-tile ``patch_offset_words`` owner output for the LO/HI rows — the
    guard that keeps the fast path a mirror, not a reimplementation."""
    res = _resources()
    seed = 4242
    ks0 = hr.key_schedule_2x32(
        hr.epoch_key_2x32(hr.seed_to_key(seed), 0, device=0, stream=0)
    )[0]
    carry = 0x10000 - (ks0 & 0xFFFF) - 3
    for offset in (0, 64, carry, carry + 1):
        spec = _spec(seed, offset)
        res2 = hr.SpyreRngResources()
        got = res2._advance_uniform(0, 4, spec).params
        # Per-tile owner: patch a fresh single-tile stream at x0_base + t.
        k_prime = hr.epoch_key_2x32(hr.seed_to_key(seed), spec.epoch, device=0, stream=0)
        x0_base, _ = hr.fold_key_schedule_2x32(k_prime, spec.base_counter)
        blocks = np.arange(_LANES, dtype=np.uint32)
        for t in range(4):
            tile = rm.build_band_lxlu_stream(
                blocks,
                x0_base=0,
                x1=0,
                rotations=[16] * _ROUNDS,
                injections=[(0, 0)] * (_ROUNDS // 4),
                rounds=_ROUNDS,
                egress=rm.EgressMode.UNIFORM,
            )
            patched = rm.patch_offset_words(
                tile, blocks=blocks, x0_base=x0_base + t
            )
            np.testing.assert_array_equal(
                got[t * _STRIDE + 5], patched[5].words,
                err_msg=f"offset={offset} tile={t} LO",
            )
            np.testing.assert_array_equal(
                got[t * _STRIDE + 6], patched[6].words,
                err_msg=f"offset={offset} tile={t} HI",
            )


@_needs_codegen_patch
def test_carry_changes_only_the_carrying_tiles_hi_row():
    """At the carry, the device write still covers all 2n rows, but only the
    carrying tile's HI row VALUE differs from the pre-carry build."""
    seed, n_tiles = 12345, 4
    carry = _carry_offset(seed, n_tiles)
    before = _full(seed, carry - 1, n_tiles)
    after = _full(seed, carry, n_tiles)
    changed = [
        r for r in range(before.shape[0]) if not np.array_equal(before[r], after[r])
    ]
    # Every tile's LO row plus ONLY the last tile's HI row (A9 derivation).
    assert changed == [t * _STRIDE + 5 for t in range(n_tiles)] + [
        (n_tiles - 1) * _STRIDE + 6
    ], changed


# ---------------------------------------------------------------------------
# Invalidation contract — positive and negative per rule.
# ---------------------------------------------------------------------------


@_needs_codegen_patch
def test_same_seed_offset_advance_patches_without_rebuild():
    res = _resources()
    bridge: _FakeDeviceBridge = res._device_bridge
    res.ensure_uniform_device_params(_spec(7, 0), 4, 0)
    res.ensure_uniform_device_params(_spec(7, 256), 4, 0)
    assert bridge.uploads == [_STRIDE * 4]
    assert len(bridge.writes) == 1 and len(bridge.writes[0]) == 8
    assert res.uniform_last_device_update(0, 4) == "patch"
    assert res.uniform_patch_source(0, 4) == "codegen"


def test_seed_change_forces_full_rebuild_not_patch():
    res = _resources()
    bridge: _FakeDeviceBridge = res._device_bridge
    res.ensure_uniform_device_params(_spec(7, 0), 4, 0)
    res.ensure_uniform_device_params(_spec(8, 0), 4, 0)
    assert bridge.uploads == [_STRIDE * 4, _STRIDE * 4]
    assert bridge.writes == []


def test_epoch_rollover_forces_full_rebuild():
    res = _resources()
    bridge: _FakeDeviceBridge = res._device_bridge
    res.ensure_uniform_device_params(_spec(7, 0), 4, 0)
    res.ensure_uniform_device_params(_spec(7, _EPOCH), 4, 0)
    assert bridge.uploads == [_STRIDE * 4, _STRIDE * 4]
    assert bridge.writes == []


def test_width_change_is_a_separate_resident():
    res = _resources()
    bridge: _FakeDeviceBridge = res._device_bridge
    res.ensure_uniform_device_params(_spec(7, 0), 4, 0)
    res.ensure_uniform_device_params(_spec(7, 0), 2, 0)
    assert bridge.uploads == [_STRIDE * 4, _STRIDE * 2]
    assert bridge.writes == []
    # The 4-tile resident is untouched by the 2-tile draw.
    assert res.has_uniform_device_resident(0, 4)
    assert res.has_uniform_device_resident(0, 2)
    assert not res.has_uniform_device_resident(0, 8)


def test_repeated_spec_writes_nothing():
    res = _resources()
    bridge: _FakeDeviceBridge = res._device_bridge
    res.ensure_uniform_device_params(_spec(7, 0), 4, 0)
    n_uploads, n_writes = len(bridge.uploads), len(bridge.writes)
    res.ensure_uniform_device_params(_spec(7, 0), 4, 0)
    assert len(bridge.uploads) == n_uploads
    assert len(bridge.writes) == n_writes
    assert res.uniform_last_device_update(0, 4) == "cached"


def test_cold_plan_params_seed_the_resident_and_the_next_draw_patches():
    """A cold plan's already-built words upload AS the resident (no second
    host build), and the FOLLOWING draw patches — the stream mirror is
    reconstructed from the owner's slot order, not rebuilt."""
    res = _resources()
    bridge: _FakeDeviceBridge = res._device_bridge
    host_params = _full(99, 0, 4)
    res.ensure_uniform_device_params(_spec(99, 0), 4, 0, host_params=host_params)
    assert bridge.uploads == [_STRIDE * 4]
    tensor = res.ensure_uniform_device_params(_spec(99, 256), 4, 0)
    assert bridge.uploads == [_STRIDE * 4], "the advance must PATCH, not rebuild"
    assert len(bridge.writes) == 1 and len(bridge.writes[0]) == 8
    np.testing.assert_array_equal(bridge.to_host(tensor), _full(99, 256, 4))


def test_host_params_of_the_wrong_shape_refuses_typed():
    res = _resources()
    with pytest.raises(hr.SpyreRngError):
        res.ensure_uniform_device_params(
            _spec(99, 0), 2, 0, host_params=_full(99, 0, 4)
        )


def test_no_patch_owner_falls_back_visibly_and_stays_correct(monkeypatch):
    """With the encoding owner unavailable, the HOST mirror rebuilds fully
    (unoptimized, marked ``patch_source=None``) and the device stays
    byte-correct. The vectorized uniform patch keys on
    ``encode_rfp32_values``, not on the raw-egress ``patch_fn`` — killing the
    encoder is what disables it. Never a cheaper wrong answer."""
    monkeypatch.setattr(hr, "_load_codegen_encoder", lambda: None)
    res = hr.SpyreRngResources(device_bridge=_FakeDeviceBridge())
    bridge: _FakeDeviceBridge = res._device_bridge
    res.ensure_uniform_device_params(_spec(7, 0), 4, 0)
    tensor = res.ensure_uniform_device_params(_spec(7, 256), 4, 0)
    assert res.uniform_patch_source(0, 4) is None
    np.testing.assert_array_equal(bridge.to_host(tensor), _full(7, 256, 4))


def test_no_offset_index_owner_uploads_in_full():
    res = hr.SpyreRngResources(device_bridge=_FakeDeviceBridge())
    bridge: _FakeDeviceBridge = res._device_bridge
    res.ensure_uniform_device_params(_spec(7, 0), 4, 0)
    seen = len(bridge.uploads)
    # Kill ONLY the offset-index owner (the row positions): the device path
    # must not guess a row set — it re-uploads.
    res._offset_rows_cache = None
    res._offset_rows_resolved = True
    tensor = res.ensure_uniform_device_params(_spec(7, 256), 4, 0)
    assert len(bridge.uploads) == seen + 1
    np.testing.assert_array_equal(bridge.to_host(tensor), _full(7, 256, 4))


def test_transport_refusal_falls_back_to_full_upload():
    """A device that refuses the row scatter (WITNESSED on silicon
    2026-11-23: ``aten::index_copy_`` has no spyre kernel) gets the patched
    host mirror as a full-tensor upload — byte-correct, and still skipping
    the per-draw host rebuild."""

    class _RefusingBridge(_FakeDeviceBridge):
        def write_rows(self, resident, row_indices, host_rows):
            raise NotImplementedError("aten::index_copy.out: no spyre kernel")

    res = hr.SpyreRngResources(device_bridge=_RefusingBridge())
    bridge: _RefusingBridge = res._device_bridge
    res.ensure_uniform_device_params(_spec(7, 0), 4, 0)
    tensor = res.ensure_uniform_device_params(_spec(7, 256), 4, 0)
    assert bridge.uploads == [_STRIDE * 4, _STRIDE * 4]
    assert bridge.writes == []
    assert res.uniform_last_device_update(0, 4) == "upload"
    np.testing.assert_array_equal(bridge.to_host(tensor), _full(7, 256, 4))


@_needs_codegen_patch
def test_uniform_patch_set_excludes_counter():
    """The offset-only patch set is the LO/HI pair, NOT the codegen owner's
    full ``(blocks, x0_base)`` sensitivity set. ``offset_dependent_indices``
    returns (COUNTER, X0_BASE_LO, X0_BASE_HI); COUNTER is sensitive to
    ``blocks`` (fixed 0..31 per tile — the per-tile counter advance is baked
    into ``x0_base + t``, not the COUNTER word), so feeding the full set into
    the patch would rewrite every COUNTER row on every draw: correct bytes
    but a third wider write, and it would mask the derivation this lane
    rests on. This test FAILS if the patch set is ever replaced with the
    owner's full three-slot result."""
    res = _resources()
    owner_indices = rm.offset_dependent_indices(
        rounds=_ROUNDS, egress=rm.EgressMode.UNIFORM
    )
    assert owner_indices == (4, 5, 6), owner_indices
    patch_rows = res._uniform_patch_rows()
    # The offset-only set: the two X0_BASE words, COUNTER (position 0)
    # explicitly dropped.
    assert patch_rows == (5, 6), patch_rows
    assert set(patch_rows) < set(owner_indices), (
        "the offset patch set must be a PROPER subset of the (blocks, x0_base) "
        "sensitivity set — equal sets mean COUNTER crept back in"
    )


def test_separate_devices_never_alias():
    res = _resources()
    bridge: _FakeDeviceBridge = res._device_bridge
    res.ensure_uniform_device_params(_spec(7, 0), 4, 0)
    res.ensure_uniform_device_params(_spec(7, 0), 4, 1)
    assert bridge.uploads == [_STRIDE * 4, _STRIDE * 4]


# ---------------------------------------------------------------------------
# Plan + route integration (offline; the launcher seam is monkeypatched the
# same way test_rng_residency.py does for the raw egress).
# ---------------------------------------------------------------------------


@_needs_codegen_patch
def test_warm_uniform_plan_skips_the_params_build(monkeypatch):
    hr.reset_rng_resources()
    monkeypatch.setattr(hr, "_RNG_RESOURCES", _resources())
    seed = 0xC0FFEE
    hr.get_rng_resources().ensure_uniform_device_params(_spec(seed, 0), 4, 0)

    calls = []
    real = hr.draw_params_stream

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(hr, "draw_params_stream", spy)

    # COLD (no 4-tile resident for THIS seed yet is not the query — the
    # resident exists, so the plan must skip the build).
    plan = hr.build_rng_launch_plan(
        seed=seed,
        offset=256,
        egress=hr.EgressMode.UNIFORM,
        resident=True,
        n_tiles=4,
    )
    assert calls == [], "a warm uniform plan must NOT build the params words"
    assert plan.params is None
    assert plan.sticks == _STRIDE * 4
    assert plan.lanes == _LANES
    assert plan.output_sticks == 2 * 4

    # A DIFFERENT WIDTH has no resident: full build, pre-A9 bytes.
    plan2 = hr.build_rng_launch_plan(
        seed=seed,
        offset=256,
        egress=hr.EgressMode.UNIFORM,
        resident=True,
        n_tiles=2,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(plan2.params, _full(seed, 256, 2))


def test_cold_uniform_plan_builds_in_full(monkeypatch):
    hr.reset_rng_resources()
    monkeypatch.setattr(hr, "_RNG_RESOURCES", _resources())
    plan = hr.build_rng_launch_plan(
        seed=1, offset=0, egress=hr.EgressMode.UNIFORM, resident=True, n_tiles=4
    )
    assert plan.params is not None
    np.testing.assert_array_equal(plan.params, _full(1, 0, 4))


def test_resident_uniform_plan_refuses_nonzero_stream(monkeypatch):
    hr.reset_rng_resources()
    monkeypatch.setattr(hr, "_RNG_RESOURCES", _resources())
    hr.get_rng_resources().ensure_uniform_device_params(_spec(3, 0), 4, 0)
    with pytest.raises(hr.SpyreRngError, match="nonzero stream"):
        hr.build_rng_launch_plan(
            seed=3,
            offset=64,
            egress=hr.EgressMode.UNIFORM,
            resident=True,
            n_tiles=4,
            stream=5,
        )


def test_resolve_launch_params_uniform_uses_the_width_keyed_resident(monkeypatch):
    """``_resolve_launch_params`` dispatches a UNIFORM plan to the uniform
    resident of the plan's width and returns its tensor."""
    hr.reset_rng_resources()
    res = _resources()
    monkeypatch.setattr(hr, "_RNG_RESOURCES", res)
    seed = 42
    # Cold plan carries the full words; the resolver seeds the resident from
    # them (one upload, no rebuild) and returns a tensor of the declared ABI.
    plan = hr.build_rng_launch_plan(
        seed=seed, offset=0, egress=hr.EgressMode.UNIFORM, n_tiles=4
    )
    tensor = hr._resolve_launch_params(plan, 0, None)
    assert tuple(tensor.shape) == (plan.sticks, plan.lanes)
    np.testing.assert_array_equal(res._device_bridge.to_host(tensor), _full(seed, 0, 4))
    # A resident-only plan resolves through the resident (patch on advance).
    plan2 = hr.build_rng_launch_plan(
        seed=seed, offset=256, egress=hr.EgressMode.UNIFORM, resident=True, n_tiles=4
    )
    assert plan2.params is None
    tensor2 = hr._resolve_launch_params(plan2, 0, None)
    np.testing.assert_array_equal(
        res._device_bridge.to_host(tensor2), _full(seed, 256, 4)
    )
