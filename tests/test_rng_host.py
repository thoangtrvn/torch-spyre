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
"""Host-side device-RNG tests (torch-spyre lane A of the frozen RNG contract).

Covers the three obligations of the host half:

(a) REAL seed state — ``manual_seed``/``manual_seed_all`` set state (no more
    silent no-op) and ``get_rng_state``/``set_rng_state`` round-trip;
(b) the params-stream builder reproduces the three published Random123
    2x32-20 anchors, reconstructing x0/x1 bit-exact against the frozen vectors
    (and, when the backend is importable, is byte-identical to the reference
    builder it ports);
(c) the route refuses TYPED for T>=2 / pre-registration; within one T=1 tile it
    is a device fill (Path A) whose device-gated tests live in
    ``test_rng_uniform_device.py``.

The oracle JSON in ``tests/data/`` is a read-only copy of the frozen KAT vectors
from ``spyre-knowledgebase`` (provenance in the file's ``_vendored_from`` field);
it is NOT this repo's artifact. See ``docs/rng-host-device-contract.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import torch_spyre.rng as hr

_DATA = Path(__file__).parent / "data" / "rng_kat_raw_state_2x32_20.json"
_FROZEN = json.loads(_DATA.read_text())
_VECTORS = _FROZEN["vectors"]
_LANES = 32
_ROUNDS = 20


def _stream_for_vector(v: dict) -> tuple:
    """Build the params stream for one frozen KAT vector via the ported builder."""
    c0, c1 = v["counter"]
    k0, k1 = v["key"]
    blocks = np.arange(_LANES, dtype=np.uint32)
    injections = hr.injection_words_2x32((k0, k1), _ROUNDS)
    rotations = [hr.ROTATIONS_2X32[r % 8] for r in range(_ROUNDS)]
    return hr.build_band_lxlu_stream(
        blocks,
        x0_base=(c0 + k0) & 0xFFFFFFFF,
        x1=(c1 + k1) & 0xFFFFFFFF,
        rotations=rotations,
        injections=injections,
        rounds=_ROUNDS,
        egress=hr.EgressMode.RAW_STATE,
    )


# ---------------------------------------------------------------------------
# (b) The params stream reproduces the three published anchors, bit-exact.
# ---------------------------------------------------------------------------


def test_params_stream_is_251_raw_sticks():
    stream = _stream_for_vector(_VECTORS[0])
    assert (
        len(stream)
        == 251
        == hr.band_lxlu_pop_count(_ROUNDS, egress=hr.EgressMode.RAW_STATE)
    )
    sticks = hr.stream_to_sticks(stream)
    assert sticks.shape == (251, _LANES)
    assert sticks.dtype == np.uint32
    # The four band setup constants arrive as their own bit patterns, in order.
    head = [int(w.words[0]) for w in stream[:4]]
    assert head == [0x4B00FFFF, 0x4B7F0000, 0x37800000, 0x4B000000]
    # Slot order is the declared pop order.
    assert stream[0].slot is hr.StreamSlot.BAND_SETUP_M16B
    assert stream[4].slot is hr.StreamSlot.COUNTER
    assert stream[-1].slot is hr.StreamSlot.INJ0_HI


@pytest.mark.parametrize("index", [0, 1, 2])
def test_builder_reproduces_published_anchors_bit_exact(index):
    """Replay the params stream through the band arithmetic; match the freeze."""
    v = _VECTORS[index]
    stream = _stream_for_vector(v)
    x0, x1 = hr.replay_band_stream_raw(stream)
    assert x0.tolist() == v["expected_x0"]
    assert x1.tolist() == v["expected_x1"]
    # Lane 0 IS the PUBLISHED Random123 2x32-20 vector (lanes 1..31 are the
    # external reference oracle, same published spec).
    assert int(x0[0]) == v["expected_x0"][0]
    assert int(x1[0]) == v["expected_x1"][0]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_replay_matches_the_independent_threefry_oracle(index):
    """Two methods: stream replay == the canonical Threefry2x32-20 oracle."""
    v = _VECTORS[index]
    c0, c1 = v["counter"]
    k0, k1 = v["key"]
    blocks = np.arange(_LANES, dtype=np.uint32)
    x0, x1 = hr.replay_band_stream_raw(_stream_for_vector(v))
    ref0, ref1 = hr.threefry2x32_20(
        (blocks + np.uint32(c0)).astype(np.uint32),
        np.full(_LANES, np.uint32(c1), dtype=np.uint32),
        (k0, k1),
    )
    assert x0.tolist() == ref0.tolist()
    assert x1.tolist() == ref1.tolist()


def test_builder_is_byte_identical_to_the_reference_it_ports():
    """The port must be byte-for-byte the reference builder (skip if absent).

    This is the contract's load-bearing check: a divergence here is the exact
    failure mode the contract exists to prevent. Byte parity is checked with a
    DIFFERENT method than the arithmetic replay above (an independent
    implementation comparison, not the same threefry oracle).
    """
    try:
        from sentient_codegen.emulator import threefry_recipe_model as rm
        from sentient_codegen.emulator import threefry_reference as tf
    except Exception:
        pytest.skip(
            "sentient_codegen not importable (source env.sh) — "
            "byte-parity vs the reference builder not checked here"
        )
    for v in _VECTORS:
        c0, c1 = v["counter"]
        k0, k1 = v["key"]
        blocks = np.arange(_LANES, dtype=np.uint32)
        injections = tf.injection_words_2x32((k0, k1), _ROUNDS)
        rotations = [tf.ROTATIONS_2X32[r % 8] for r in range(_ROUNDS)]
        ref = rm.build_band_lxlu_stream(
            blocks,
            x0_base=(c0 + k0) & 0xFFFFFFFF,
            x1=(c1 + k1) & 0xFFFFFFFF,
            rotations=rotations,
            injections=injections,
            rounds=_ROUNDS,
            egress=rm.EgressMode.RAW_STATE,
        )
        port = hr.build_band_lxlu_stream(
            blocks,
            x0_base=(c0 + k0) & 0xFFFFFFFF,
            x1=(c1 + k1) & 0xFFFFFFFF,
            rotations=rotations,
            injections=injections,
            rounds=_ROUNDS,
            egress=hr.EgressMode.RAW_STATE,
        )
        assert len(ref) == len(port) == 251
        for i, (a, b) in enumerate(zip(ref, port)):
            assert a.slot.value == b.slot.value, i
            assert np.array_equal(a.words, b.words), i


def test_draw_params_stream_derives_from_seed_and_offset():
    """The seed->key->fold chain yields a full T=1 stream; seeds differ."""
    s0 = hr.draw_params_stream(seed=0, offset=0)
    s1 = hr.draw_params_stream(seed=1, offset=0)
    assert len(s0) == len(s1) == 251
    assert hr.stream_to_sticks(s0).shape == (251, _LANES)
    # Different seeds must not produce an identical stream.
    assert not all(np.array_equal(a.words, b.words) for a, b in zip(s0, s1))
    # Deterministic for the same (seed, offset).
    s0_again = hr.draw_params_stream(seed=0, offset=0)
    assert all(np.array_equal(a.words, b.words) for a, b in zip(s0, s0_again))


def test_counter_block_limit_is_refused():
    from torch_spyre.rng import SpyreRngError

    with pytest.raises(SpyreRngError):
        hr.build_band_lxlu_stream(
            np.array([1 << 23], dtype=np.uint32),
            x0_base=0,
            x1=0,
            rotations=[hr.ROTATIONS_2X32[r % 8] for r in range(_ROUNDS)],
            injections=hr.injection_words_2x32((0, 0), _ROUNDS),
            rounds=_ROUNDS,
        )


# ---------------------------------------------------------------------------
# (a) Real generator/seed state — no silent no-op; state round-trips.
# ---------------------------------------------------------------------------


def test_generator_state_round_trips():
    st = hr.SpyreGeneratorState()
    st.manual_seed(123456789)
    assert st.seed == 123456789
    assert st.offset == 0
    st.offset = 4096
    barrel = st.to_tensor()
    restored = hr.SpyreGeneratorState.from_tensor(barrel)
    assert restored.seed == 123456789
    assert restored.offset == 4096


def test_generator_state_rejects_a_foreign_blob():
    import torch

    with pytest.raises(ValueError):
        hr.SpyreGeneratorState.from_tensor(torch.zeros(16, dtype=torch.uint8))


def test_reserve_never_straddles_an_epoch():
    st = hr.SpyreGeneratorState()
    st.manual_seed(0)
    epoch = 1 << 30
    st.offset = epoch - 1  # one block before the epoch boundary
    start = st.reserve(4)
    # The draw is pushed to the next epoch rather than crossing the boundary.
    assert start == epoch
    assert st.offset == epoch + 4


def _spyre_module_or_skip():
    import torch

    mod = getattr(torch, "spyre", None)
    if mod is None or not hasattr(mod, "get_rng_state"):
        pytest.skip("torch.spyre device module unavailable in this environment")
    return mod


def test_manual_seed_sets_real_state_and_get_set_round_trip():
    """The old behavior was a silent no-op; assert state actually changes."""
    spyre = _spyre_module_or_skip()
    spyre.manual_seed(987654)
    state_a = spyre.get_rng_state()
    decoded = hr.SpyreGeneratorState.from_tensor(state_a)
    assert decoded.seed == 987654
    assert decoded.offset == 0

    # A different seed visibly changes the state.
    spyre.manual_seed(111)
    state_b = spyre.get_rng_state()
    assert not bool((state_a == state_b).all())

    # set_rng_state round-trips the earlier state exactly.
    spyre.set_rng_state(state_a)
    assert bool((spyre.get_rng_state() == state_a).all())
    assert hr.SpyreGeneratorState.from_tensor(spyre.get_rng_state()).seed == 987654


def test_manual_seed_all_sets_state():
    spyre = _spyre_module_or_skip()
    spyre.manual_seed_all(424242)
    decoded = hr.SpyreGeneratorState.from_tensor(spyre.get_rng_state())
    assert decoded.seed == 424242


# ---------------------------------------------------------------------------
# (c) The route refuses TYPED pre-registration and T>=2; within one T=1 tile it
#     is a device fill (Path A), not a refusal.
# ---------------------------------------------------------------------------


def test_launch_plan_refuses_typed_pre_registration(monkeypatch):
    from torch_spyre.rng import SpyreRngError

    # Pin the pre-registration state regardless of the ambient backend.
    monkeypatch.setattr(hr, "rng_pattern_registered", lambda: False)
    with pytest.raises(SpyreRngError) as excinfo:
        hr.build_rng_launch_plan(seed=1, offset=0)
    # A typed refusal propagates as a hard failure (NotImplementedError).
    assert isinstance(excinfo.value, NotImplementedError)
    assert "not wired yet" in str(excinfo.value) or "rng_kat" in str(excinfo.value)


def test_launch_plan_multitile_gate(monkeypatch):
    """Multi-tile is wired only for the silicon-witnessed UNIFORM widths.

    The tile count is controlled by ``n_tiles`` (not ``numel``, which is the
    per-tile lane count). A RAW / unwitnessed multi-tile request refuses typed;
    a witnessed UNIFORM width builds the plan (output 2n sticks, n*257 params).
    """
    from torch_spyre.rng import SpyreRngError

    # RAW multi-tile has no silicon receipt -> typed refusal (fires before
    # registration, so no monkeypatch needed here).
    with pytest.raises(SpyreRngError) as excinfo:
        hr.build_rng_launch_plan(
            seed=1, numel=hr.RNG_KAT_LANES, egress=hr.EgressMode.RAW_STATE, n_tiles=2
        )
    assert "T>=2" in str(excinfo.value) or "T-DV-34" in str(excinfo.value)
    # UNIFORM at a witnessed width builds.
    monkeypatch.setattr(hr, "rng_pattern_registered", lambda: True)
    plan = hr.build_rng_launch_plan(
        seed=1, numel=hr.RNG_KAT_LANES, egress=hr.EgressMode.UNIFORM, n_tiles=2
    )
    assert plan.n_tiles == 2
    assert plan.output_sticks == 4
    assert plan.sticks == 514


def test_launch_plan_builds_the_uniform_plan(monkeypatch):
    """UNIFORM is a route now: the plan builds (257 params sticks, 2 out sticks).

    The old assertion here was that the uniform grid REFUSED (P5 owed). With the
    silicon-witnessed UNIFORM egress landed, ``build_rng_launch_plan`` builds it:
    pops 257 (vs raw 251), egress 2 sticks (vs raw 4)."""
    # Registration must be live for a non-resident build; pin it so the test
    # does not depend on the ambient backend import state.
    monkeypatch.setattr(hr, "rng_pattern_registered", lambda: True)
    plan = hr.build_rng_launch_plan(seed=1, numel=32, egress=hr.EgressMode.UNIFORM)
    assert plan.egress is hr.EgressMode.UNIFORM
    assert plan.sticks == hr.band_lxlu_pop_count(_ROUNDS, egress=hr.EgressMode.UNIFORM)
    assert plan.sticks == 257
    assert plan.output_sticks == hr.band_egress_stick_count(
        _ROUNDS, egress=hr.EgressMode.UNIFORM
    )
    assert plan.output_sticks == 2
    assert plan.params is not None and plan.params.shape == (257, 32)


def test_route_uniform_draw_refuses_only_beyond_one_tile(monkeypatch):
    """Path A contract: only T>=2 (numel > 64) is a typed refusal.

    Within one T=1 tile (numel <= ``_RNG_MAX_UNIFORM_ELEMS``) the route is a
    FILL, not a refusal: it succeeds on a live device, or raises the
    registration / runtime-unavailable seam — never the T>=2 refusal.
    """
    from torch_spyre.rng import SpyreRngError, _RNG_MAX_UNIFORM_ELEMS

    monkeypatch.setenv(hr.DeviceRngSwitch.ENV, "1")
    assert hr.device_rng_enabled() is True
    # More than one T=1 tile is T>=2: typed refusal.
    with pytest.raises(SpyreRngError) as excinfo:
        hr.route_uniform_draw(0.0, 1.0, _RNG_MAX_UNIFORM_ELEMS + 1)
    assert "T>=2" in str(excinfo.value) or "T-DV-34" in str(excinfo.value)
    # One full tile is NOT a T>=2 refusal (fill / registration / runtime seam).
    try:
        out = hr.route_uniform_draw(0.0, 1.0, _RNG_MAX_UNIFORM_ELEMS)
    except Exception as exc:  # noqa: BLE001 — either pre-device seam is acceptable
        msg = str(exc)
        assert "T>=2" not in msg and "T-DV-34" not in msg
    else:
        assert out.shape == (_RNG_MAX_UNIFORM_ELEMS,)


def test_normal_kernel_refuses_when_switch_on(monkeypatch):
    """normal_ still refuses with the switch on (P1 Box-Muller unwitnessed).

    (uniform_ was split out: after Path A it fills from device bits up to one
    T=1 tile; its device-gated fill tests live in test_rng_uniform_device.py.)
    """
    import torch

    spyre = _spyre_module_or_skip()
    from torch_spyre.rng import SpyreRngError

    if spyre.is_available() is not True:
        pytest.skip("no live spyre device to allocate a tensor (codegen-only)")
    monkeypatch.setenv(hr.DeviceRngSwitch.ENV, "1")
    x = torch.empty(32, device="spyre")
    with pytest.raises(SpyreRngError):
        torch.ops.aten.normal_(x, 0.0, 1.0)


def test_uniform_refuses_a_per_call_generator_on_the_device_path(monkeypatch):
    """Device RNG has ONE per-device default generator; a per-call ``generator=``
    is not supported yet (multi-stream unbuilt). It must refuse TYPED, not
    silently draw from the default stream — the switch-off host path DOES honor
    ``generator=``, so a silent ignore here would be a correctness trap.

    Calls the guard helper directly (the decorated kernel name resolves to
    ``None`` on import, and the public generator overload does not route here
    in a codegen-only build), so the guard is asserted where it is enforced."""
    import torch

    from torch_spyre.ops.eager import _reject_per_call_generator
    from torch_spyre.rng import SpyreRngError

    monkeypatch.setenv(hr.DeviceRngSwitch.ENV, "1")
    assert hr.device_rng_enabled() is True
    g = torch.Generator()
    with pytest.raises(SpyreRngError) as excinfo:
        _reject_per_call_generator(g)
    assert "generator" in str(excinfo.value).lower()
    # A missing generator is allowed (no-op).
    assert _reject_per_call_generator(None) is None


def test_local_band_counts_mirror_the_codegen_owner():
    """Drift guard: the LOCAL band-count mirrors equal the codegen owners.

    ``torch_spyre.rng.band_lxlu_pop_count`` / ``band_egress_stick_count`` are
    local mirrors hand-derived from the ``BAND_*_SLOTS`` tuples, NOT the codegen
    owner. When codegen is importable, assert the mirror equals
    ``sentient_codegen.emulator.threefry_recipe_model``; skip with a clear
    reason when it is absent (this file has no module-level codegen import).
    """
    try:
        from sentient_codegen.emulator import threefry_recipe_model as rm
    except Exception:
        pytest.skip(
            "sentient_codegen not importable (source env.sh) — local band-count "
            "mirror vs codegen owner not checked here"
        )
    for egress in (hr.EgressMode.RAW_STATE, hr.EgressMode.UNIFORM):
        cg_egress = hr._to_codegen_egress(egress)
        assert hr.band_lxlu_pop_count(_ROUNDS, egress=egress) == (
            rm.band_lxlu_pop_count(_ROUNDS, egress=cg_egress)
        ), f"band_lxlu_pop_count drift for {egress!r}"
        assert hr.band_egress_stick_count(_ROUNDS, egress=egress) == (
            rm.band_egress_stick_count(_ROUNDS, egress=cg_egress)
        ), f"band_egress_stick_count drift for {egress!r}"
