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
"""OFFLINE end-to-end proof of the user-facing UNIFORM draw (multi-tile).

The device tests in ``test_rng_uniform_device.py`` exercise ``uniform_`` on real
silicon (and skip without a device). This file proves the SAME route OFFLINE,
without hardware: it feeds ``torch_spyre.rng``'s own params stream (the bytes
``build_rng_launch_plan`` produces, the exact stream the device launcher binds
to segment 0) through the emitted UNIFORM kernel (``build_rng_kat_uniform_binary``)
in the emulator's full transport (``L3LU -> LXLU -> SFP -> LXSU -> L3SU`` with an
``LXScratchpad`` and a real ``HBMMemory``), then checks the stored uniforms
BIT-EXACT against the host Threefry oracle.

The uniform epilogue writes ``u = (w >> 8) * 2**-24`` per lane (the C5/P5 grid,
``raw_words_to_uniform``); each tile is one counter block = 2 egress sticks (the
x0 uniforms then the x1 uniforms), so ``n`` tiles = ``64 * n`` fp32 values.

WHY THIS IS THE INTEGRATION PROOF, NOT A RE-DERIVATION
------------------------------------------------------
The expected values come from a DIFFERENT implementation than the bytes under
test: the host ``threefry2x32_20`` oracle keyed from ``(seed, offset)``. A
params-stream or key-derivation divergence between the torch host half and the
codegen kernel shows up here as a bit mismatch. (A deliberately double-folded
key derivation was run while developing this file and FAILED — the test has
teeth; it is not a tautology.)

EMULATOR BOUNDARY (stated so the proof is not over-read)
--------------------------------------------------------
The device tile loop stores egress in 16-tile / 32-stick bursts, so a wide draw
is SEVERAL segment-1 writes; this file concatenates them in ascending device
offset to recover draw order. The full loop is executed: ``n`` tiles = ``64*n``
values. The route admits ``1..63`` tiles (``_RNG_WITNESSED_UNIFORM_TILES_MAX``,
the LX block-load ceiling) and refuses ``>= 64`` typed. Bit-exact replay is
proven here across that whole envelope — ``n in {1,2,4,16,32,63}`` including the
full 63-tile / 4032-value draw — against the host Threefry oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

import torch_spyre.rng as hr

pytest.importorskip(
    "sentient_codegen",
    reason="codegen backend not importable — source env.sh / add the codegen "
    "tree to PYTHONPATH for the offline uniform replay",
)

if not hr.rng_pattern_registered():
    pytest.skip(
        "backend pattern 'rng_kat' is not registered in the importable codegen "
        "tree; point PYTHONPATH at a codegen tip that carries it",
        allow_module_level=True,
    )

_TARGET = "rcudd1a"
_OUTPUT_SEGMENT = 1
SEED = 42


# ---------------------------------------------------------------------------
# Emulator harness (full transport).
# ---------------------------------------------------------------------------


def _sfp_readings():
    from sentient_codegen.emulator import threefry_recipe_model as rm

    hw = rm.HardwareReadings(
        xnor_width=rm.XnorWidth.FULL_WORD,
        and_not_result=rm.AndNotResult.BIT_PATTERN,
        right_shift_fill=rm.RightShiftFill.LOGICAL,
        left_shift_16=rm.LeftShift16.TRUNCATES,
        i2f32_signedness=rm.I2F32Signedness.SIGNED,
        r2i32_rounding=rm.R2I32Rounding.FLOOR,
        fma_rounding=rm.FmaRounding.SINGLE,
        icvt_executes_at=rm.IcvtModeBit.M0,
    )
    return rm.SfpReadings(
        hardware=hw,
        rfp32_tie=rm.Rfp32Tie.TIES_AWAY,
        zero_sign=rm.ZeroSign.POSITIVE_ZERO,
        shr32_shift_by=rm.Shr32ShiftBy.SIGNED_6BIT,
        fmul_m1=rm.FmulM1.RFP32,
        logical_operand_kind=rm.LogicalOperandKind.RAW_BITS_ANY_KIND,
        m1_fma_width=rm.M1ArithWidth.RFP32,
        m1_fnms_width=rm.M1ArithWidth.RFP32,
    )


def _param_sticks(params: np.ndarray):
    """Rows of the (sticks, 32) uint32 params array as emulator DataSticks."""
    from sentient_codegen.emulator.data import DataStick

    sticks = []
    for row in np.asarray(params):
        stick = DataStick.from_numpy(np.asarray(row).view(np.float32), "f32")
        stick.raw_words = np.array(row, dtype=np.uint32)
        sticks.append(stick)
    return sticks


def _run_uniform_transport(binary: bytes, params: np.ndarray) -> np.ndarray:
    """Replay ``binary`` with ``params`` at segment 0; return segment-1 fp32."""
    from sentient_codegen.emulator.data import HBMMemory, LXScratchpad
    from sentient_codegen.emulator.decode_to_ir import build_executor_unit_programs
    from sentient_codegen.emulator.executor import DataflowSequentialExecutor
    from sentient_codegen.emulator.loader import load_init_packets_bytes
    from sentient_codegen.emulator.multicore import (
        ReadWriteLoggingHBM,
        _base_register_inits_and_spr,
    )
    from sentient_codegen.gen.registry import get_generation
    from sentient_codegen.scheduler.address_units import StickOffset

    gen = get_generation(_TARGET)
    packet = load_init_packets_bytes(binary, gen, _TARGET, record_sites=True)[0]
    programs = build_executor_unit_programs(packet)
    inits, spr = _base_register_inits_and_spr(packet, programs)

    hbm = HBMMemory(gen)
    hbm.write(0, StickOffset(0), _param_sticks(params))  # params @ segment 0
    loghbm = ReadWriteLoggingHBM(hbm)
    scratch = LXScratchpad(gen)
    DataflowSequentialExecutor(target=_TARGET, sfp_readings=_sfp_readings()).execute(
        programs, loghbm, scratch, spr, inits
    )

    assert scratch.fallback_events == {}, "params did not reach LX (fabricated zeros)"
    # The device tile loop stores in chunks (the L3SU store drains egress in
    # 16-tile / 32-stick bursts), so the full width is several segment-1 writes;
    # concatenate them in ascending device offset to recover the draw order.
    output_writes = [w for w in loghbm.writes if w[0] == _OUTPUT_SEGMENT]
    assert output_writes, f"expected output stores, got {loghbm.writes}"
    output_writes.sort(key=lambda w: int(w[1].value))
    words = np.concatenate(
        [
            np.asarray(s.raw_words, dtype=np.uint32)
            for _seg, _off, sticks in output_writes
            for s in sticks
        ]
    )
    # rfp32/float words -> the stored fp32 uniform values.
    return words.view(np.float32)


def _host_uniforms(seed: int, offset: int, n_tiles: int) -> np.ndarray:
    """Host Threefry2x32-20 oracle mapped to uniforms, per tile.

    Tile ``t`` advances the counter base by ``t`` (``draw_params_stream``'s
    device tile loop); the key is the host-folded epoch key. Each tile yields
    64 values: the x0 lanes then the x1 lanes, each mapped by
    ``raw_words_to_uniform`` — the same C5/P5 grid the uniform epilogue writes.
    """
    seed_key = hr.seed_to_key(seed)
    k_prime = hr.epoch_key_2x32(seed_key, offset >> 30, device=0, stream=0)
    blocks = np.arange(hr.RNG_KAT_LANES, dtype=np.uint32)
    zeros = np.zeros(hr.RNG_KAT_LANES, dtype=np.uint32)
    per_tile = []
    for t in range(n_tiles):
        x0, x1 = hr.threefry2x32_20(blocks + np.uint32(t), zeros, k_prime)
        per_tile.append(hr.raw_words_to_uniform(np.concatenate([x0, x1])))
    return np.concatenate(per_tile)


# ---------------------------------------------------------------------------
# 1. End-to-end bit-exactness through the emulator's supported depth.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_tiles", [1, 2, 4, 16, 32, 63])
def test_uniform_draw_is_bit_exact_end_to_end(n_tiles):
    plan = hr.build_rng_launch_plan(
        seed=SEED, offset=0, egress=hr.EgressMode.UNIFORM, n_tiles=n_tiles
    )
    binary, _xlat = hr.build_rng_kat_uniform_binary(n_tiles=n_tiles)
    got = _run_uniform_transport(binary, plan.params)

    assert got.size == 64 * n_tiles, f"one tile = 64 uniforms; got {got.size}"
    expected = _host_uniforms(SEED, 0, n_tiles)
    np.testing.assert_array_equal(
        got.view(np.uint32),
        expected.view(np.uint32),
        err_msg=f"uniform draw diverged from the host Threefry oracle (n={n_tiles})",
    )


def test_successive_tiles_advance_the_counter():
    """Tile t depends on its counter base, not on the tile index repeat — the
    second tile's stream differs from the first (guards a stuck tile loop)."""
    plan = hr.build_rng_launch_plan(
        seed=SEED, offset=0, egress=hr.EgressMode.UNIFORM, n_tiles=2
    )
    binary, _ = hr.build_rng_kat_uniform_binary(n_tiles=2)
    got = _run_uniform_transport(binary, plan.params)
    assert not np.array_equal(got[:64], got[64:]), "tile 1 repeated tile 0"


# ---------------------------------------------------------------------------
# 2. Envelope: admitted through 63 tiles, typed-refused above; loop parametric.
# ---------------------------------------------------------------------------


def test_route_admits_the_witnessed_cap_and_builds_a_loop_binary():
    """n_tiles 1..63 is admitted (the LX block-load ceiling). The tile count is
    carried in one loop instruction, so the emitted binary is the same SIZE at
    every width (not unrolled per tile); only the loop-count immediate differs."""
    cap = hr._RNG_WITNESSED_UNIFORM_TILES_MAX
    assert cap == 63
    b1, _ = hr.build_rng_kat_uniform_binary(n_tiles=1)
    bcap, _ = hr.build_rng_kat_uniform_binary(n_tiles=cap)
    # The tile count rides in the loop instruction, not in a per-tile unroll: a
    # 63-tile binary must be far smaller than 63 copies of the single-tile body
    # (a width-proportional unroll would overflow the LX block load).
    assert len(bcap) < 2 * len(b1), (
        f"tile loop appears unrolled: n=63 binary {len(bcap)} B vs 63x single "
        f"tile {63 * len(b1)} B (single-tile body {len(b1)} B)"
    )
    plan = hr.build_rng_launch_plan(
        seed=SEED, offset=0, egress=hr.EgressMode.UNIFORM, n_tiles=cap
    )
    # 257 pops/tile (uniform egress) x cap tiles, 32 lanes.
    assert plan.params.shape == (257 * cap, hr.RNG_KAT_LANES)
    assert plan.n_tiles == cap


def test_cap_plus_one_tile_refuses_typed_at_build_and_plan():
    cap = hr._RNG_WITNESSED_UNIFORM_TILES_MAX
    with pytest.raises(hr.SpyreRngError, match="cap"):
        hr.build_rng_kat_uniform_binary(n_tiles=cap + 1)
    with pytest.raises(hr.SpyreRngError):
        hr.build_rng_launch_plan(
            seed=SEED, offset=0, egress=hr.EgressMode.UNIFORM, n_tiles=cap + 1
        )


def test_non_whole_tile_width_refuses_typed():
    """A width that is not a whole number of 64-word tiles refuses typed (never
    a silent split)."""
    from torch_spyre.rng import _select_uniform_binding

    with pytest.raises(hr.SpyreRngError):
        _select_uniform_binding(
            numel=65,
            from_=0.0,
            to=1.0,
            is_float32=True,
            is_contiguous=True,
            device_size=(2, 32),
        )


def test_unwitnessed_target_refuses_typed():
    """RNG is 1p0-only today; a 1p5 target refuses rather than emit a
    wrong-generation binary."""
    with pytest.raises(hr.SpyreRngError, match="1p0"):
        hr.build_rng_kat_uniform_binary(target="sen1p5")


# ---------------------------------------------------------------------------
# 3. The wide draw drains egress in bursts; the concatenation recovers order.
# ---------------------------------------------------------------------------


def test_wide_draw_drains_multiple_egress_bursts_in_offset_order():
    """A 63-tile draw exceeds the 16-tile / 32-stick L3SU store burst, so it is
    SEVERAL segment-1 writes. The full 4032 values are recovered only by
    concatenating the bursts in ascending device offset (which the bit-exact
    n=63 case above already asserts); this pins the burst mechanics explicitly."""
    from sentient_codegen.emulator.data import HBMMemory, LXScratchpad
    from sentient_codegen.emulator.decode_to_ir import build_executor_unit_programs
    from sentient_codegen.emulator.executor import DataflowSequentialExecutor
    from sentient_codegen.emulator.loader import load_init_packets_bytes
    from sentient_codegen.emulator.multicore import (
        ReadWriteLoggingHBM,
        _base_register_inits_and_spr,
    )
    from sentient_codegen.gen.registry import get_generation
    from sentient_codegen.scheduler.address_units import StickOffset

    plan = hr.build_rng_launch_plan(
        seed=SEED, offset=0, egress=hr.EgressMode.UNIFORM, n_tiles=63
    )
    binary, _ = hr.build_rng_kat_uniform_binary(n_tiles=63)
    gen = get_generation(_TARGET)
    packet = load_init_packets_bytes(binary, gen, _TARGET, record_sites=True)[0]
    programs = build_executor_unit_programs(packet)
    inits, spr = _base_register_inits_and_spr(packet, programs)
    hbm = HBMMemory(gen)
    hbm.write(0, StickOffset(0), _param_sticks(plan.params))
    loghbm = ReadWriteLoggingHBM(hbm)
    DataflowSequentialExecutor(target=_TARGET, sfp_readings=_sfp_readings()).execute(
        programs, loghbm, LXScratchpad(gen), spr, inits
    )
    writes = [w for w in loghbm.writes if w[0] == _OUTPUT_SEGMENT]
    assert len(writes) > 1, "a 63-tile draw must drain in multiple store bursts"
    assert sum(len(w[2]) for w in writes) == 2 * 63, (
        "the bursts together must cover every egress stick (no dropped tail)"
    )
    offsets = [int(w[1].value) for w in writes]
    assert offsets == sorted(offsets), "bursts must be contiguous ascending"
