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
"""Offline connection proof: lane A's host params + lane B's registered kernel.

The two halves of the frozen RNG contract are tested in isolation elsewhere
(``tests/test_rng_host.py`` for the host params stream; lane B's codegen tests
for the pattern and binary). This file is the CONNECTION proof — the only place
both halves must agree end to end — and it is deliberately OFFLINE (no device):

1. **Connected compile byte-match.** The launcher's ``build_rng_kat_binary``
   compiles the registered ``rng_kat`` route and the result is byte-identical to
   the silicon-witnessed raw binary ``e697b657…`` (4352 B). A green compile is
   not a value proof; this pins the artifact.

2. **Full-transport emulator replay.** Lane A's params stream — the SAME
   ``build_band_lxlu_stream`` / ``stream_to_sticks`` the host route uses — is
   fed through lane B's emitted kernel (``L3LU -> LXLU -> SFP -> LXSU -> L3SU``)
   with an ``LXScratchpad`` and a real ``HBMMemory``. The store transport is
   exercised (the egress stride is load-bearing), and the reconstructed full
   32-bit ``x0/x1`` must equal the three published Random123 2x32-20 anchors
   BIT-EXACT for all 32 lanes. This is the proof the params and the kernel agree.

Requires lane B's ``sentient_codegen`` tree on ``PYTHONPATH`` (with ``rng_kat``
registered); otherwise the module skips rather than pretending. No hardware and
no ``spyre`` device are required.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import torch_spyre.rng as hr

pytest.importorskip(
    "sentient_codegen",
    reason="lane B's sentient_codegen is not importable — source env.sh / add "
    "the registered codegen tree to PYTHONPATH to run the connection proof",
)

if not hr.rng_pattern_registered():  # lane B has not registered rng_kat here
    pytest.skip(
        "backend pattern 'rng_kat' is not registered — the refusing base will not "
        "emit the KAT binary; point PYTHONPATH at the registered codegen tree",
        allow_module_level=True,
    )

_DATA = Path(__file__).parent / "data" / "rng_kat_raw_state_2x32_20.json"
_FROZEN = json.loads(_DATA.read_text())
_VECTORS = _FROZEN["vectors"]
_LANES = 32
_ROUNDS = 20
_TARGET = "rcudd1a"
_OUTPUT_SEGMENT = 1
_RAW_SHA256 = "e697b657d14269f219626f9e7418b29c223332e2e8eb03a61c3e091f4d263d9d"


# ---------------------------------------------------------------------------
# 1. Connected compile: the artifact is the silicon-witnessed binary.
# ---------------------------------------------------------------------------


def test_connected_compile_emits_the_silicon_witnessed_raw_binary():
    binary, xlat_plan = hr.build_rng_kat_binary()
    assert hashlib.sha256(binary).hexdigest() == _RAW_SHA256
    assert len(binary) == 4352
    # The plan rides with the SAME scheduled program: two symbols, read params
    # then write output — the frozen 2-segment ABI.
    syms = [
        (s.symbol_id, s.access_mode, s.required_extent_bytes) for s in xlat_plan.symbols
    ]
    assert [a for _, a, _ in syms] == ["read", "write"]
    assert [e for _, _, e in syms] == [251 * 128, 4 * 128]


def test_launch_plan_declares_the_two_segment_abi():
    plan = hr.build_rng_launch_plan(seed=0)
    assert plan.declared_segment_count == 2
    assert plan.xlat_segments == {"input": 0, "output": 1}
    assert plan.sticks == 251 and plan.lanes == 32 and plan.output_sticks == 4
    assert plan.params.shape == (251, 32) and plan.params.dtype == np.uint32


# ---------------------------------------------------------------------------
# 2. Full-transport emulator replay: lane A's params through lane B's kernel.
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


def _host_param_sticks(vector: dict):
    """Lane A's params stream for one frozen vector, as emulator DataSticks.

    This uses the SAME builder the host route uses (``build_band_lxlu_stream`` +
    ``stream_to_sticks``) — the point of the test is that these bytes, not a
    re-derived copy, drive the kernel.
    """
    from sentient_codegen.emulator.data import DataStick

    c0, c1 = vector["counter"]
    k0, k1 = vector["key"]
    blocks = np.arange(_LANES, dtype=np.uint32)
    stream = hr.build_band_lxlu_stream(
        blocks,
        x0_base=(c0 + k0) & 0xFFFFFFFF,
        x1=(c1 + k1) & 0xFFFFFFFF,
        rotations=[hr.ROTATIONS_2X32[r % 8] for r in range(_ROUNDS)],
        injections=hr.injection_words_2x32((k0, k1), _ROUNDS),
        rounds=_ROUNDS,
        egress=hr.EgressMode.RAW_STATE,
    )
    array = hr.stream_to_sticks(stream)  # (251, 32) uint32, recipe order
    sticks = []
    for row in array:
        stick = DataStick.from_numpy(np.asarray(row).view(np.float32), "f32")
        stick.raw_words = np.array(row, dtype=np.uint32)
        sticks.append(stick)
    return sticks


def _run_full_transport(binary: bytes, vector: dict):
    """L3LU -> LXLU -> SFP -> LXSU -> L3SU; return the segment-1 egress sticks."""
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
    hbm.write(0, StickOffset(0), _host_param_sticks(vector))  # params @ segment 0
    loghbm = ReadWriteLoggingHBM(hbm)
    scratch = LXScratchpad(gen)
    executor = DataflowSequentialExecutor(target=_TARGET, sfp_readings=_sfp_readings())
    executor.execute(programs, loghbm, scratch, spr, inits)

    assert scratch.fallback_events == {}, "params did not reach LX (fabricated zeros)"
    output_writes = [w for w in loghbm.writes if w[0] == _OUTPUT_SEGMENT]
    assert len(output_writes) == 1, f"expected one output store, got {loghbm.writes}"
    return output_writes[0][2]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_host_params_replay_bit_exact_for_all_published_anchors(index):
    from sentient_codegen.emulator import threefry_recipe_model as rm

    binary, _ = hr.build_rng_kat_binary()
    vector = _VECTORS[index]
    sticks = _run_full_transport(binary, vector)
    assert len(sticks) == 4, "the raw egress stores four sticks"
    words = [np.asarray(s.raw_words, dtype=np.uint32) for s in sticks]
    x0, x1 = rm.reconstruct_raw_state_egress(words)
    assert list(map(int, x0)) == vector["expected_x0"]
    assert list(map(int, x1)) == vector["expected_x1"]
