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
"""Host side of device RNG: generator state, the params stream, and the route.

Two halves of one contract live here; the codegen backend (lane B) owns the
other. The FROZEN contract is ``docs/rng-host-device-contract.md`` in this repo
(copied from ``receipts/rng-connect-2026-11-19/CONTRACT.md``). Read it first.
This module does NOT launch a kernel and does NOT draw on the device; see
"What is and is not connected" below.

WHAT THIS MODULE OWNS (the torch-spyre / host half)
---------------------------------------------------
1. **Generator state.** ``SpyreGeneratorState`` is a real ``(seed, offset)``
   state, one per device. ``manual_seed``/``manual_seed_all`` set it (no more
   silent no-op); ``get_rng_state``/``set_rng_state`` round-trip it. This is
   the design-of-record state from doc 43 §3.1 / §3.2: a u64 seed and a u64
   block offset, one generator per device.
2. **The params stream.** ``build_band_lxlu_stream`` is a THIN DELEGATING
   WRAPPER around the single owner in codegen,
   ``sentient_codegen.emulator.threefry_recipe_model.build_band_lxlu_stream``
   (``threefry_recipe_model.py``). The recipe is NOT reimplemented here: this
   module only adapts the codegen ``StreamWord`` tuple into torch-spyre's
   ``(sticks, lanes)`` numpy form and translates the codegen ``StreamSlot`` /
   ``EgressMode`` / ``SetupStaging`` vocabulary into the local mirrors. A
   divergence is the exact failure mode the contract exists to prevent, so
   ``tests/test_rng_host.py`` still pins the result against the three published
   Random123 2x32-20 anchors (x0/x1 bit-exact) and against the reference
   builder byte-for-byte. The same delegation applies to ``threefry2x32_20``,
   ``key_schedule_2x32`` and ``injection_words_2x32`` (owner:
   ``sentient_codegen.emulator.threefry_reference``).
3. **The route scaffold.** ``build_rng_launch_plan`` turns a draw into the
   params stream and a launch ABI, and REFUSES TYPED (``SpyreRngError``) until
   lane B registers the ``rng_kat`` pattern. It also refuses, typed, the
   unwitnessed uniform fp32 grid and every T≥2 request.

WHAT IS AND IS NOT CONNECTED (honesty bounds — do not blur these)
----------------------------------------------------------------
- The device draw path is NOT live on its own. It needs (a) lane B's ``rng_kat``
  registration AND (b) the integration step that launches the kernel and reads
  the raw egress. Until both land, the user-facing ``uniform_``/``normal_``
  kernels stay the DOCUMENTED host-CPU offload (``torch_spyre/ops/eager.py``);
  with the device-RNG switch on they REFUSE typed instead of pretending.
- The raw-state egress (4 sticks, ``e697b657…``) is the silicon-witnessed
  artifact. The uniform fp32 grid ``u = hi*2^-16 + floor(lo/256)*2^-24`` is
  UNWITNESSED (P5 owed) and is refused — never claimed.
- T=1, C=1, 32 lanes, rounds=20 only. T≥2 is unbuilt (T-DV-34) and refused.

TERMINOLOGY: this is AIU 1.0 (RCUDD1A). The RNG state/recipe is format-neutral
here (32-bit words, not DL16/BF16).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "DeviceRngSwitch",
    "EgressMode",
    "RngDrawSpec",
    "RngLaunchPlan",
    "SetupStaging",
    "SpyreGeneratorState",
    "SpyreRngError",
    "SpyreRngResources",
    "StreamSlot",
    "StreamWord",
    "build_band_lxlu_stream",
    "build_rng_kat_binary",
    "build_rng_launch_plan",
    "device_rng_enabled",
    "draw_params_stream",
    "draw_raw_state",
    "epoch_key_2x32",
    "fold_key_schedule_2x32",
    "get_rng_resources",
    "injection_words_2x32",
    "key_schedule_2x32",
    "launch_raw_state_draw",
    "raw_words_to_uniform",
    "reconstruct_raw_state_from_tensor",
    "route_uniform_draw",
    "replay_band_stream_raw",
    "rng_pattern_registered",
    "seed_to_key",
    "stream_to_sticks",
    "threefry2x32_20",
]

# ---------------------------------------------------------------------------
# Constants — frozen with the contract (doc 43 §1, §3.1; RNG_KAT_ROUNDS owner).
# ---------------------------------------------------------------------------

#: Round count. Single owner in the backend is ``scheduler/ir.py::RNG_KAT_ROUNDS``;
#: this is the host mirror of the frozen T=1 contract, not an independent choice.
RNG_KAT_ROUNDS = 20
#: 32 lanes per stick (fp32/rfp32 word per lane). Frozen by the contract.
RNG_KAT_LANES = 32

#: threefry.h R_32x2_{0..7}_0, indexed by round mod 8.
ROTATIONS_2X32: tuple[int, ...] = (13, 15, 26, 6, 17, 29, 16, 24)
#: threefry.h SKEIN_KS_PARITY32.
SKEIN_KS_PARITY32 = 0x1BD11BDA

_EPOCH_BITS = 30
_EPOCH_BLOCKS = 1 << _EPOCH_BITS
_U32_MASK = 0xFFFFFFFF
#: ``0x4B000000`` == 2^23 as an rfp32 word; band halves are stored as B + h.
_B_VAL = float(1 << 23)


class SpyreRngError(NotImplementedError):
    """A device RNG request that is refused, typed, on the host.

    Subclasses ``NotImplementedError`` so it PROPAGATES through
    ``torch.compile`` as a hard failure rather than being swallowed — the same
    fail-loud contract softmax uses (``SoftmaxEnvelopeError``). A silent host
    substitute is never acceptable.
    """


class EgressMode(Enum):
    """Which epilogue the band recipe emits (mirrors the reference enum)."""

    UNIFORM = "uniform"  # 2 egress sticks, u = hi*2^-16 + floor(lo/256)*2^-24
    RAW_STATE = "raw_state"  # 4 egress sticks, raw x0/x1 halves (the KAT)


class SetupStaging(Enum):
    """Where the four band setup constants are staged (mirrors the reference)."""

    LXLU = "lxlu"  # witnessed form: setup constants streamed via LXLU
    INIT_FLITS = "init_flits"  # offline-only alternative (not used by the KAT)


class StreamSlot(Enum):
    """What each LXLU pop carries (mirrors the reference ``StreamSlot``).

    The slot labels are carried for parity/testing; only ``words`` feeds the
    emitted params stream. The order is what the contract pins.
    """

    COUNTER = "counter"
    X0_BASE_LO = "x0_base_lo"
    X0_BASE_HI = "x0_base_hi"
    X1_LO = "x1_lo"
    X1_HI = "x1_hi"
    INJ0_LO = "inj0_lo"
    INJ0_HI = "inj0_hi"
    ROT_SCALE = "rot_scale"
    INJ1_LO = "inj1_lo"
    INJ1_HI = "inj1_hi"
    BAND_ROT_MASK = "band_rot_mask"
    BAND_ROT_SC1 = "band_rot_sc1"
    BAND_ROT_SC2 = "band_rot_sc2"
    BAND_ROT_Q = "band_rot_q"
    BAND_EPI_M8 = "band_epi_m8"
    BAND_EPI_NEG9 = "band_epi_neg9"
    BAND_EPI_S32 = "band_epi_s32"
    BAND_SETUP_M16B = "band_setup_m16b"
    BAND_SETUP_MHI = "band_setup_mhi"
    BAND_SETUP_S16 = "band_setup_s16"
    BAND_SETUP_B = "band_setup_b"


#: The four band setup slot labels, in pop order. The constant WORDS are owned
#: by the codegen stream builder (``threefry_recipe_model``); only the slot
#: vocabulary is mirrored here for the local ``band_lxlu_pop_count``.
BAND_SETUP_SLOTS = (
    StreamSlot.BAND_SETUP_M16B,
    StreamSlot.BAND_SETUP_MHI,
    StreamSlot.BAND_SETUP_S16,
    StreamSlot.BAND_SETUP_B,
)
BAND_CORE_PROLOGUE_SLOTS = (
    StreamSlot.COUNTER,
    StreamSlot.X0_BASE_LO,
    StreamSlot.X0_BASE_HI,
    StreamSlot.X1_LO,
    StreamSlot.X1_HI,
)
BAND_PROLOGUE_SLOTS = BAND_SETUP_SLOTS + BAND_CORE_PROLOGUE_SLOTS
BAND_ROUND_SLOTS = (
    StreamSlot.INJ0_LO,
    StreamSlot.INJ0_HI,
    StreamSlot.BAND_ROT_MASK,
    StreamSlot.BAND_ROT_SC1,
    StreamSlot.BAND_ROT_MASK,
    StreamSlot.BAND_ROT_SC1,
    StreamSlot.BAND_ROT_SC2,
    StreamSlot.BAND_ROT_SC2,
    StreamSlot.BAND_ROT_Q,
    StreamSlot.BAND_ROT_Q,
    StreamSlot.INJ1_LO,
    StreamSlot.INJ1_HI,
)
BAND_EPILOGUE_SLOTS = (
    StreamSlot.INJ0_LO,
    StreamSlot.INJ0_HI,
    StreamSlot.BAND_EPI_M8,
    StreamSlot.BAND_EPI_NEG9,
    StreamSlot.BAND_EPI_S32,
    StreamSlot.BAND_EPI_M8,
    StreamSlot.BAND_EPI_NEG9,
    StreamSlot.BAND_EPI_S32,
)
BAND_EPILOGUE_SLOTS_RAW = (StreamSlot.INJ0_LO, StreamSlot.INJ0_HI)


def band_lxlu_pop_count(
    rounds: int,
    *,
    setup_staging: SetupStaging = SetupStaging.LXLU,
    egress: EgressMode = EgressMode.UNIFORM,
) -> int:
    """LXLU sticks per tile. 251 for RAW_STATE@20.

    LOCAL MIRROR, not the codegen owner: this is hand-derived from the
    ``BAND_*_SLOTS`` tuples above (the line 184 note), which mirror the codegen
    slot vocabulary. The codegen owner is
    ``sentient_codegen.emulator.threefry_recipe_model.band_lxlu_pop_count``;
    ``tests/test_rng_host.py::test_local_band_counts_mirror_the_codegen_owner``
    guards against drift when codegen is importable.
    """
    prologue = (
        BAND_SETUP_SLOTS + BAND_CORE_PROLOGUE_SLOTS
        if setup_staging is SetupStaging.LXLU
        else BAND_CORE_PROLOGUE_SLOTS
    )
    epilogue = (
        BAND_EPILOGUE_SLOTS if egress is EgressMode.UNIFORM else BAND_EPILOGUE_SLOTS_RAW
    )
    return len(prologue) + rounds * len(BAND_ROUND_SLOTS) + len(epilogue)


def band_egress_stick_count(
    rounds: int,
    *,
    setup_staging: SetupStaging = SetupStaging.LXLU,
    egress: EgressMode = EgressMode.UNIFORM,
) -> int:
    """The LX egress sticks the band recipe writes: 2 (uniform) / 4 (raw).

    LOCAL MIRROR of the codegen egress size, not the codegen owner
    (``sentient_codegen.emulator.threefry_recipe_model.band_egress_stick_count``);
    the same drift-guard test asserts parity when codegen is importable.
    """
    del rounds, setup_staging  # the egress size is a constant of the mode
    return 2 if egress is EgressMode.UNIFORM else 4


# ---------------------------------------------------------------------------
# Threefry2x32-20 (thin delegation to the codegen owner)
# ---------------------------------------------------------------------------


def _rotl32(x: np.ndarray, r: int) -> np.ndarray:
    """32-bit rotate-left used by the stream-replay oracle.

    The generator itself is owned by
    ``sentient_codegen.emulator.threefry_reference``; this helper survives only
    for :func:`replay_band_stream_raw`, which decodes a params stream back into
    the band arithmetic to check the bytes it carries.
    """
    r %= 32
    if r == 0:
        return x
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def threefry2x32_20(
    c0: np.ndarray, c1: np.ndarray, key: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """The canonical generator: Threefry2x32-20 of counter (c0, c1).

    Thin adapter over the single owner
    ``sentient_codegen.emulator.threefry_reference.threefry2x32_20``: it masks
    the key words to u32 (the historical host behaviour) and coerces the counter
    to uint32 arrays, then delegates every round to codegen. No round arithmetic
    lives here.
    """
    from sentient_codegen.emulator.threefry_reference import (
        threefry2x32_20 as _ref_threefry2x32_20,
    )

    k0, k1 = (int(w) & _U32_MASK for w in key)
    c0 = np.asarray(c0, dtype=np.uint32)
    c1 = np.asarray(c1, dtype=np.uint32)
    return _ref_threefry2x32_20(c0, c1, (k0, k1))


# ---------------------------------------------------------------------------
# Key schedule, the host fold, and the epoch key (doc 43 §3.1)
# ---------------------------------------------------------------------------


def seed_to_key(seed: int) -> tuple[int, int]:
    """``seed_key = (seed_lo, seed_hi)`` — the u64 seed split into two u32 words
    (doc 43 §3.1). A negative seed is taken modulo 2^64, matching torch's int64
    ``manual_seed`` argument width."""
    s = int(seed) & 0xFFFFFFFFFFFFFFFF
    return s & _U32_MASK, (s >> 32) & _U32_MASK


def key_schedule_2x32(key: tuple[int, int]) -> tuple[int, int, int]:
    """(ks0, ks1, ks2) with ks2 = C240 ^ k0 ^ k1 (threefry.h).

    Thin adapter over
    ``sentient_codegen.emulator.threefry_reference.key_schedule_2x32``; masks
    the key words to u32 for the historical host behaviour.
    """
    from sentient_codegen.emulator.threefry_reference import (
        key_schedule_2x32 as _ref_key_schedule_2x32,
    )

    k0, k1 = (int(w) & _U32_MASK for w in key)
    return _ref_key_schedule_2x32((k0, k1))


def injection_words_2x32(
    key: tuple[int, int], rounds: int
) -> tuple[tuple[int, int], ...]:
    """Per-injection (x0 word, x1 word) for injections s = 1 .. rounds // 4.

    Injection s follows round 4s-1: x0 += ks[s mod 3], x1 += ks[(s+1) mod 3] + s.
    Thin adapter over
    ``sentient_codegen.emulator.threefry_reference.injection_words_2x32``.
    """
    from sentient_codegen.emulator.threefry_reference import (
        injection_words_2x32 as _ref_injection_words_2x32,
    )

    k0, k1 = (int(w) & _U32_MASK for w in key)
    return _ref_injection_words_2x32((k0, k1), rounds)


def epoch_key_2x32(
    seed_key: tuple[int, int], epoch: int, *, device: int, stream: int
) -> tuple[int, int]:
    """``k' = Threefry2x32_20(seed_key, (epoch[31:0], device<<24|stream<<8|epoch[39:32]))``.

    HOST-ONLY (doc 43 §3.1): the device never sees the seed, only the folded
    key schedule. ``epoch`` is in blocks (``offset >> 30``).
    """
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")
    if not 0 <= device < 256:
        raise ValueError(f"device must fit in 8 bits, got {device}")
    if not 0 <= stream < 65536:
        raise ValueError(f"stream must fit in 16 bits, got {stream}")
    c0 = epoch & _U32_MASK
    c1 = ((device << 24) | (stream << 8) | ((epoch >> 32) & 0xFF)) & _U32_MASK
    k0, k1 = (int(w) & _U32_MASK for w in seed_key)
    x0, x1 = threefry2x32_20(
        np.array([c0], np.uint32), np.array([c1], np.uint32), (k0, k1)
    )
    return int(x0[0]), int(x1[0])


def fold_key_schedule_2x32(key: tuple[int, int], offset: int) -> tuple[int, int]:
    """Host fold of the key schedule into the input (doc 43 §3.1).

    Returns ``(x0_base, x1)`` with ``x0_base = ((offset mod 2^30) + ks0) mod 2^32``
    and ``x1 = ks1`` (a per-op constant). The counter is
    ``((offset mod 2^30) + block, 0)``, so the device prologue is one add32.
    """
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    ks0, ks1, _ = key_schedule_2x32(key)
    return ((offset % _EPOCH_BLOCKS) + ks0) & _U32_MASK, ks1


# ---------------------------------------------------------------------------
# The params stream — thin delegating wrapper over the codegen owner.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamWord:
    """One LXLU pop: a labeled stick of raw rfp32 words (one per lane)."""

    slot: StreamSlot
    words: np.ndarray = field(compare=False)


def _to_codegen_egress(egress: EgressMode):
    from sentient_codegen.emulator import threefry_recipe_model as _rm

    return _rm.EgressMode(egress.value)


def _to_codegen_setup_staging(setup_staging: SetupStaging):
    from sentient_codegen.emulator import threefry_recipe_model as _rm

    return _rm.SetupStaging(setup_staging.value)


def _from_codegen_stream(stream: Sequence) -> tuple[StreamWord, ...]:
    """Adapt codegen ``StreamWord`` pops into torch-spyre ``StreamWord`` pops.

    The slot vocabulary is matched by value (the two enums carry identical
    strings), so a new slot on the codegen side either round-trips or fails
    loud on the local ``StreamSlot(...)`` lookup — never a silent drop.
    """
    return tuple(
        StreamWord(StreamSlot(w.slot.value), np.asarray(w.words, dtype=np.uint32))
        for w in stream
    )


def _to_codegen_stream(stream: Sequence) -> tuple:
    """Adapt local ``StreamWord`` pops back into codegen ``StreamWord`` pops.

    The codegen patch owner (``patch_offset_words``/``offset_dependent_indices``)
    matches slots against the CODEGEN ``StreamSlot`` enum, so its input must be
    codegen ``StreamWord`` objects — a local slot (a different enum with equal
    values) compares unequal and would make the owner refuse an empty
    offset-dependent set. Slots are mapped by value; the words array carries
    through unchanged.
    """
    from sentient_codegen.emulator import threefry_recipe_model as _rm

    return tuple(
        _rm.StreamWord(
            _rm.StreamSlot(w.slot.value), np.asarray(w.words, dtype=np.uint32)
        )
        for w in stream
    )


def build_band_lxlu_stream(
    blocks: np.ndarray,
    *,
    x0_base: int,
    x1: int,
    rotations: Sequence[int],
    injections: Sequence[tuple[int, int]],
    rounds: int,
    setup_staging: SetupStaging = SetupStaging.LXLU,
    egress: EgressMode = EgressMode.RAW_STATE,
) -> tuple[StreamWord, ...]:
    """The host-side LXLU pops for one band tile, in recipe order.

    Thin delegating wrapper: the recipe is OWNED by
    ``sentient_codegen.emulator.threefry_recipe_model.build_band_lxlu_stream``.
    This function only translates the local ``SetupStaging`` / ``EgressMode``
    vocabulary into the codegen enums, calls the owner, and adapts the returned
    ``StreamWord`` tuple back into torch-spyre ``StreamWord`` objects with
    ``uint32`` word arrays. No recipe arithmetic is reimplemented here.

    ``blocks`` are the per-lane counter indices (< 2^23, exact in rfp32). T=1
    (RAW_STATE, the default here) yields exactly **251** sticks of 32 lanes:
    4 setup constants, COUNTER, 4 folded-key sticks, 20 rounds x 12, and the
    final x0-injection pair. A codegen refusal (e.g. the counter block-index
    limit) is re-raised as :class:`SpyreRngError` so every host refusal stays
    the one typed exception.
    """
    from sentient_codegen.emulator import threefry_recipe_model as _rm

    try:
        ref_stream = _rm.build_band_lxlu_stream(
            blocks,
            x0_base=x0_base,
            x1=x1,
            rotations=rotations,
            injections=injections,
            rounds=rounds,
            setup_staging=_to_codegen_setup_staging(setup_staging),
            egress=_to_codegen_egress(egress),
        )
    except _rm.RecipeModelError as exc:
        raise SpyreRngError(str(exc)) from exc
    return _from_codegen_stream(ref_stream)


def stream_to_sticks(stream: Sequence[StreamWord]) -> np.ndarray:
    """Stack a stream into the ``(sticks, lanes)`` uint32 params array.

    NOTE (transport, integration step): on silicon, fp32 words past a
    multi-stick tensor's FIRST stick are swizzled, so a host route that writes
    this array to HBM must write it in device word order. This function returns
    recipe order; the swizzle is the launcher's concern, not the builder's.
    """
    if not stream:
        raise ValueError("empty stream")
    return np.stack([np.asarray(w.words, dtype=np.uint32) for w in stream], axis=0)


def _decode_rfp32_word(word) -> float:
    """Raw rfp32 word (uint32) to its exact float value."""
    return float(np.array([int(word)], dtype=np.uint32).view(np.float32)[0])


def replay_band_stream_raw(
    stream: Sequence[StreamWord], *, rounds: int = RNG_KAT_ROUNDS
) -> tuple[np.ndarray, np.ndarray]:
    """Replay a RAW_STATE params stream through the band arithmetic on the host.

    Recovers the semantic constants the stream carries (per-lane blocks, the
    folded ``x0_base``/``x1``, the per-round injection words, and the final x0
    injection) and runs the folded Threefry2x32-``rounds`` — the same
    arithmetic the device recipe performs. Returns ``(x0, x1)`` uint32 arrays,
    one word per lane.

    This is the host oracle the params-stream builder is checked against: it
    proves the BYTES the builder emits encode the KAT's numbers and reconstruct
    the published vectors. It does NOT model the transport (LXLU swizzle) or
    the egress store — those are the launcher's concern.
    """
    b = _B_VAL
    # COUNTER stick: words are the rfp32 encoding of the (small) block indices.
    blocks = (
        np.asarray(stream[4].words, dtype=np.uint32).view(np.float32).astype(np.uint32)
    )
    # Folded x0_base: LO is 2^0 bits, HI is (hi - 128).
    lo = int(_decode_rfp32_word(stream[5].words[0]))
    hi = int(_decode_rfp32_word(stream[6].words[0])) + 128
    x0_base = ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)
    # Folded x1: LO is (kl + B), HI is (kh + B).
    kl = int(_decode_rfp32_word(stream[7].words[0]) - b)
    kh = int(_decode_rfp32_word(stream[8].words[0]) - b)
    x1_const = ((kh & 0xFFFF) << 16) | (kl & 0xFFFF)
    # Per-round injection words (INJ0: [lo-B, hi-B-128], INJ1: [lo+B, hi+B-128]).
    w0s: list[int] = []
    w1s: list[int] = []
    for r in range(rounds):
        base = 9 + r * 12
        i0l = int(_decode_rfp32_word(stream[base].words[0]) + b)
        i0h = int(_decode_rfp32_word(stream[base + 1].words[0]) + b + 128)
        i1l = int(_decode_rfp32_word(stream[base + 10].words[0]) - b)
        i1h = int(_decode_rfp32_word(stream[base + 11].words[0]) - b + 128)
        w0s.append(((i0h & 0xFFFF) << 16) | (i0l & 0xFFFF))
        w1s.append(((i1h & 0xFFFF) << 16) | (i1l & 0xFFFF))
    # Final x0 injection (the x0 part of injection s = rounds/4, which has no
    # round start to fold into): [fl, fh-128] with no +/- B.
    fl = int(_decode_rfp32_word(stream[-2].words[0]))
    fh = int(_decode_rfp32_word(stream[-1].words[0])) + 128
    final = ((fh & 0xFFFF) << 16) | (fl & 0xFFFF)

    with np.errstate(over="ignore"):
        x0 = blocks + np.uint32(x0_base)
        x1 = np.full_like(blocks, np.uint32(x1_const))
        for r in range(rounds):
            if r % 4 == 0 and r > 0:
                x0 = x0 + np.uint32(w0s[r])
            x0 = x0 + x1
            x1 = _rotl32(x1, ROTATIONS_2X32[r % 8]) ^ x0
            if r % 4 == 3:
                x1 = x1 + np.uint32(w1s[r])
        x0 = x0 + np.uint32(final)
    return x0, x1


def draw_params_stream(
    *,
    seed: int,
    offset: int = 0,
    blocks: np.ndarray | None = None,
    rounds: int = RNG_KAT_ROUNDS,
    device: int = 0,
    stream: int = 0,
    egress: EgressMode = EgressMode.RAW_STATE,
) -> tuple[StreamWord, ...]:
    """Derive the full params stream for one T=1 tile from ``(seed, offset)``.

    Applies the doc 43 §3.1 host chain: ``seed -> seed_key -> epoch key k' ->
    fold -> (x0_base, x1)`` plus ``injection_words_2x32(k', rounds)`` and
    ``rotations = ROTATIONS_2X32[r % 8]``. ``blocks`` defaults to the 32 lanes
    ``0..31`` for the single T=1 tile.
    """
    if blocks is None:
        blocks = np.arange(RNG_KAT_LANES, dtype=np.uint32)
    else:
        blocks = np.asarray(blocks, dtype=np.uint32)
    seed_key = seed_to_key(seed)
    epoch = offset >> _EPOCH_BITS
    k_prime = epoch_key_2x32(seed_key, epoch, device=device, stream=stream)
    x0_base, x1 = fold_key_schedule_2x32(k_prime, offset)
    injections = injection_words_2x32(k_prime, rounds)
    rotations = [ROTATIONS_2X32[r % 8] for r in range(rounds)]
    return build_band_lxlu_stream(
        blocks,
        x0_base=x0_base,
        x1=x1,
        rotations=rotations,
        injections=injections,
        rounds=rounds,
        egress=egress,
    )


# ---------------------------------------------------------------------------
# Generator state (doc 43 §3.2): one (seed, offset) per device.
# ---------------------------------------------------------------------------

_RNG_STATE_MAGIC = b"SPYRNGS1"  # exactly 8 bytes (struct field width)
_RNG_STATE_VERSION = 1
_RNG_STATE_STRUCT = struct.Struct("<8sIQQ")  # magic, version, seed, offset
#: torch's default CPU seed; matching it keeps an unseeded torch.spyre draw
#: consistent with the rest of torch before the user calls manual_seed.
_DEFAULT_SEED = 67280421310721


@dataclass
class SpyreGeneratorState:
    """A real ``(seed u64, offset u64)`` generator state (one per device).

    The offset is the block-offset the host folds into the next draw's key
    schedule (doc 43 §3.2). It advances on every device draw: ``route_uniform_draw``
    calls :meth:`reserve` for ``numel`` blocks before launching, so successive
    draws differ and a fixed seed reproduces from a reset offset. Reproducibility
    is carried by this ``(seed, offset)`` pair — ``manual_seed`` sets the seed
    and zeroes the offset, and ``get_rng_state``/``set_rng_state`` round-trip both.
    """

    seed: int = _DEFAULT_SEED
    offset: int = 0

    def manual_seed(self, seed: int) -> None:
        """Set the seed and reset the offset (torch ``manual_seed`` semantics)."""
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.offset = 0

    def reserve(self, blocks: int) -> int:
        """Reserve ``blocks`` block-indices and return the pre-reservation offset.

        Rounds up to the next 2^30-block epoch boundary rather than crossing it
        (doc 43 §3.2), so a draw never straddles a re-key.
        """
        if blocks < 0:
            raise ValueError(f"blocks must be non-negative, got {blocks}")
        start = self.offset
        # Never let one draw straddle an epoch: round start up if it would.
        if (start % _EPOCH_BLOCKS) + blocks > _EPOCH_BLOCKS:
            start = ((start // _EPOCH_BLOCKS) + 1) * _EPOCH_BLOCKS
        self.offset = start + blocks
        return start

    def to_tensor(self):
        """Serialize to a CPU ``torch.ByteTensor`` (``get_rng_state`` shape)."""
        import torch

        raw = _RNG_STATE_STRUCT.pack(
            _RNG_STATE_MAGIC, _RNG_STATE_VERSION, self.seed, self.offset
        )
        return torch.tensor(list(raw), dtype=torch.uint8)

    @classmethod
    def from_tensor(cls, state) -> "SpyreGeneratorState":
        """Restore from a ``torch.ByteTensor`` produced by :meth:`to_tensor`."""
        import torch

        if not isinstance(state, torch.Tensor):
            raise TypeError(
                f"set_rng_state: expected a torch.ByteTensor, got {type(state)!r}"
            )
        raw = bytes(int(b) & 0xFF for b in state.reshape(-1).tolist())
        if len(raw) != _RNG_STATE_STRUCT.size:
            raise ValueError(
                f"set_rng_state: expected {_RNG_STATE_STRUCT.size} bytes, got {len(raw)}"
            )
        magic, version, seed, offset = _RNG_STATE_STRUCT.unpack(raw)
        if magic != _RNG_STATE_MAGIC or version != _RNG_STATE_VERSION:
            raise ValueError("set_rng_state: not a spyre RNG state (bad magic/version)")
        return cls(seed=seed, offset=offset)

    def get_state(self):
        return self.to_tensor()

    def set_state(self, state) -> None:
        other = self.from_tensor(state)
        self.seed = other.seed
        self.offset = other.offset


# ---------------------------------------------------------------------------
# Residency + partition manager (Option 1 shared construction, brief §2.0/§7.2)
# ---------------------------------------------------------------------------
#
# The params stream is draw-invariant except three sticks: COUNTER,
# X0_BASE_LO, X0_BASE_HI. ``SpyreRngResources`` keeps ONE resident params buffer
# per device and, on an offset advance within the same seed+epoch, patches only
# those three sticks instead of rebuilding the whole 251-stick stream. A seed
# change (``manual_seed``/``set_rng_state``) or an epoch rollover re-keys the
# schedule and so forces a FULL rebuild.
#
# The DEVICE half (``ensure_device_params``) keeps a resident device tensor per
# device: first build uploads the full 251-stick (32 KB) params ONCE, an offset
# advance writes only the three offset-dependent rows (384 B) in place, and a
# seed change / epoch rollover re-uploads. That is the actual data-movement win
# (251 -> 3 rows); throughput is still launch/sync-bound and must be measured
# on the box, not inferred.
#
# The codegen owner of the patch recipe
# (``threefry_recipe_model.patch_offset_words``) and of the row indices
# (``threefry_recipe_model.offset_dependent_indices``) is landed by a PARALLEL
# lane, so both are INJECTED/resolved lazily. The class is unit-testable with a
# local patch function, a fake device bridge, and no device. When the codegen
# patch is not importable, an offset advance falls back to a full rebuild /
# full upload: byte-identical (correct), merely unoptimized, and MARKED via
# ``patch_source()``/``last_device_update()`` — never a silent fallback. That is
# a performance fallback, never a correctness one.

#: Blocks are rfp32-exact only below 2^23 (``build_band_lxlu_stream`` guard).
_EXACT_BLOCK_LIMIT = 1 << 23


@dataclass(frozen=True)
class RngDrawSpec:
    """The canonical host-side draw request (Option 4 later reuses this shape).

    ``seed_lo``/``seed_hi`` are the u64 seed split into two u32 words (the
    ``seed_to_key`` order). ``epoch`` is the block-epoch index
    (``offset >> 30``); ``base_counter`` is the offset WITHIN that epoch
    (``offset & (2^30 - 1)``), i.e. the counter base the draw folds in.
    """

    seed_lo: int
    seed_hi: int
    base_counter: int
    epoch: int

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("seed_lo", self.seed_lo, 1 << 32),
            ("seed_hi", self.seed_hi, 1 << 32),
        ):
            if not 0 <= int(value) < limit:
                raise ValueError(f"RngDrawSpec.{name} must fit in u32, got {value!r}")
        if not 0 <= int(self.epoch):
            raise ValueError(
                f"RngDrawSpec.epoch must be non-negative, got {self.epoch!r}"
            )
        if not 0 <= int(self.base_counter) < _EPOCH_BLOCKS:
            raise ValueError(
                "RngDrawSpec.base_counter must be an offset within one epoch "
                f"[0, 2^{_EPOCH_BITS}), got {self.base_counter!r}"
            )

    @classmethod
    def from_seed_offset(cls, seed: int, offset: int) -> "RngDrawSpec":
        """Build the spec from the u64 ``(seed, offset)`` generator state."""
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        seed_lo, seed_hi = seed_to_key(seed)
        return cls(
            seed_lo=seed_lo,
            seed_hi=seed_hi,
            base_counter=offset % _EPOCH_BLOCKS,
            epoch=offset // _EPOCH_BLOCKS,
        )


def _device_index(device) -> int:
    """Normalize ``None`` / int / ``torch.device`` into a device index."""
    if device is None:
        return 0
    if isinstance(device, int):
        return int(device)
    index = getattr(device, "index", None)
    if index is None:
        # e.g. torch.device("spyre") with no explicit index -> device 0.
        return 0
    return int(index)


def _default_build_stream(
    spec: RngDrawSpec,
    device_index: int,
    *,
    rounds: int,
    egress: EgressMode,
    stream: int,
) -> "tuple[StreamWord, ...]":
    """Full params stream for ``spec`` (the un-optimized baseline build).

    Mirrors :func:`draw_params_stream` exactly for the same ``(seed, offset)``:
    ``blocks = 0..lanes-1`` and ``x0_base = (base_counter + ks0) mod 2^32``, so a
    resident first build is byte-identical to the per-draw full upload path.
    """
    seed_key = (spec.seed_lo, spec.seed_hi)
    k_prime = epoch_key_2x32(seed_key, spec.epoch, device=device_index, stream=stream)
    x0_base, x1 = fold_key_schedule_2x32(k_prime, spec.base_counter)
    injections = injection_words_2x32(k_prime, rounds)
    rotations = [ROTATIONS_2X32[r % 8] for r in range(rounds)]
    blocks = np.arange(RNG_KAT_LANES, dtype=np.uint32)
    return build_band_lxlu_stream(
        blocks,
        x0_base=x0_base,
        x1=x1,
        rotations=rotations,
        injections=injections,
        rounds=rounds,
        egress=egress,
    )


def _load_codegen_patch_fn():
    """The codegen offset-patch owner, or ``None`` if that lane has not landed."""
    try:
        from sentient_codegen.emulator import threefry_recipe_model as _rm
    except Exception:  # backend absent, or the patch lane not merged yet
        return None
    return getattr(_rm, "patch_offset_words", None)


def _load_codegen_offset_indices():
    """The codegen offset-dependent-index owner, or ``None`` if unavailable."""
    try:
        from sentient_codegen.emulator import threefry_recipe_model as _rm
    except Exception:
        return None
    return getattr(_rm, "offset_dependent_indices", None)


class _TorchDeviceBridge:
    """The default device bridge: real ``torch`` tensors on the target device.

    The bridge is the ONLY place the host touches device memory, so it is both
    the real transport and the seam an offline test injects to COUNT bytes. Its
    three methods define the residency protocol:

    * ``upload(host_words, device) -> resident`` — allocate a resident tensor and
      copy the WHOLE ``(sticks, lanes)`` params array (the first-draw / rebuild
      cost: 251 sticks for RAW_STATE).
    * ``write_rows(resident, row_indices, host_rows)`` — write ONLY the named
      rows in place (the offset-advance cost: 3 sticks = 384 B).
    * ``to_host(resident) -> np.ndarray`` — mirror the device buffer back for
      verification.

    ``host_words``/``host_rows`` are ``uint32`` numpy arrays in recipe order;
    ``int32`` holds the same 32-bit words byte for byte.
    """

    def upload(self, host_words, device):
        import torch

        dev = torch.device("spyre") if device is None else device
        src = np.ascontiguousarray(host_words.view(np.int32))
        return torch.from_numpy(src).to(dev)

    def write_rows(self, resident, row_indices, host_rows):
        import torch

        rows = tuple(int(r) for r in row_indices)
        idx = torch.as_tensor(rows, dtype=torch.long, device=resident.device)
        src = np.ascontiguousarray(host_rows.view(np.int32))
        values = torch.from_numpy(src).to(resident.device)
        resident.index_copy_(0, idx, values)

    def to_host(self, resident):
        return resident.detach().to("cpu").numpy().view(np.uint32)


class _DeviceResident:
    """Bookkeeping for one device's resident params buffer.

    ``stream``/``params`` are the HOST mirror (recipe order + stacked sticks).
    ``device_tensor`` is the RESIDENT device buffer (an object owned by the
    injected device bridge; a ``torch.Tensor`` for the default bridge).
    ``device_reflects`` is the ``(seed, epoch, base_counter)`` the device buffer
    currently encodes — it lags the host mirror and is what the per-draw
    minimal write is derived from. ``patch_source`` records which patch owner
    advanced the host stream (``"codegen"`` / ``"injected"``) or ``None`` when
    the last update was a full rebuild (including the no-patch fallback).
    """

    __slots__ = (
        "seed",
        "epoch",
        "base_counter",
        "stream",
        "params",
        "device_tensor",
        "device_reflects",
        "patch_source",
        "last_device_update",
    )

    def __init__(self, seed, epoch, base_counter, stream, params):
        self.seed = seed
        self.epoch = epoch
        self.base_counter = base_counter
        self.stream = stream
        self.params = params
        self.device_tensor = None
        self.device_reflects = None
        self.patch_source = None
        self.last_device_update = None

    def reflected_key(self) -> tuple:
        """The ``(seed, epoch, base_counter)`` tuple this resident encodes."""
        return (self.seed, self.epoch, self.base_counter)


class SpyreRngResources:
    """Per-device residency + partition manager for the RNG params stream.

    ONE instance serves a process; its state is keyed by device index so two
    devices never alias. It owns two responsibilities (brief §2.0):

    * **Partition allocation** — :meth:`reserve` hands out pairwise-DISJOINT
      counter ranges as ``(epoch, base_counter)``; ranges never overlap, never
      cross the ``2^30`` epoch, and never produce a block index ``>= 2^23`` (the
      rfp32 exactness bound; a reservation that cannot fit is rolled to the next
      epoch, and a single reservation above ``2^23`` is refused).
    * **Resident params** — :meth:`ensure_params` builds the buffer ONCE, patches
      only the offset-dependent words on a same-seed offset advance, and does a
      FULL rebuild on a seed change / epoch rollover. The reflected
      ``(seed, epoch, base_counter)`` is cached and the invalidation is explicit.

    ``build_fn(spec, device_index) -> tuple[StreamWord, ...]`` and
    ``patch_fn(stream, *, blocks, x0_base) -> tuple[StreamWord, ...]`` are
    INJECTED so the class unit-tests without the codegen patch and without a
    device. The default ``patch_fn`` is the codegen owner, resolved lazily.

    :meth:`ensure_device_params` adds the DEVICE half: it keeps a resident
    device tensor per device and, on a same-seed offset advance, writes only the
    three offset-dependent rows (384 B) into it instead of re-uploading the
    whole 251-stick (32 KB) params tensor. The write goes through an injected
    ``device_bridge`` (default :class:`_TorchDeviceBridge`), which is also the
    seam an offline test uses to COUNT the bytes actually moved.
    """

    _DEFAULT = object()

    def __init__(
        self,
        *,
        build_fn=_DEFAULT,
        patch_fn=_DEFAULT,
        device_bridge=_DEFAULT,
        rounds: int = RNG_KAT_ROUNDS,
        egress: EgressMode = EgressMode.RAW_STATE,
        stream: int = 0,
    ) -> None:
        self._build_fn = build_fn
        self._patch_fn = patch_fn
        self._device_bridge = (
            _TorchDeviceBridge() if device_bridge is self._DEFAULT else device_bridge
        )
        self._rounds = int(rounds)
        self._egress = egress
        self._stream = int(stream)
        self._residents: dict[int, _DeviceResident] = {}
        # Partition allocator cursor: (epoch, next_counter). Starts at epoch 0.
        self._alloc_epoch = 0
        self._alloc_counter = 0
        # Lazy cache of the codegen offset-dependent row indices (may be None:
        # no owner). ``_offset_rows_resolved`` distinguishes "not computed yet"
        # from "computed and absent", so a missing owner is not re-probed.
        self._offset_rows_cache: "tuple[int, ...] | None" = None
        self._offset_rows_resolved = False

    # -- partition allocator -------------------------------------------------

    def reserve(self, n: int) -> "tuple[int, int]":
        """Reserve ``n`` disjoint block-counters; return ``(epoch, base_counter)``.

        The returned range is ``[base_counter, base_counter + n)`` within
        ``epoch``. Ranges handed out here are pairwise disjoint, never cross the
        ``2^30`` epoch boundary, and never contain a block index ``>= 2^23``.
        """
        n = int(n)
        if n < 0:
            raise ValueError(f"reserve(n) requires n >= 0, got {n}")
        if n == 0:
            return self._alloc_epoch, self._alloc_counter
        if n > _EXACT_BLOCK_LIMIT:
            raise SpyreRngError(
                f"reserve({n}): a single draw cannot exceed 2^23 = {_EXACT_BLOCK_LIMIT} "
                "block-counters (rfp32 exactness bound); split the draw."
            )
        # Roll to the next epoch rather than emit a block index past the limit.
        if self._alloc_counter + n > _EXACT_BLOCK_LIMIT:
            self._alloc_epoch += 1
            self._alloc_counter = 0
        epoch = self._alloc_epoch
        base = self._alloc_counter
        # Structural invariant: base+n stays inside [0, 2^23) < 2^30.
        assert base + n <= _EXACT_BLOCK_LIMIT <= _EPOCH_BLOCKS
        self._alloc_counter = base + n
        return epoch, base

    # -- resident params -----------------------------------------------------

    def ensure_params(self, spec: RngDrawSpec, device=None) -> np.ndarray:
        """Return the resident ``(sticks, lanes)`` params for ``spec`` on ``device``.

        Invalidation contract (brief §2.0):

        * first draw for a device -> full build, upload the resident buffer;
        * offset advance, same seed + epoch -> patch ONLY the offset-dependent
          words (falls back to a full rebuild if the codegen patch is absent);
        * seed change / epoch rollover -> FULL rebuild (the key-derived words
          change);
        * a different ``device`` index -> separate resources, never aliased.
        """
        if not isinstance(spec, RngDrawSpec):
            raise TypeError(f"ensure_params expects an RngDrawSpec, got {type(spec)!r}")
        return self._advance(_device_index(device), spec).params

    def ensure_device_params(self, spec: RngDrawSpec, device=None, *, bridge=None):
        """Return the RESIDENT DEVICE tensor for ``spec`` on ``device``.

        This is the Option 1 data-movement win: the params live in device memory
        once, and a same-seed + same-epoch offset advance writes ONLY the three
        offset-dependent rows (COUNTER, X0_BASE_LO, X0_BASE_HI) — 3 sticks =
        384 B — instead of re-uploading the full 251-stick (32 128 B) tensor.

        The minimal write is derived from ``device_reflects`` (what the device
        buffer currently encodes) vs the host's target spec, so it is correct
        even if ``ensure_params`` already advanced the host mirror:

        * first draw, seed change, epoch rollover, or no offset-index owner ->
          FULL upload (``bridge.upload``);
        * same seed + epoch, offset advanced -> ``bridge.write_rows`` of the
          offset-dependent rows only;
        * the device buffer already reflects ``spec`` -> nothing is written.

        ``bridge`` defaults to the per-instance device bridge; an offline test
        injects a counting fake to assert the exact bytes moved.
        """
        if not isinstance(spec, RngDrawSpec):
            raise TypeError(
                f"ensure_device_params expects an RngDrawSpec, got {type(spec)!r}"
            )
        dev = _device_index(device)
        res = self._advance(dev, spec)
        bridge = self._device_bridge if bridge is None else bridge
        target = res.reflected_key()

        if res.device_tensor is None:
            res.device_tensor = bridge.upload(res.params, device)
            res.last_device_update = "upload"
        elif res.device_reflects == target:
            # Replay-safe: the device buffer already encodes this spec.
            res.last_device_update = "cached"
            return res.device_tensor
        elif (
            res.device_reflects is not None
            and res.device_reflects[0] == res.seed
            and res.device_reflects[1] == res.epoch
        ):
            rows = self._offset_dependent_rows()
            if rows is None:
                # No offset-index owner (codegen absent): full re-upload keeps
                # this CORRECT, merely unoptimized. Marked via patch_source.
                res.device_tensor = bridge.upload(res.params, device)
                res.last_device_update = "upload"
            else:
                host_rows = np.ascontiguousarray(res.params[list(rows)])
                bridge.write_rows(res.device_tensor, rows, host_rows)
                res.last_device_update = "patch"
        else:
            # Seed change or epoch rollover relative to what the device holds:
            # the key-derived words change, so the whole buffer must be re-sent.
            res.device_tensor = bridge.upload(res.params, device)
            res.last_device_update = "upload"
        res.device_reflects = target
        return res.device_tensor

    # -- invalidation core ---------------------------------------------------

    def _advance(self, dev: int, spec: RngDrawSpec) -> _DeviceResident:
        """Advance the HOST resident for ``spec``; return the resident record.

        Implements the invalidation contract: a same-seed + same-epoch offset
        advance patches (when a patch owner is available), a seed change or an
        epoch rollover rebuilds fully, and a repeated spec is a no-op.
        """
        res = self._residents.get(dev)
        seed = (spec.seed_lo, spec.seed_hi)
        if res is None:
            return self._full_build(dev, spec)
        same_seed = res.seed == seed
        same_epoch = res.epoch == spec.epoch
        if same_seed and same_epoch and res.base_counter == spec.base_counter:
            # Replay-safe: nothing advances, return the cached buffer unchanged.
            return res
        if same_seed and same_epoch:
            patched = self._patch_offset(dev, res, spec)
            if patched is not None:
                return patched
            # Patch path unavailable (codegen lane not merged): rebuild fully.
            return self._full_build(dev, spec)
        # Seed change or epoch rollover: key-derived words change -> rebuild.
        return self._full_build(dev, spec)

    def _full_build(self, dev: int, spec: RngDrawSpec) -> _DeviceResident:
        stream = self._build(spec, dev)
        params = stream_to_sticks(stream)
        prev = self._residents.get(dev)
        # A full rebuild that keeps the SAME (seed, epoch) leaves the device
        # buffer valid for every key-derived row; carrying it lets the device
        # path still do the 3-row write (the seed/epoch key comparison in
        # ensure_device_params decides whether that is safe).
        carry_device = (
            prev is not None
            and prev.device_tensor is not None
            and prev.seed == (spec.seed_lo, spec.seed_hi)
            and prev.epoch == spec.epoch
        )
        res = _DeviceResident(
            seed=(spec.seed_lo, spec.seed_hi),
            epoch=spec.epoch,
            base_counter=spec.base_counter,
            stream=stream,
            params=params,
        )
        res.patch_source = None
        if carry_device and prev is not None:
            res.device_tensor = prev.device_tensor
            res.device_reflects = prev.device_reflects
        self._residents[dev] = res
        return res

    def _patch_offset(
        self, dev: int, res: _DeviceResident, spec: RngDrawSpec
    ) -> "_DeviceResident | None":
        patch_fn, source = self._resolve_patch()
        if patch_fn is None:
            return None
        seed_key = (spec.seed_lo, spec.seed_hi)
        k_prime = epoch_key_2x32(seed_key, spec.epoch, device=dev, stream=self._stream)
        x0_base, _x1 = fold_key_schedule_2x32(k_prime, spec.base_counter)
        blocks = np.arange(RNG_KAT_LANES, dtype=np.uint32)
        # The codegen owner keys its slot set on the CODEGEN StreamSlot enum, so
        # hand it codegen StreamWords (a local enum compares unequal there).
        new_codegen_stream = patch_fn(
            _to_codegen_stream(res.stream), blocks=blocks, x0_base=x0_base
        )
        new_stream = _from_codegen_stream(new_codegen_stream)
        new_params = stream_to_sticks(new_stream)
        patched_res = _DeviceResident(
            seed=seed_key,
            epoch=spec.epoch,
            base_counter=spec.base_counter,
            stream=new_stream,
            params=new_params,
        )
        patched_res.patch_source = source
        # Same (seed, epoch) as the prior resident by construction: carry the
        # resident device buffer so the next device draw writes only 3 rows.
        patched_res.device_tensor = res.device_tensor
        patched_res.device_reflects = res.device_reflects
        self._residents[dev] = patched_res
        return patched_res

    def _build(self, spec: RngDrawSpec, dev: int):
        if self._build_fn is self._DEFAULT:
            return _default_build_stream(
                spec,
                dev,
                rounds=self._rounds,
                egress=self._egress,
                stream=self._stream,
            )
        return self._build_fn(spec, dev)

    def _resolve_patch(self) -> "tuple[Callable[..., tuple] | None, str | None]":
        """The patch owner plus its source tag (``codegen`` / ``injected``)."""
        if self._patch_fn is self._DEFAULT:
            fn = _load_codegen_patch_fn()
            return fn, ("codegen" if fn is not None else None)
        if self._patch_fn is None:
            return None, None
        return self._patch_fn, "injected"

    def _resolve_patch_fn(self):
        """The patch owner alone (back-compat shim over :meth:`_resolve_patch`)."""
        return self._resolve_patch()[0]

    def _offset_dependent_rows(self) -> "tuple[int, ...] | None":
        """Row indices that depend on ``(blocks, x0_base)``, per codegen.

        Delegates to the codegen recipe owner ``offset_dependent_indices`` so
        the torch-spyre half never hardcodes the three row positions. Returns
        ``None`` when that owner is unavailable — the device path then does a
        full upload (correct, unoptimized), never a guessed row set.
        """
        if not self._offset_rows_resolved:
            self._offset_rows_cache = self._compute_offset_rows()
            self._offset_rows_resolved = True
        return self._offset_rows_cache

    def _compute_offset_rows(self) -> "tuple[int, ...] | None":
        fn = _load_codegen_offset_indices()
        if fn is None:
            return None
        try:
            return tuple(
                int(i)
                for i in fn(
                    rounds=self._rounds, egress=_to_codegen_egress(self._egress)
                )
            )
        except Exception:
            # A codegen-side refusal (e.g. an unsupported egress) is not a
            # reason to crash the host path: fall back to the full upload.
            return None

    # -- observability -------------------------------------------------------

    def has_device_resident(self, device=None) -> bool:
        """True iff a DEVICE params tensor already exists for ``device``.

        This is the residency-state query ``build_rng_launch_plan`` consults to
        decide whether the full params words can be skipped (design note §3.1):
        when a resident tensor exists the launcher's resident branch reads it,
        so a per-draw plan needs only the declared ABI. A cold cache returns
        ``False`` and the plan is built in full, byte-identical to before.
        """
        res = self._residents.get(_device_index(device))
        return res is not None and res.device_tensor is not None

    def stream_index(self) -> int:
        """The RNG stream every resident buffer in this instance is built with.

        Named ``stream_index`` (not ``stream``) to avoid confusion with
        ``_DeviceResident.stream``, which is the stream WORDS tuple. The
        resident invalidation key is (seed, epoch, base_counter) and carries no
        stream index, so a single instance can only honour one stream at a time;
        a caller needing per-stream residency creates one instance per stream.
        """
        return self._stream

    def reflected(self, device=None) -> "RngDrawSpec | None":
        """The ``(seed, epoch, base_counter)`` the resident buffer reflects."""
        res = self._residents.get(_device_index(device))
        if res is None:
            return None
        return RngDrawSpec(
            seed_lo=res.seed[0],
            seed_hi=res.seed[1],
            base_counter=res.base_counter,
            epoch=res.epoch,
        )

    def patch_source(self, device=None) -> "str | None":
        """Which patch owner last advanced ``device``'s params, or ``None``.

        ``"codegen"`` = the real ``threefry_recipe_model.patch_offset_words``;
        ``"injected"`` = a caller-supplied patch fn; ``None`` = the last update
        was a full rebuild (first draw, seed change, epoch rollover, or the
        no-patch fallback). This makes a silently-unoptimized fallback visible.
        """
        res = self._residents.get(_device_index(device))
        return None if res is None else res.patch_source

    def last_device_update(self, device=None) -> "str | None":
        """The last device-side write kind: ``upload`` / ``patch`` / ``cached``."""
        res = self._residents.get(_device_index(device))
        return None if res is None else res.last_device_update

    def offset_dependent_rows(self) -> "tuple[int, ...] | None":
        """Public view of the rows the device patch writes (``None`` if absent)."""
        return self._offset_dependent_rows()


#: The process-wide resources instance. Per-device state lives inside it, keyed
#: by device index — this is a process handle, NOT a bare module-global buffer.
_RNG_RESOURCES: "SpyreRngResources | None" = None


def get_rng_resources() -> SpyreRngResources:
    """Return the process-wide :class:`SpyreRngResources` (created on demand)."""
    global _RNG_RESOURCES
    if _RNG_RESOURCES is None:
        _RNG_RESOURCES = SpyreRngResources()
    return _RNG_RESOURCES


def reset_rng_resources() -> None:
    """Drop the process-wide resources (tests / device teardown).

    Not part of the public RNG API surface; exposed for deterministic test
    isolation so a stale resident buffer never leaks between cases.
    """
    global _RNG_RESOURCES
    _RNG_RESOURCES = None


# ---------------------------------------------------------------------------
# The route scaffold (torch-spyre half of "how the two lanes meet").
# ---------------------------------------------------------------------------

#: The switch that turns the device RNG route ON. Default OFF: with it off the
#: user-facing kernels stay the documented host-CPU offload. With it on, an
#: unwired device draw REFUSES TYPED instead of silently offloading.
_DEVICE_RNG_ENV = "TORCH_SPYRE_DEVICE_RNG"


class DeviceRngSwitch:
    """Reads the ``TORCH_SPYRE_DEVICE_RNG`` env switch (truthy: 1/true/yes/on)."""

    ENV = _DEVICE_RNG_ENV

    @staticmethod
    def enabled() -> bool:
        return os.environ.get(_DEVICE_RNG_ENV, "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )


def device_rng_enabled() -> bool:
    return DeviceRngSwitch.enabled()


def rng_pattern_registered() -> bool:
    """True iff the backend registered the ``rng_kat`` pattern.

    Lane B owns registration (``PATTERN_REGISTRY`` + factory + ``resolve_pattern``).
    This is a live registry check, so it flips to True only once the backend
    half has landed in the running environment. A missing backend (no
    ``sentient_codegen`` on PYTHONPATH) is NOT registered — the route then
    refuses typed rather than pretending.
    """
    try:
        from sentient_codegen.scheduler.matmul import PATTERN_REGISTRY
    except Exception:  # ImportError, or any backend import fault — be honest: unwired
        return False
    return "rng_kat" in PATTERN_REGISTRY


@dataclass(frozen=True)
class RngLaunchPlan:
    """The host-prepared T=1 launch: the params stream + the declared ABI.

    This is a PLAN. It carries everything the launcher needs (the 251-stick
    params array, the 2-segment ABI, the raw-egress reconstruction rule) but
    does NOT launch — that is the integration step, after lane B registers.

    A RESIDENT-ONLY plan (``params is None``) omits the words because they are
    already resident on the device: it still declares the full ABI so the
    launcher can validate the resident tensor's shape, and it refuses a forced
    full upload (``use_resident=False``) rather than launching an empty params.
    """

    #: (sticks, lanes) uint32 params array in recipe order (251 x 32 for raw).
    #: ``None`` on a RESIDENT-ONLY plan: the words are already resident on the
    #: device (``SpyreRngResources``), so the plan carries only the declared
    #: ABI (``sticks``/``lanes``/...) and the launcher reads the resident tensor.
    params: "np.ndarray | None"
    #: Egress mode this stream was built for (raw-state is the witnessed one).
    egress: EgressMode
    sticks: int
    lanes: int
    rounds: int
    #: XLAT segment layout: input -> segment 0, output -> segment 1.
    xlat_segments: dict
    declared_segment_count: int
    #: Raw-egress reconstruction: x0 = (x0h<<16)|x0l, x1 = (x1h<<16)|x1l.
    output_sticks: int
    #: The draw spec this plan was derived from (Option 1 residency). ``None`` on
    #: a hand-built plan; the launcher then falls back to the baked-in params.
    spec: "RngDrawSpec | None" = None
    #: The RNG stream this plan was built for. Carried so the launcher can hold
    #: the resident path honest: the resident buffer belongs to
    #: ``SpyreRngResources`` and is keyed by (seed, epoch, base_counter), which
    #: carries no stream index. A resident launch of a plan whose stream differs
    #: from the resources' stream would silently ship the resource's words, so
    #: the launcher refuses typed instead.
    stream: int = 0


def build_rng_launch_plan(
    *,
    seed: int,
    offset: int = 0,
    numel: int = RNG_KAT_LANES,
    device: int = 0,
    stream: int = 0,
    egress: EgressMode = EgressMode.RAW_STATE,
    resident: bool = False,
) -> RngLaunchPlan:
    """Build the T=1 params stream + ABI, refusing every unwired path, typed.

    Refusals (all ``SpyreRngError``, so they propagate as hard failures):

    * **pre-registration** — ``rng_kat`` is not in the backend pattern registry.
    * **T≥2** — a request covering more than one 32-lane tile is unbuilt
      (T-DV-34).

    Both the RAW_STATE and the (silicon-witnessed, P5) UNIFORM egress build a
    plan now; the launcher for each validates its own egress. The returned plan
    is still just a plan — the launcher is the integration step.

    ``resident=True`` (design note §3.1) skips the full params-stream build when
    a DEVICE resident buffer already exists for ``device``: the launcher takes
    the resident branch and reads that tensor, so materialising the 251 sticks
    here would only be discarded. Such a plan carries ``params=None`` but the
    same declared ABI (``sticks`` from the LOCAL ``band_lxlu_pop_count`` mirror
    — drift-guarded against the codegen owner — and ``lanes`` from
    ``RNG_KAT_LANES``, not hardcoded). On a COLD cache — no resident buffer —
    this falls back to the
    ordinary full build, so the default (and cold) path is byte-identical to
    the pre-residency behaviour. ``draw_raw_state`` opts in; every other caller
    keeps the default ``resident=False``.
    """
    if not isinstance(egress, EgressMode):
        raise TypeError(f"egress must be an EgressMode, got {egress!r}")
    # T>=2 guard: one tile is 32 lanes. More elements => T>=2 => unbuilt.
    tiles = (int(numel) + RNG_KAT_LANES - 1) // RNG_KAT_LANES
    if tiles >= 2:
        raise SpyreRngError(
            f"device RNG T>=2 is unbuilt (T-DV-34): this request covers {numel} "
            f"elements = {tiles} tiles; only the single 32-lane T=1 tile is "
            f"witnessed. Draw on the host, or split into independent T=1 draws."
        )
    # Pre-registration guard (lane B not landed yet).
    if not rng_pattern_registered():
        raise SpyreRngError(
            "device RNG is not wired yet: the backend pattern 'rng_kat' is not "
            "registered (lane B). No device draw is performed. The user-facing "
            "uniform_/normal_ kernels remain the documented host-CPU offload "
            "until registration + integration land."
        )
    spec = RngDrawSpec.from_seed_offset(seed, offset)
    if resident and get_rng_resources().has_device_resident(device):
        if stream != 0:
            # Silent-wrong guard: RngDrawSpec does NOT carry ``stream``, so the
            # resident invalidation key cannot distinguish stream 0 from stream
            # N. Skipping the build here would serve the resource's own stream-0
            # bytes for a stream-N request. Refuse typed instead.
            raise SpyreRngError(
                "resident plan construction cannot honour a nonzero stream: "
                "RngDrawSpec carries no stream index, so the resident key would "
                f"silently serve stream-0 params for a stream-{stream} request. "
                "A nonzero stream requires launching with use_resident=False "
                "(the launcher also refuses a resident launch whose plan.stream "
                "differs from the resources' stream), or a separate resources "
                "key per stream."
            )
        # Resident path: the launcher reads the device tensor, so build only the
        # declared ABI — no params words. Shape derives from the LOCAL
        # ``band_lxlu_pop_count`` mirror (== the stream length; drift-guarded
        # against the codegen owner in test_rng_host) and the lane constant.
        params = None
        sticks = band_lxlu_pop_count(RNG_KAT_ROUNDS, egress=egress)
        lanes = RNG_KAT_LANES
    else:
        params_stream = draw_params_stream(
            seed=seed,
            offset=offset,
            rounds=RNG_KAT_ROUNDS,
            device=device,
            stream=stream,
            egress=egress,
        )
        params = stream_to_sticks(params_stream)
        sticks = int(params.shape[0])
        lanes = int(params.shape[1])
    return RngLaunchPlan(
        params=params,
        egress=egress,
        sticks=sticks,
        lanes=lanes,
        rounds=RNG_KAT_ROUNDS,
        xlat_segments={"input": 0, "output": 1},
        declared_segment_count=2,
        output_sticks=band_egress_stick_count(RNG_KAT_ROUNDS, egress=egress),
        spec=spec,
        stream=int(stream),
    )


#: One T=1 launch reconstructs ``x0`` and ``x1`` — 32 lanes each, full 32-bit
#: words — so a single launch yields ``2 * RNG_KAT_LANES`` = 64 uniform words.
#: That bounds the T=1 device uniform fill: ``numel <= 64``.
_RNG_MAX_UNIFORM_ELEMS = 2 * RNG_KAT_LANES


def raw_words_to_uniform(words, from_: float = 0.0, to: float = 1.0) -> np.ndarray:
    """Map full 32-bit random words to ``[from_, to)`` (the P0 uniform grid).

    ``u = (word >> 8) * 2**-24`` (doc 43 P0 / the C-5 fp32 grid): the top 24
    bits of each word, scaled into the unit interval, then affinely scaled to
    ``[from_, to)``. This is a PURE/DETERMINISTIC function of the raw bits — it
    is exercised offline on known words. It is NOT the silicon-witnessed uniform
    *epilogue* (the ``u = hi*2^-16 + floor(lo/256)*2^-24`` grid, P5 owed): it
    rebuilds uniform values from the witnessed RAW-STATE egress on the host.
    """
    w = np.asarray(words, dtype=np.uint32)
    u = (w >> np.uint32(8)).astype(np.float64) * (2.0**-24)
    return (float(from_) + u * (float(to) - float(from_))).astype(np.float32)


def _generator_state():
    """The live per-device ``SpyreGeneratorState`` from the ``torch.spyre`` module.

    Returns ``None`` if the device module is not registered (e.g. the backend
    half imported standalone). The default generator (``device=None``) is the
    one ``torch.spyre.manual_seed`` seeds and ``get_rng_state`` round-trips, so
    a draw here advances the state the user can reproduce/reset.
    """
    try:
        import torch
    except Exception:
        return None
    mod = getattr(torch, "spyre", None)
    impl = getattr(mod, "_impl", None) if mod is not None else None
    if impl is None or not hasattr(impl, "_rng_state"):
        return None
    try:
        return impl._rng_state(None)
    except Exception:
        return None


def route_uniform_draw(
    from_: float, to: float, numel: int, *, device=None
) -> np.ndarray:
    """Device route for a user-facing uniform draw: fill from DEVICE bits.

    Called by the ``uniform_``/``rand`` kernels only when the device-RNG switch
    is ON. Returns a host ``float32`` array of ``numel`` uniforms in
    ``[from_, to)`` whose *random bits* come from the silicon-witnessed T=1
    raw-state kernel (``draw_raw_state``), never from a host RNG.

    Design:

    * **Reserve** ``numel`` blocks from the device generator so successive
      draws differ (offset advances) and a fixed seed reproduces (offset reset
      by ``manual_seed``/``set_rng_state``). Graph replay must therefore
      advance the offset across replays — see ``SpyreGeneratorState.reserve``.
    * **Generate on device** via ``draw_raw_state(seed, offset)`` (one T=1
      launch yields 64 full 32-bit words: ``x0`` then ``x1``). At most ``64``
      elements are served per draw.
    * **Map on the host** with :func:`raw_words_to_uniform`
      (``u=(w>>8)*2**-24`` then affine scale). The elementwise bit-ops it needs
      (``>>``/``&``) are NOT registered on spyre (only add/sub/mul/div/pow/...
      are), so the map cannot run on device today; the random *bits* are
      device-generated regardless, never from a host RNG.

    Refuses typed (``SpyreRngError``) when ``numel`` exceeds one T=1 tile
    (T-DV-34 unbuilt) or when ``rng_kat`` is not registered. There is
    deliberately no host-RNG fallback: with the switch ON the caller asked for
    device RNG, so a silent host substitute is forbidden.
    """
    numel = int(numel)
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}")
    if numel > _RNG_MAX_UNIFORM_ELEMS:
        raise SpyreRngError(
            f"device RNG T>=2 is unbuilt (T-DV-34): {numel} elements exceed the "
            f"{_RNG_MAX_UNIFORM_ELEMS} words a single T=1 launch yields (one "
            f"32-lane tile). Draw on the host, or split into independent draws."
        )
    if numel == 0:
        return np.empty(0, dtype=np.float32)
    if not rng_pattern_registered():
        raise SpyreRngError(
            "device RNG is not wired yet: backend pattern 'rng_kat' is not "
            "registered (lane B). Turn off TORCH_SPYRE_DEVICE_RNG to use the "
            "documented host-CPU offload, or wait for registration."
        )
    state = _generator_state()
    if state is None:
        raise SpyreRngError(
            "device RNG: the spyre generator state is unreachable (torch.spyre "
            "not registered); cannot reserve a draw offset. Refusing rather than "
            "falling back to a host draw."
        )
    # Reserve one block per element so a second draw advances the offset.
    offset = state.reserve(numel)
    x0, x1 = draw_raw_state(state.seed, offset, device=device)
    words = np.concatenate([np.asarray(x0, np.uint32), np.asarray(x1, np.uint32)])[
        :numel
    ]
    return raw_words_to_uniform(words, from_, to)


# ---------------------------------------------------------------------------
# Device launch (the integration step): compile + launch the registered
# rng_kat T=1 raw-state kernel, then reconstruct the full 32-bit state.
# ---------------------------------------------------------------------------
#
# This is the seam that connects the host half above to the backend half (lane
# B). It mirrors ``torch_spyre/ops/softmax.py``'s plan-validating launch: the
# binary is compiled from the registered ``rng_kat`` pattern, an XLAT
# relocation plan is derived from the SAME scheduled program, the positional
# bindings are validated against it, and only then is the C++ launch issued.
# The 2-segment ABI is the frozen contract's: segment 0 = params, segment 1 =
# raw-state output.
#
# HONESTY: ``launch_raw_state_draw`` needs a real ``spyre`` device and the flex
# runtime (``torch_spyre._C``); it is NOT exercised on a host. The offline proof
# that the two halves agree is the emulator full-transport replay in
# ``tests/test_rng_kat_integration.py`` (lane A's params through lane B's
# kernel -> the three published anchors bit-exact), plus the connected-compile
# byte match against the silicon-witnessed ``e697b657…``.

#: Cached ``(binary, xlat_plan)`` for the fixed T=1 raw-state KAT. The kernel is
#: shape-invariant (one fixed silicon-witnessed cell), so it is compiled once.
_RAW_KAT_CACHE: "tuple[bytes, object] | None" = None


#: Targets with a silicon-witnessed device RNG kernel. RNG is 1p0-only today
#: (the witnessed binary e697b657 is a rcudd1a emission); 1p5 has no RNG
#: contract. A non-witnessed target REFUSES rather than emit a wrong-generation
#: binary, and the compile target + stick geometry come from the SAME generation.
_RNG_WITNESSED_TARGETS = frozenset({"rcudd1a"})


def _stick_bytes(target: str) -> int:
    """One stick = 32 lanes x 4 bytes = 128 on rcudd1a (gen-owned, not typed)."""
    from sentient_codegen.gen.registry import get_generation

    return int(get_generation(target).hw.stick_bytes)


def build_rng_kat_binary(target: "str | None" = None) -> "tuple[bytes, object]":
    """Compile (once) the registered ``rng_kat`` T=1 raw-state binary + plan.

    The target is resolved from the generation registry (``get_default_target()``)
    unless passed explicitly, and threaded through the compile so the target and
    the stick geometry share one generation. Returns ``(binary, xlat_plan)``; for
    a rcudd1a target the binary is the silicon-witnessed ``e697b657…`` (4352 B).
    The XLAT plan is derived from the SAME scheduled program (T-LR-2B). Raises
    ``SpyreRngError`` on a target with no witnessed RNG kernel (RNG is 1p0-only),
    or if the backend has not registered ``rng_kat`` (lane B not landed).
    """
    global _RAW_KAT_CACHE
    from sentient_codegen.gen.registry import get_default_target

    tgt = get_default_target() if target is None else target
    if tgt not in _RNG_WITNESSED_TARGETS:
        raise SpyreRngError(
            f"device RNG has no witnessed kernel for target {tgt!r}; it is 1p0 "
            f"({sorted(_RNG_WITNESSED_TARGETS)})-only (1p5 RNG is unbuilt)"
        )
    if _RAW_KAT_CACHE is not None:
        return _RAW_KAT_CACHE
    if not rng_pattern_registered():
        raise SpyreRngError(
            "device RNG is not wired: backend pattern 'rng_kat' is not registered"
        )
    from sentient_codegen.backend.launch_abi import build_xlat_relocation_plan
    from sentient_codegen.backend.ttir_to_schedule import compile_scheduled_program
    from sentient_codegen.emulator import threefry_recipe_model as _rm
    from sentient_codegen.scheduler.engine import (
        TilingConfig,
        finalize_pattern_program,
        resolve_pattern,
    )

    out_sticks = _rm.band_egress_stick_count(
        RNG_KAT_ROUNDS, egress=_rm.EgressMode.RAW_STATE
    )
    tiling = TilingConfig(
        M=out_sticks,
        K=1,
        N=RNG_KAT_LANES,
        tile_m=out_sticks,
        tile_k=1,
        tile_n=RNG_KAT_LANES,
        num_cores=1,
        element_bits=32,
    )
    pattern = resolve_pattern("rng_kat", tiling=tiling, target=tgt)
    scheduled = finalize_pattern_program(pattern, tiling, tgt)
    binary = compile_scheduled_program(scheduled, target=tgt)
    sb = _stick_bytes(tgt)
    params_bytes = band_lxlu_pop_count(RNG_KAT_ROUNDS, egress=EgressMode.RAW_STATE) * sb
    out_bytes = out_sticks * sb
    xlat_plan = build_xlat_relocation_plan(
        scheduled,
        target=tgt,
        binary=binary,
        num_cores=1,
        symbol_extents={0: params_bytes, 1: out_bytes},
    )
    _RAW_KAT_CACHE = (binary, xlat_plan)
    return _RAW_KAT_CACHE


#: Cached ``(binary, xlat_plan)`` for the fixed T=1 UNIFORM-egress KAT. The
#: binary is the silicon-witnessed ``d0930bd5…`` (receipt ``rng-p5-uniform-kat``);
#: the plan declares the output symbol's fp32 dtype/layout so the T-LR-7 gate
#: fires on the caller tensor (design 53 Q3.2). Distinct from the raw cache.
_UNIFORM_KAT_CACHE: "tuple[bytes, object] | None" = None


def build_rng_kat_uniform_binary(target: "str | None" = None) -> "tuple[bytes, object]":
    """Compile (once) the registered ``rng_kat`` T=1 UNIFORM binary + plan.

    The UNIFORM analogue of :func:`build_rng_kat_binary` (2 egress sticks, 257
    params sticks, the one-stick egress stride seed). For a rcudd1a target the
    binary is the silicon-witnessed ``d0930bd5…`` (4608 B, receipt
    ``rng-p5-uniform-kat``). The XLAT plan is derived from the SAME scheduled
    program AND declares the output symbol (segment 1) as ``f32``/``IEEE_FP32``
    /32-bit/contiguous (T-LR-7) — the raw build passes no ``symbol_dtypes`` and
    so leaves that gate unfired; the uniform caller-tensor binding needs it
    armed (design 53 Q3.2). Raises ``SpyreRngError`` on a non-witnessed target
    or if the backend has not registered ``rng_kat``.
    """
    global _UNIFORM_KAT_CACHE
    from sentient_codegen.gen.registry import get_default_target, get_generation

    tgt = get_default_target() if target is None else target
    if tgt not in _RNG_WITNESSED_TARGETS:
        raise SpyreRngError(
            f"device RNG has no witnessed kernel for target {tgt!r}; it is 1p0 "
            f"({sorted(_RNG_WITNESSED_TARGETS)})-only (1p5 RNG is unbuilt)"
        )
    if _UNIFORM_KAT_CACHE is not None:
        return _UNIFORM_KAT_CACHE
    if not rng_pattern_registered():
        raise SpyreRngError(
            "device RNG is not wired: backend pattern 'rng_kat' is not registered"
        )
    from sentient_codegen.backend.launch_abi import build_xlat_relocation_plan
    from sentient_codegen.backend.ttir_to_schedule import compile_scheduled_program
    from sentient_codegen.emulator import threefry_recipe_model as _rm
    from sentient_codegen.scheduler.engine import (
        TilingConfig,
        finalize_pattern_program,
    )
    from sentient_codegen.scheduler.rng import rng_kat_pattern_registered

    out_sticks = _rm.band_egress_stick_count(
        RNG_KAT_ROUNDS, egress=_rm.EgressMode.UNIFORM
    )
    tiling = TilingConfig(
        M=out_sticks,
        K=1,
        N=RNG_KAT_LANES,
        tile_m=out_sticks,
        tile_k=1,
        tile_n=RNG_KAT_LANES,
        num_cores=1,
        element_bits=32,
    )
    # The registered factory with egress=UNIFORM — NOT resolve_pattern, whose
    # two-arg registry hop defaults to RAW_STATE. This is the same route the
    # codegen half admitted (binary byte-identical to the offline constructor).
    pattern = rng_kat_pattern_registered(
        tiling=tiling, target=tgt, egress=_rm.EgressMode.UNIFORM
    )
    scheduled = finalize_pattern_program(pattern, tiling, tgt)
    binary = compile_scheduled_program(scheduled, target=tgt)
    sb = _stick_bytes(tgt)
    params_bytes = band_lxlu_pop_count(RNG_KAT_ROUNDS, egress=EgressMode.UNIFORM) * sb
    out_bytes = out_sticks * sb
    # Declare the fp32 output symbol so the T-LR-7 dtype/layout gate fires on
    # the caller tensor. DERIVED from the generation's own dtype map, not typed.
    _f32 = get_generation(tgt).resolve_device_dtype("f32")
    symbol_dtypes = {
        1: {
            "logical_dtype": _f32.logical_dtype,
            "device_dtype": _f32.device_dtype,
            "device_element_bits": _f32.device_bits,
            "contiguous_stick_layout": True,
        }
    }
    xlat_plan = build_xlat_relocation_plan(
        scheduled,
        target=tgt,
        binary=binary,
        num_cores=1,
        symbol_extents={0: params_bytes, 1: out_bytes},
        symbol_dtypes=symbol_dtypes,
    )
    _UNIFORM_KAT_CACHE = (binary, xlat_plan)
    return _UNIFORM_KAT_CACHE


def _resolve_launch_params(plan: "RngLaunchPlan", dev, use_resident):
    """Return the device params tensor for ``plan`` (resident or full upload).

    Shared by the raw and uniform launchers so the resident-buffer contract
    (stream identity, declared-shape check, resident-only refusal) is one
    implementation, not two copies that can drift.
    """
    if use_resident is not False and plan.spec is not None:
        # Resident DEVICE tensor: no per-draw full re-materialization.
        resources = get_rng_resources()
        # The resident buffer is built with the RESOURCE's own stream, and the
        # resident invalidation key (seed, epoch, base_counter) carries no
        # stream index. A plan built for a different stream would silently ship
        # the resource's words and discard ``plan.params`` -- refuse typed.
        if plan.stream != resources.stream_index():
            raise SpyreRngError(
                f"resident launch cannot honour stream {plan.stream}: this "
                f"device's resident params buffer is built with the resource's "
                f"stream {resources.stream_index()}, and the resident key "
                "(seed, epoch, base_counter) carries no stream index. Remedy: "
                "launch with use_resident=False to use this plan's own "
                f"stream-{plan.stream} params, or keep a separate "
                "SpyreRngResources per stream (the multi-generator direction)."
            )
        params = resources.ensure_device_params(plan.spec, dev)
        # Validate against the plan's DECLARED ABI (sticks, lanes), not the
        # words: a resident-only plan (``params is None``) declares them from
        # the local band-count mirror (drift-guarded vs the codegen owner) so
        # the shape contract still holds.
        if params.shape != (plan.sticks, plan.lanes):
            raise SpyreRngError(
                "resident params shape "
                f"{tuple(params.shape)} disagrees with the plan's declared "
                f"{(plan.sticks, plan.lanes)}; refusing a mid-stream device buffer"
            )
        return params
    if plan.params is None:
        raise SpyreRngError(
            "this launch plan is resident-only (no params words were built); "
            "use_resident=False needs the full params array. Rebuild with "
            "build_rng_launch_plan(..., resident=False)."
        )
    import torch

    # Recipe order (the launcher's device-word-order / swizzle concern, not
    # the builder's). int32 holds the same 32-bit words, byte for byte.
    return torch.from_numpy(np.ascontiguousarray(plan.params.view(np.int32))).to(dev)


def launch_raw_state_draw(plan: RngLaunchPlan, *, device=None, use_resident=None):
    """Launch the registered T=1 raw-state KAT for ``plan``; return its output.

    Requires a real ``spyre`` device and the flex runtime. Returns the 4-stick
    raw egress tensor, shape ``(4, 32)``; feed it to
    :func:`reconstruct_raw_state_from_tensor` to recover ``(x0, x1)``. The
    positional binding is ``[params, out]`` — exactly the frozen 2-segment ABI
    (segment 0 = params, segment 1 = raw-state output). The launch is gated by
    ``validate_positional_launch`` against the plan derived beside the binary, so
    a permuted or truncated binding refuses before the C++ call (T-LR-2B).

    Option 1 wiring: when ``use_resident`` is not ``False`` and the plan carries
    a :class:`RngDrawSpec`, the launch uses the device's RESIDENT params tensor
    (:meth:`SpyreRngResources.ensure_device_params`) — the whole 251-stick
    (32 KB) params tensor is uploaded ONCE, and a same-seed offset advance then
    writes only the three offset-dependent rows (384 B) into it. The bytes are
    identical to the per-draw full-upload path, so this never changes the
    draw's result. ``use_resident=False`` forces the baked-in ``plan.params``
    and the old per-draw full upload (the pre-Option-1 behavior).

    A resident-only plan (``plan.params is None``, built with
    ``build_rng_launch_plan(..., resident=True)``) still validates the resident
    tensor against the plan's DECLARED ``(sticks, lanes)``; forcing
    ``use_resident=False`` on it refuses typed rather than launching an empty
    params tensor.

    STREAM IDENTITY: the resident buffer is built with the RESOURCE's stream,
    and its invalidation key (seed, epoch, base_counter) carries no stream
    index. When the resident branch is taken and ``plan.stream`` differs from
    :meth:`SpyreRngResources.stream_index`, this refuses ``SpyreRngError``
    rather than silently launching the resource's words and discarding
    ``plan.params``. ``stream == 0`` (every production caller) is unaffected.
    """
    import torch

    from torch_spyre._C import launch_kernel_from_bytes

    if plan.egress is not EgressMode.RAW_STATE:
        raise SpyreRngError(
            "launch_raw_state_draw only launches the raw-state KAT egress; got "
            f"{plan.egress!r}"
        )
    binary, xlat_plan = build_rng_kat_binary()
    dev = torch.device("spyre") if device is None else device
    params = _resolve_launch_params(plan, dev, use_resident)
    out = torch.zeros((plan.output_sticks, plan.lanes), dtype=torch.int32, device=dev)
    tensors_by_symbol = {
        xlat_plan.symbols[0].symbol_id: params,
        xlat_plan.symbols[1].symbol_id: out,
    }
    specs = {
        sym.symbol_id: (
            int(
                tensors_by_symbol[sym.symbol_id].element_size()
                * tensors_by_symbol[sym.symbol_id].nelement()
            ),
            "input" if sym.access_mode == "read" else "output",
        )
        for sym in xlat_plan.symbols
    }
    from sentient_codegen.backend.launch_abi import validate_positional_launch

    validate_positional_launch(xlat_plan, binary, specs)
    launch_kernel_from_bytes(
        binary, [tensors_by_symbol[s.symbol_id] for s in xlat_plan.symbols]
    )
    return out


def _device_stick_layout(tensor):
    """``tuple`` of the tensor's device stick layout, or ``None`` if unknown.

    The device-side layout (``[sticks, elems_per_stick]``) decides whether a
    caller tensor's allocation is exactly the 2 dense sticks the uniform egress
    writes. Unknown (host / ``_C`` absent) returns ``None`` so the caller takes
    the always-safe scratch path rather than guessing a direct binding.
    """
    try:
        from torch_spyre import _C

        return tuple(int(v) for v in _C.get_spyre_tensor_layout(tensor).device_size)
    except Exception:
        return None


def _select_uniform_binding(
    *,
    numel: int,
    from_: float,
    to: float,
    is_float32: bool,
    is_contiguous: bool,
    device_size,
) -> str:
    """Decide how to serve a switch-ON ``uniform_()`` fill (doc 54 §4.1).

    Returns one of:

    * ``"empty"`` — ``numel == 0`` (nothing to write; a no-op).
    * ``"direct"`` — bind the caller tensor itself as segment 1, ZERO host
      round trip. Requires the identity map (``[0,1)``), float32, contiguous,
      and a device layout of exactly the 2 dense sticks ``(2, 32)`` — a 1-D
      ``33 <= numel <= 64``. This is the silicon-witnessed binding, witnessed at
      BOTH ENDS as its own cells (numel=33 33/33 and numel=64 64/64 bit-exact
      vs the host map, 4 seeds each). The launcher reports the PHYSICAL
      stick-layout extent (256 B), so 33..63 — logical 132..252 B but a 256 B
      two-stick allocation — passes the ABI window gate.
    * ``"fallback"`` — take the EXISTING host-map path (device bits ->
      ``raw_words_to_uniform`` -> ``self.copy_``), unchanged from today. This is
      the answer for EVERY non-direct caller, with no new refusal (doc 54 §4.1
      last rows, §5):

        - a non-identity ``(from_, to)`` — no on-device affine kernel; the host
          map applies it;
        - non-fp32 — ``spyre__uniform_`` casts on the host today; keep it;
        - non-contiguous / a non-``(2,32)`` device layout (e.g. a 2-D ``(8,8)``
          is EIGHT sticks) — keep today's path;
        - ``1 <= numel <= 32`` — a ONE-stick (128 B) allocation, which cannot
          host the 2-stick store; the CHOOSER takes the host path (not a gate-8
          refusal — the scratch/copy that would serve it is unavailable).

      The scratch path (a ``(2,32)`` device scratch + one ``copy_from_d2d``) is
      UNAVAILABLE for f32 on 1p0: ``copy_from_d2d`` is an Inductor custom op
      lowered by ``lowering.mutate_to(dst, src)`` to a compiled pointwise copy
      kernel (NOT a raw memcpy), and an f32 device→device copy is refused by the
      pointer-grain gate (``SENTIENT-POINTER-ELEMENT-GRAIN-COMPUTE-ONLY-POINTEE``
      — the f32 pointwise-copy route is unbuilt). Established by execution on
      the box 2026-11-19: full ``copy_from_d2d``, full ``copy_`` and sliced
      ``copy_`` all refuse for f32. Doc 54 §3 pre-authorises this fallback:
      option B "is the correct first cut if the scratch path proves
      unavailable".

    Raises typed (``SpyreRngError``) ONLY for ``numel > 64`` (T-DV-34, one T=1
    tile) — every other case is a device path or the unchanged host path, never
    a new refusal (doc 54 §4.1 / §5).
    """
    if numel == 0:
        return "empty"
    if numel > _RNG_MAX_UNIFORM_ELEMS:
        raise SpyreRngError(
            f"device RNG T>=2 is unbuilt (T-DV-34): {numel} elements exceed the "
            f"{_RNG_MAX_UNIFORM_ELEMS} words a single T=1 launch yields (one "
            "32-lane tile producing u0,u1). Draw on the host, or split into "
            "independent draws."
        )
    if float(from_) != 0.0 or float(to) != 1.0:
        return "fallback"
    if not is_float32:
        return "fallback"
    if not is_contiguous:
        return "fallback"
    # Direct bind for the whole (2,32) two-stick cell (1-D 33..64). A 33..63
    # caller is allocation-safe: its device allocation is two sticks (256 B),
    # and the launcher reports that PHYSICAL stick-layout extent (not the
    # logical numel*4), so the launch-ABI window gate accepts the 2-stick write
    # — silicon-witnessed for numel=33 (33/33 bit-exact, stream identical to the
    # host map) and numel=64 (64/64), the two ENDS of the cell, each its own
    # witness. 1..32 is a ONE-stick (128 B) allocation: it cannot host the
    # 2-stick store, so the CHOOSER sends it to the host path (not a gate-8
    # refusal — the scratch/copy that would serve it is unavailable, see below).
    if device_size == (2, RNG_KAT_LANES):
        return "direct"
    return "fallback"


def launch_uniform_draw(
    plan: RngLaunchPlan, *, out=None, device=None, use_resident=None
):
    """Launch the registered T=1 UNIFORM KAT for ``plan`` into ``out``.

    Requires a real ``spyre`` device and the flex runtime. Binds the caller
    tensor ``out`` as segment 1 per :func:`_select_uniform_binding` — the
    silicon-witnessed DIRECT 2-stick binding (33<=numel<=64, dense (2,32)) — with
    the plan's fp32 output symbol declared (T-LR-7), so there is ZERO host
    round trip of the generated floats. A caller that would need the scratch +
    device→device copy refuses typed there (f32 d2d copy is unbuilt on 1p0).
    When ``out`` is ``None`` it writes the full ``(2,32)`` egress into a fresh
    device tensor and returns it (no copy needed — the caller owns it).

    The kernel egress is the bare ``[0,1)`` fp32 grid ``u = (w>>8)*2**-24``;
    affine scaling is NOT applied here (no on-device affine kernel), so the
    caller (``route_uniform_fill``) only routes the identity ``[0,1)`` map.
    """
    import torch

    from torch_spyre._C import launch_kernel_from_bytes

    if plan.egress is not EgressMode.UNIFORM:
        raise SpyreRngError(
            "launch_uniform_draw only launches the UNIFORM KAT egress; got "
            f"{plan.egress!r}"
        )
    binary, xlat_plan = build_rng_kat_uniform_binary()
    dev = torch.device("spyre") if device is None else device
    params = _resolve_launch_params(plan, dev, use_resident)

    if out is None:
        target = torch.empty(
            (plan.output_sticks, plan.lanes), dtype=torch.float32, device=dev
        )
    else:
        if out.device.type != "spyre":
            raise SpyreRngError(
                "launch_uniform_draw requires a spyre-device caller tensor; got "
                f"device {out.device!r}. Draw on the host with the switch off."
            )
        # This is the device launcher: it serves only the DIRECT binding (the
        # identity map). route_uniform_fill routes every other caller to the
        # host path before reaching here.
        decision = _select_uniform_binding(
            numel=int(out.numel()),
            from_=0.0,
            to=1.0,
            is_float32=(out.dtype == torch.float32),
            is_contiguous=out.is_contiguous(),
            device_size=_device_stick_layout(out),
        )
        if decision == "empty":
            return out
        if decision != "direct":
            raise SpyreRngError(
                "launch_uniform_draw serves only the direct (2,32) float32 "
                f"binding; caller (numel={out.numel()}) must use the host path "
                "(route_uniform_fill returns False for it). Internal routing bug."
            )
        target = out

    tensors_by_symbol = {
        xlat_plan.symbols[0].symbol_id: params,
        xlat_plan.symbols[1].symbol_id: target,
    }

    def _window_bytes(tensor) -> int:
        """The tensor's PHYSICAL allocation bytes (the stick-layout footprint).

        A direct-bound 33..63 fp32 caller is logically ``numel*4`` bytes (e.g. a
        33-element tensor is 132 B) but PHYSICALLY two 128-byte sticks = 256 B
        (``_C.get_spyre_tensor_layout().device_size == [2,32]``). The launch-ABI
        window gate (gate 8) compares the window against the binary's
        stick-aligned 2-stick write AND requires the window to be the ALLOCATION
        size — the stick-layout bytes, not the logical nbytes. So report the
        physical stick footprint (``max(logical, prod(device_size)*elem)``).
        Confirmed by execution: logical 132 B raised
        SENTIENT-LAUNCH-ABI-MISALIGNED-WINDOW; physical 256 B binds and the
        stream matches the host map (numel=33, 4 seeds). Falls back to logical
        bytes when the layout is unknown (the ``out is None`` scratch case)."""
        logical = int(tensor.element_size() * tensor.nelement())
        ds = _device_stick_layout(tensor)
        if ds:
            prod = 1
            for v in ds:
                prod *= int(v)
            return max(logical, prod * int(tensor.element_size()))
        return logical

    specs = {
        sym.symbol_id: (
            _window_bytes(tensors_by_symbol[sym.symbol_id]),
            "input" if sym.access_mode == "read" else "output",
        )
        for sym in xlat_plan.symbols
    }
    from sentient_codegen.backend.launch_abi import (
        validate_positional_launch,
        validate_tensor_dtype_and_layout,
    )

    validate_positional_launch(xlat_plan, binary, specs)
    # Arm the T-LR-7 dtype/layout gate: the plan declared segment 1 as fp32
    # (IEEE_FP32, contiguous). A wrong-dtype or mis-stickified caller tensor
    # refuses here rather than being bit-reinterpreted on device.
    validate_tensor_dtype_and_layout(xlat_plan, tensors_by_symbol)
    launch_kernel_from_bytes(
        binary, [tensors_by_symbol[s.symbol_id] for s in xlat_plan.symbols]
    )
    return target


def route_uniform_fill(self, from_=0.0, to=1.0) -> bool:
    """Device route for a switch-ON ``uniform_`` fill (doc 54 §4.1).

    Returns ``True`` when the fill was performed ON DEVICE (zero host round
    trip, direct-bind): the caller tensor is float32, contiguous, its device
    layout is exactly ``(2,32)`` (a 1-D ``33 <= numel <= 64``), and the map is
    the identity ``[0,1)``. Returns ``False`` when the caller must take the
    EXISTING host-map path — every other case (non-identity ``(from_, to)``,
    non-fp32, non-contiguous / non-``(2,32)``, and ``numel <= 32`` whose scratch
    path is f32-blocked). See :func:`_select_uniform_binding`.

    Raises typed (``SpyreRngError``) ONLY for ``numel > 64`` (T-DV-34). For the
    direct path the device generator must be reachable (registration + state),
    exactly as today's ``route_uniform_draw`` requires; a direct phone with the
    generator unreachable raises rather than silently host-drawing.

    Reserves ``numel`` block-counters from the device generator (successive
    draws differ, a fixed seed reproduces), draws ONE T=1 counter block and
    writes it into ``self`` via :func:`launch_uniform_draw`. The params stream
    is rebuilt in full per direct draw (the resident buffer is keyed for the raw
    egress); a uniform residency is a follow-on.
    """
    import torch

    numel = int(self.numel())
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}")
    action = _select_uniform_binding(
        numel=numel,
        from_=from_,
        to=to,
        is_float32=(self.dtype == torch.float32),
        is_contiguous=self.is_contiguous(),
        device_size=_device_stick_layout(self),
    )
    if action == "empty":
        return True
    if action == "fallback":
        # Not a refusal: the caller takes today's host-map path (device bits ->
        # host map -> self.copy_), which applies affine bounds / casts / strided.
        return False
    # action == "direct": a device draw. Same generator/registry guards the
    # host path enforces, but raised (not silently host-drawn) for a device call.
    if not rng_pattern_registered():
        raise SpyreRngError(
            "device RNG is not wired yet: backend pattern 'rng_kat' is not "
            "registered. Turn off TORCH_SPYRE_DEVICE_RNG to use the documented "
            "host-CPU offload."
        )
    state = _generator_state()
    if state is None:
        raise SpyreRngError(
            "device RNG: the spyre generator state is unreachable (torch.spyre "
            "not registered); cannot reserve a draw offset. Refusing rather than "
            "falling back to a host draw."
        )
    # Reserve numel block-counters so a second draw advances the offset.
    offset = state.reserve(numel)
    plan = build_rng_launch_plan(
        seed=state.seed,
        offset=offset,
        # ONE counter block = 32 lanes; a single launch yields u0,u1 = 64 words.
        numel=RNG_KAT_LANES,
        device=_device_index(self.device),
        egress=EgressMode.UNIFORM,
        # Full params build (not the raw-keyed resident buffer); the values
        # themselves never round-trip through the host.
        resident=False,
    )
    launch_uniform_draw(plan, out=self, device=self.device, use_resident=False)
    return True


def reconstruct_raw_state_from_tensor(out) -> "tuple[np.ndarray, np.ndarray]":
    """Recover full 32-bit ``(x0, x1)`` from the 4-stick raw egress tensor.

    The frozen reconstruction: each egress word MUST be a canonical band word
    ``0x4B000000 | half`` (``half < 2^16``); then ``half = word & 0xFFFF``,
    ``x0 = (x0h<<16)|x0l``, ``x1 = (x1h<<16)|x1l``. A non-canonical word
    refuses rather than silently returning a truncated state.
    """
    words = np.asarray(out.detach().to("cpu").numpy(), dtype=np.uint32)
    if words.ndim != 2 or words.shape[0] != 4:
        raise SpyreRngError(
            "raw-state egress must be 4 sticks (x0h, x0l, x1h, x1l); got shape "
            f"{words.shape}"
        )
    halves = []
    for i, row in enumerate(words):
        if ((row & np.uint32(0xFF000000)) != np.uint32(0x4B000000)).any():
            raise SpyreRngError(
                f"raw egress stick {i} is not a canonical band word "
                f"(0x4B000000|half); the device did not write raw state"
            )
        halves.append((row & np.uint32(0xFFFF)).astype(np.uint64))
    x0h, x0l, x1h, x1l = halves
    return (
        ((x0h << np.uint64(16)) | x0l).astype(np.uint32),
        ((x1h << np.uint64(16)) | x1l).astype(np.uint32),
    )


def draw_raw_state(seed: int, offset: int = 0, *, device=None):
    """Connected T=1 draw: build the plan, launch it, return ``(x0, x1)``.

    This is the whole wired path for the one witnessed cell. It needs a real
    device; the host-side derivation and the device kernel are proven to agree
    offline (``tests/test_rng_kat_integration.py``). T>=2 / the uniform fp32
    grid still refuse typed (DV-34 / P5 unbuilt) inside ``build_rng_launch_plan``.
    """
    plan = build_rng_launch_plan(
        seed=seed,
        offset=offset,
        device=_device_index(device),
        egress=EgressMode.RAW_STATE,
        # The launcher reads the device-resident params when one exists, so do
        # not materialise the words here (design note §3.1); a cold cache still
        # builds them in full inside the builder.
        resident=True,
    )
    return reconstruct_raw_state_from_tensor(launch_raw_state_draw(plan, device=device))
