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
from typing import Sequence

import numpy as np

__all__ = [
    "DeviceRngSwitch",
    "EgressMode",
    "RngLaunchPlan",
    "SetupStaging",
    "SpyreGeneratorState",
    "SpyreRngError",
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
    "injection_words_2x32",
    "key_schedule_2x32",
    "launch_raw_state_draw",
    "reconstruct_raw_state_from_tensor",
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
    """LXLU sticks per tile (the reference count law). 251 for RAW_STATE@20."""
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
    """The LX egress sticks the band recipe writes: 2 (uniform) / 4 (raw)."""
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
    schedule (doc 43 §3.2). It advances by the reservation of a launched draw;
    because the device path is not yet connected, no draw here advances it —
    what is real today is the seed, the round-trip, and the ability to reserve.
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
    """

    #: (sticks, lanes) uint32 params array in recipe order (251 x 32 for raw).
    params: np.ndarray
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


def build_rng_launch_plan(
    *,
    seed: int,
    offset: int = 0,
    numel: int = RNG_KAT_LANES,
    device: int = 0,
    stream: int = 0,
    egress: EgressMode = EgressMode.RAW_STATE,
) -> RngLaunchPlan:
    """Build the T=1 params stream + ABI, refusing every unwired path, typed.

    Refusals (all ``SpyreRngError``, so they propagate as hard failures):

    * **pre-registration** — ``rng_kat`` is not in the backend pattern registry
      (lane B has not landed). This is the expected state today.
    * **uniform fp32 grid** — the user-facing uniform output is UNWITNESSED
      (P5 owed); only the raw-state KAT egress is witnessed.
    * **T≥2** — a request covering more than one 32-lane tile is unbuilt
      (T-DV-34).

    When all guards pass (only possible after lane B registers AND for the
    raw-state egress), the returned plan is still just a plan — the launcher is
    the integration step.
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
    # Uniform grid guard: unwitnessed (P5 owed), never claimed.
    if egress is EgressMode.UNIFORM:
        raise SpyreRngError(
            "the uniform fp32 grid u = hi*2^-16 + floor(lo/256)*2^-24 is "
            "UNWITNESSED (P5 owed). Only the raw-state KAT egress (4 sticks, "
            "x0/x1 halves) is silicon-witnessed. Refusing rather than emit an "
            "unwitnessed user-facing stream."
        )
    # Pre-registration guard (lane B not landed yet).
    if not rng_pattern_registered():
        raise SpyreRngError(
            "device RNG is not wired yet: the backend pattern 'rng_kat' is not "
            "registered (lane B). No device draw is performed. The user-facing "
            "uniform_/normal_ kernels remain the documented host-CPU offload "
            "until registration + integration land."
        )
    params_stream = draw_params_stream(
        seed=seed,
        offset=offset,
        rounds=RNG_KAT_ROUNDS,
        device=device,
        stream=stream,
        egress=egress,
    )
    params = stream_to_sticks(params_stream)
    return RngLaunchPlan(
        params=params,
        egress=egress,
        sticks=int(params.shape[0]),
        lanes=int(params.shape[1]),
        rounds=RNG_KAT_ROUNDS,
        xlat_segments={"input": 0, "output": 1},
        declared_segment_count=2,
        output_sticks=band_egress_stick_count(RNG_KAT_ROUNDS, egress=egress),
    )


def route_uniform_draw(from_: float, to: float, numel: int) -> None:
    """The device route for a user-facing uniform draw — REFUSES, typed.

    Called by the ``uniform_``/``rand`` kernels only when the device-RNG switch
    is ON. Every path here refuses:

    * T≥2 (more than 32 elements) is unbuilt (T-DV-34);
    * the uniform fp32 grid is unwitnessed (P5 owed);
    * and, even for a single T=1 tile, ``rng_kat`` must be registered first.

    There is deliberately no fallback: with the switch ON, a caller asked for
    device RNG, so silently substituting a host draw would be the exact
    silent-mislead the contract forbids.
    """
    tiles = (int(numel) + RNG_KAT_LANES - 1) // RNG_KAT_LANES
    if tiles >= 2:
        raise SpyreRngError(
            f"device RNG T>=2 is unbuilt (T-DV-34): {numel} elements = {tiles} "
            f"tiles; only the single 32-lane T=1 tile is witnessed."
        )
    if (float(from_), float(to)) != (0.0, 1.0):
        raise SpyreRngError(
            f"uniform_({from_}, {to}): the device uniform grid is fixed at [0, 1); "
            f"arbitrary from_/to_ scaling is unwitnessed. Refusing."
        )
    if not rng_pattern_registered():
        raise SpyreRngError(
            "device RNG is not wired yet: backend pattern 'rng_kat' is not "
            "registered (lane B). Turn off TORCH_SPYRE_DEVICE_RNG to use the "
            "documented host-CPU offload, or wait for registration + integration."
        )
    # Registered, but the uniform grid itself is still unwitnessed (P5).
    raise SpyreRngError(
        "device RNG is registered but the uniform fp32 grid is UNWITNESSED "
        "(P5 owed): only the raw-state KAT egress is witnessed. A user-facing "
        "uniform draw is not yet connectable."
    )


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


def launch_raw_state_draw(plan: RngLaunchPlan, *, device=None):
    """Launch the registered T=1 raw-state KAT for ``plan``; return its output.

    Requires a real ``spyre`` device and the flex runtime. Returns the 4-stick
    raw egress tensor, shape ``(4, 32)``; feed it to
    :func:`reconstruct_raw_state_from_tensor` to recover ``(x0, x1)``. The
    positional binding is ``[params, out]`` — exactly the frozen 2-segment ABI
    (segment 0 = params, segment 1 = raw-state output). The launch is gated by
    ``validate_positional_launch`` against the plan derived beside the binary, so
    a permuted or truncated binding refuses before the C++ call (T-LR-2B).
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
    # Recipe order (the launcher's device-word-order / swizzle concern, not the
    # builder's). int32 holds the same 32-bit words, byte for byte.
    params = torch.from_numpy(np.ascontiguousarray(plan.params.view(np.int32))).to(dev)
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
    plan = build_rng_launch_plan(seed=seed, offset=offset, egress=EgressMode.RAW_STATE)
    return reconstruct_raw_state_from_tensor(launch_raw_state_draw(plan, device=device))
