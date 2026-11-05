# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSEs/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""bf16 -> DL16 numeric conversion through the fp32 pivot (AIU 1.0).

On Sentient 1.0 a bf16 operand must reach the device as DL16 (the custom
1-6-9 float). bf16 is E8M7; DL16 is 1-6-9 — the splits differ, so the
bits cannot be reinterpreted. This module is the conversion contract:

* ``bf16_to_dl16_words`` — the H2D-side encoder: exact bf16->fp32 upcast
  (bit shift, no rounding) then fp32->DL16 encoding with refuse-loud
  overflow and a flushed-element count on underflow.
* ``f32_to_dl16_word`` / ``dl16_word_to_f32`` — the pivot encoder and
  the output-path readback decoder.

WIRING STATUS (gauntlet r1 P3-1, 2026-11-05): this module is NOT yet
wired into the H2D path — it has zero production consumers today; the
live bf16-origin H2D path still fails loudly at DT_ERROR. The wiring
(the C++ seam branch: copyAsync/generate_dci, or deeptools ConvertElem
BFLOAT16->SEN169_FP16) is a box-gated lane item, and it carries a
wiring-time obligation: the seam must consult the KB registry's
generation-keyed delete-key (conversion_pivot / require_h2d_conversion
_pivot) — or refuse on its absence — never assume the 1p0 pivot. Until
then this module is the conversion CONTRACT (tests + comparison
oracle), not the shipped conversion.

Binding constraints (T-DC-1 S3, lead-adjudicated 2026-11-05):

* The fp32 pivot is EXCLUSIVE: fp32->DL16 semantics mirror the pinned
  deeptools converter ``FloatToFp16BinInternal``
  (util/sen_data_convert.cpp:213-244 at KB pin e7b8ce6f) — including its
  guard-bit rounding (round-up when the first dropped mantissa bit is
  set, capped so the fraction never carries into the exponent). The
  defective ``Ieeef16BinToFp16bin`` (T-DC-13) is never used.
* Every boundary constant below is DERIVED from the 1-6-9 format
  parameters (6-bit exponent, bias 31, 9-bit mantissa, no zero pattern,
  no subnormals) — the same store-grid model as the KB emulator's
  ``dl16_round.py`` encodable range (raw exponent in [-30, 32]) and the
  converter's own flush threshold (``exponent < -30`` -> signed zero).
  Nothing here is a hand-written constant.

Boundary policy (wiki/decisions/bf16-dl16-underflow-policy, USER-ruled
2026-11-05): overflow REFUSES with a typed error; underflow flushes to
signed zero with the flushed-element count surfaced in the result (the
launch-receipt hook). ``SENTIENT_BF16_DL16_SATURATE=1`` (read per call)
switches overflow from refuse to saturate-to-DL16-max with the same
counted receipt. Default OFF.

Non-finite inputs refuse (boundary="non-finite"): DL16 defines no
inf/nan word (the store-grid model refuses them, T-DV-17 precedent),
and the pinned C++ encoder's field-63 "INF/NaN" marker decodes as a
large FINITE value on readback — propagating it would be a silent
corruption, exactly the defect class this module exists to kill.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field

__all__ = [
    "SentientDtypeError",
    "Bf16Dl16RangeError",
    "ConversionResult",
    "ComparisonResult",
    "DL16_MIN_NORMAL",
    "DL16_MAX",
    "DL16_MAX_WORD",
    "DL16_MAX_NEXT_UP_BF16",
    "PIVOT_TOLERANCE",
    "bf16_to_dl16_words",
    "compare_dl16_readback",
    "f32_to_dl16_word",
    "dl16_word_to_f32",
]

# ---------------------------------------------------------------------------
# Format parameters and DERIVED boundaries (1-6-9: sign 1, exponent 6,
# fraction 9; biased exponent has no zero and no subnormal interpretation).
# ---------------------------------------------------------------------------

_EXP_BITS = 6
_MANT_BITS = 9
_EXP_BIAS = 31

#: Smallest positive normal: biased field 1 -> raw exponent 1 - 31.
DL16_MIN_NORMAL = math.ldexp(1.0, 1 - _EXP_BIAS)  # 2**-30
#: Largest finite value: biased field 63 (all ones), fraction all ones.
DL16_MAX = math.ldexp(
    2.0 - 2.0 ** -_MANT_BITS, (2 ** _EXP_BITS - 1) - _EXP_BIAS
)  # (2 - 2**-9) * 2**32, silicon-witnessed class
#: The DL16 word encoding DL16_MAX (sign 0 | field 63 << 9 | 0x1FF).
DL16_MAX_WORD = (
    ((2 ** _EXP_BITS - 1) << 9) | (2 ** _MANT_BITS - 1)
)  # 0x7FFF

#: fp32->DL16 relative rounding tolerance: the mantissa hop 23 -> 9
#: bounds the round-trip error by one DL16 ulp (2**-9 relative).
PIVOT_TOLERANCE = 2.0 ** -_MANT_BITS

#: Env flag switching overflow from refuse to saturate-to-DL16-max.
_SATURATE_ENV = "SENTIENT_BF16_DL16_SATURATE"


def _bits_f32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _bf16_decode_word(word: int) -> float:
    """Decode a bf16 word (upper half of an fp32 pattern) exactly."""
    return _f32_from_bits(word << 16)


def _next_bf16_above(value: float) -> float:
    """Smallest bf16 grid point strictly greater than ``value`` (value > 0).

    Truncating the fp32 pattern to its upper 16 bits rounds a positive
    value DOWN to the bf16 grid, so the next pattern up is the first bf16
    above it — including the mantissa-carry case, where the increment
    naturally bumps the exponent.
    """
    bits = _bits_f32(value)
    word = bits >> 16
    return _bf16_decode_word(word + 1)


#: Smallest bf16 value that cannot be represented in DL16 (the first bf16
#: grid point above DL16_MAX). Derived, never asserted.
DL16_MAX_NEXT_UP_BF16 = _next_bf16_above(DL16_MAX)


# ---------------------------------------------------------------------------
# Typed error surface
# ---------------------------------------------------------------------------


class SentientDtypeError(ValueError):
    """Base class for dtype-representation refusals on the Spyre backend."""


class Bf16Dl16RangeError(SentientDtypeError):
    """A bf16 value cannot be represented in DL16 on 1.0.

    Fields: symbol_id (the tensor symbol at the conversion site),
    boundary ("max" | "subnormal-underflow" | "non-finite"), value (the
    offending value), linear_index (its position in the flattened input).
    """

    def __init__(self, symbol_id, boundary, value, linear_index):
        self.symbol_id = symbol_id
        self.boundary = boundary
        self.value = value
        self.linear_index = linear_index
        super().__init__(
            f"symbol {symbol_id!r}: bf16 value {value!r} at linear index "
            f"{linear_index} crosses the DL16 {boundary} boundary "
            f"(DL16_MAX={DL16_MAX!r}, DL16_MIN_NORMAL={DL16_MIN_NORMAL!r})"
        )


# ---------------------------------------------------------------------------
# The pivot encoder / decoder
# ---------------------------------------------------------------------------


def f32_to_dl16_word(value: float) -> int:
    """Encode one fp32 value as a DL16 word (the pivot's second hop).

    Semantics mirror the pinned ``FloatToFp16BinInternal`` except where
    this module's refuse-loud policy replaces its saturating cap:

    * non-finite input -> ``Bf16Dl16RangeError`` (the C++ encodes a
      field-63 marker that decodes as a large finite value — a silent
      corruption we refuse instead);
    * raw exponent < -30 -> signed zero (the converter's flush threshold,
      checked on the RAW exponent pre-rounding, matching the C++ and the
      store-grid model's bottom-boundary asymmetry);
    * |value| > DL16_MAX -> ``Bf16Dl16RangeError`` (the C++ silently
      caps; we refuse). The comparison uses the derived DL16_MAX, so a
      value that rounds DOWN onto the max word from below is in range.
    """
    f32 = _bits_f32(value)
    sign = (f32 >> 16) & 0x8000
    fp32_exp = (f32 >> 23) & 0xFF
    exponent = fp32_exp - 127
    fraction = f32 & 0x7FFFFF

    if fp32_exp == 0xFF:
        # NaN or Inf: no DL16 word exists (see module docstring).
        raise Bf16Dl16RangeError("<pivot>", "non-finite", value, 0)
    if fp32_exp == 0:
        # fp32 subnormal/zero: every nonzero subnormal is below the DL16
        # floor (2**-126 class << 2**-30) -> signed zero.
        return sign
    if exponent < 1 - _EXP_BIAS:
        # Below the DL16 floor -> flush to signed zero (pre-rounding,
        # mirroring the converter).
        return sign
    if value > DL16_MAX or value < -DL16_MAX:
        raise Bf16Dl16RangeError("<pivot>", "max", value, 0)

    # Guard-bit rounding, exactly FloatToFp16BinInternal: round up when
    # the first dropped fraction bit is set, unless the kept fraction is
    # already all ones (the cap that prevents a carry into the exponent).
    rounding = 0
    if ((fraction >> (23 - _MANT_BITS - 1)) & 1) and (
        (fraction >> (23 - _MANT_BITS)) != (2 ** _MANT_BITS - 1)
    ):
        rounding = 1
    frac9 = (fraction >> (23 - _MANT_BITS)) + rounding

    return sign | ((exponent + _EXP_BIAS) << 9) | frac9


def dl16_word_to_f32(word: int) -> float:
    """Decode a DL16 word as fp32 (the output-path mirror).

    Mirrors the pinned ``Fp16BinToFloatInternal``
    (util/sen_data_convert.cpp:249-262 at KB pin e7b8ce6f): sign [15],
    exponent [14:9] bias 31, fraction [8:0]; field 0 with fraction 0 is
    zero; there is no subnormal interpretation. Note the decoder maps
    field 63 to raw exponent 32 (finite), so the max word 0x7FFF decodes
    to DL16_MAX — the "INF marker" the C++ encoder writes for fp32
    inf/nan decodes here as a large finite value (documented quirk;
    this module refuses non-finite inputs so it never emits it).
    """
    v = word & 0xFFFF
    s = (v & 0x8000) >> 15
    x = (v & 0x7E00) >> 9
    y = v & 0x01FF
    if x == 0 and y == 0:
        return 0.0
    i = (s << 31) | (((x - 31 + 127) & 0xFF) << 23) | (y << (23 - 9))
    return _f32_from_bits(i)


# ---------------------------------------------------------------------------
# The H2D-side contract
# ---------------------------------------------------------------------------


@dataclass
class ConversionResult:
    """Result of a bf16 -> DL16 conversion pass.

    ``words`` are the DL16 words (uint16-range ints, stick-layout
    agnostic — the caller places them). ``flushed_count`` /
    ``saturated_count`` are the receipt counters: nonzero on a
    nonzero-magnitude input means values were lost (flushed below the
    DL16 floor, or saturated at DL16_MAX under the opt-in flag) —
    surfaced, never silent.
    """

    words: list
    symbol_id: str
    flushed_count: int = 0
    saturated_count: int = 0


def _saturate_enabled() -> bool:
    # Read per call (the SENTIENT_DISABLE_COMPOSITE_BALANCE precedent):
    # the flag must take effect without process restart.
    return os.environ.get(_SATURATE_ENV, "") == "1"


def bf16_to_dl16_words(values, symbol_id: str) -> ConversionResult:
    """Convert bf16-representable values to DL16 words via the fp32 pivot.

    ``values`` is a flat iterable of floats (bf16-exact — the caller's
    bf16 -> fp32 upcast is exact, so the pivot's first hop introduces no
    rounding). Underflow flushes to signed zero and is counted; overflow
    refuses with ``Bf16Dl16RangeError`` unless
    ``SENTIENT_BF16_DL16_SATURATE=1``, in which case it saturates to
    the DL16 max word and is counted.
    """
    words = []
    flushed = 0
    saturated = 0
    saturate = _saturate_enabled()
    for index, value in enumerate(values):
        f32 = _bits_f32(float(value))
        fp32_exp = (f32 >> 23) & 0xFF
        raw_exp = fp32_exp - 127
        if fp32_exp == 0xFF:
            raise Bf16Dl16RangeError(symbol_id, "non-finite", value, index)
        if fp32_exp == 0 or raw_exp < 1 - _EXP_BIAS:
            # Zero or below the DL16 floor -> signed zero. Counted only
            # when magnitude was lost (a nonzero input flushed — includes
            # bf16 subnormals, which upcast to fp32 subnormals).
            if value != 0.0:
                flushed += 1
            words.append(0x8000 if f32 & 0x8000_0000 else 0x0000)
            continue
        if value > DL16_MAX or value < -DL16_MAX:
            if saturate:
                saturated += 1
                words.append(
                    (0x8000 if value < 0 else 0x0000) | DL16_MAX_WORD
                )
                continue
            raise Bf16Dl16RangeError(symbol_id, "max", value, index)
        words.append(f32_to_dl16_word(float(value)))
    return ConversionResult(
        words=words,
        symbol_id=symbol_id,
        flushed_count=flushed,
        saturated_count=saturated,
    )


# ---------------------------------------------------------------------------
# The D2H-side comparison contract (S6, plan §5.4)
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Verdict of a fp32-referenced DL16 readback comparison.

    The tolerance is ASYMMETRIC by derivation (not a blanket relative
    band):

    * ``exact_count`` — expected values ON the bf16 grid that match the
      decoded readback EXACTLY. bf16's 7-bit mantissa fits DL16's 9-bit
      grid, so the input contract is exact (S3 Receipt J: relative
      max_diff 0.0 over the boundary sweep); spending tolerance here
      would excuse a real 1-ulp DL16 corruption.
    * ``tolerance_count`` — expected values OFF the bf16 grid (fp32
      intermediates) within PIVOT_TOLERANCE (2**-9) relative — the
      only arm where the fp32->DL16 mantissa hop (23 -> 9) is
      load-bearing.
    * ``flushed_count`` — below-floor expected values whose readback is
      the correctly-signed zero (the warn-with-count underflow policy;
      surfaced, not a mismatch).
    * ``mismatches`` — ``(index, expected, decoded, reason)`` tuples;
      reasons: "bf16-grid-exactness", "pivot-tolerance", "range",
      "zero-sign", "flush-sign".
    """

    symbol_id: str
    element_count: int
    exact_count: int = 0
    tolerance_count: int = 0
    flushed_count: int = 0
    max_abs_diff: float = 0.0
    mismatches: list = field(default_factory=list)


def _is_bf16_grid(value: float) -> bool:
    """True when ``value`` is exactly representable in bf16.

    bf16 is truncated fp32 (the upper 16 bits), so a value is on the
    bf16 grid iff its fp32 pattern's lower 16 bits are zero.
    """
    return _bits_f32(value) & 0xFFFF == 0


def compare_dl16_readback(expected, words, symbol_id: str) -> ComparisonResult:
    """Compare expected values against a DL16 readback, fp32-referenced.

    ``expected`` — flat iterable of floats (bf16-grid values for the
    input contract, off-grid fp32 values for intermediates).
    ``words`` — the DL16 words the device read back. Each is decoded
    through ``dl16_word_to_f32`` (the output-path mirror); the
    comparison happens in fp32 space against the expected values.

    Non-finite expected values refuse typed (the module policy: no
    DL16 inf/nan word exists, and the field-63 marker decodes as a
    large finite value — propagating it would be silent corruption).

    A length mismatch between ``expected`` and ``words`` refuses typed
    (ValueError) — never a prefix comparison. A short or stick-miscounted
    word array is the layout-bug class this oracle exists to catch;
    ``zip``-style silent truncation would green-light exactly that bug.
    """
    expected = list(expected)
    if len(expected) != len(words):
        raise ValueError(
            f"DL16 readback length mismatch for symbol {symbol_id!r}: "
            f"{len(expected)} expected values vs {len(words)} readback "
            f"words. A short or miscounted word array must refuse, never "
            f"compare a prefix — this is the layout-bug class the "
            f"comparison exists to catch."
        )
    result = ComparisonResult(
        symbol_id=symbol_id, element_count=len(words)
    )
    for index, (value, word) in enumerate(zip(expected, words)):
        value = float(value)
        f32 = _bits_f32(value)
        fp32_exp = (f32 >> 23) & 0xFF
        decoded = dl16_word_to_f32(word)
        if fp32_exp == 0xFF:
            raise Bf16Dl16RangeError(symbol_id, "non-finite", value, index)

        if value == 0.0:
            # Signed zero: the readback must be the same-signed zero.
            if decoded != 0.0 or ((word & 0x8000) != (f32 >> 16 & 0x8000)):
                result.mismatches.append(
                    (index, value, decoded, "zero-sign")
                )
            else:
                result.exact_count += 1
            continue

        if abs(value) > DL16_MAX:
            # The conversion itself refuses these; the comparison never
            # silently excuses what the encoder would have rejected.
            result.mismatches.append((index, value, decoded, "range"))
            continue

        if abs(value) < DL16_MIN_NORMAL:
            # Below the DL16 floor: the flush policy's arm. The readback
            # must be the correctly-SIGNED zero; the loss is counted,
            # never silent, never a hard mismatch.
            if decoded != 0.0 or (
                (word & 0x8000) != (0x8000 if value < 0 else 0x0000)
            ):
                result.mismatches.append(
                    (index, value, decoded, "flush-sign")
                )
            else:
                result.flushed_count += 1
            continue

        diff = abs(decoded - value)
        result.max_abs_diff = max(result.max_abs_diff, diff)
        if _is_bf16_grid(value):
            # Input contract: EXACT — a 1-ulp DL16 corruption here is a
            # mismatch a blanket 2**-9 tolerance would excuse.
            if decoded != value:
                result.mismatches.append(
                    (index, value, decoded, "bf16-grid-exactness")
                )
            else:
                result.exact_count += 1
        else:
            # fp32 intermediate: the only arm where the pivot tolerance
            # is load-bearing (the 23 -> 9 mantissa hop).
            if diff <= PIVOT_TOLERANCE * abs(value):
                result.tolerance_count += 1
            else:
                result.mismatches.append(
                    (index, value, decoded, "pivot-tolerance")
                )
    return result
