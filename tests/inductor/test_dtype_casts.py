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

"""Red-first conviction tests for dtype casts on the Spyre backend (1p0).

The bf16-on-1p0 dtype-hygiene lane: on AIU 1.0 a bf16 operand reaches the
device as DL16 (the custom 1-6-9 float). bf16 is E8M7 — the splits differ,
so reinterpreting bf16 bits as DL16 words is numerically meaningless.

Three layers are convicted here:

1. ``TestToDtypeMapping`` — the in-graph typecast surface at the pin-era
   base (60f4226c): every ``to_dtype`` resolved to ``IDENTITY_OP``. NOT
   PORTED to the new-runtime base: the surface it convicted (spyre_kernel
   create_op_spec / SDSC op-spec machinery) was deleted in Phase 4B
   (02a8c8b4) — the class is HELD in the lane workspace pending the S4'
   re-derivation adjudication (receipted absence on this base; the
   new-stack dtype-cast surface is the codegen ``tt.fp_to_fp`` admission
   path, KB-side).

2. ``TestBitMoveGarbage`` — the executable statement of the defect's
   magnitude. A bf16 bit pattern decoded as a DL16 word (decoder mirroring
   the pinned deeptools ``Fp16BinToFloatInternal``,
   ``util/sen_data_convert.cpp:249`` at pin ``e7b8ce6f``) yields garbage for
   the boundary corpus. These PASS today and forever: they document WHY the
   identity mapping is wrong (values computed from the decoder model, never
   hand-asserted).

3. ``TestBf16Dl16Boundaries`` — the refuse-loud / flush-count contract for
   the bf16->DL16 conversion (fp32 pivot). RED on current main: the
   conversion module does not exist yet, so each test fails on import. Turns
   green when the conversion lands with ``Bf16Dl16RangeError`` and the
   flush-count receipt.
"""

import struct
import unittest


def _can_import_mapping_surface() -> bool:
    """The mapping test needs torch_spyre._C (DataFormats /
    encode_constant are imported at superdsc module level). In codegen-only
    local environments the extension is not built; there the red conviction
    is carried by the source-pinned receipt instead (the mapping is
    deterministic: superdsc._get_op_func returns IDENTITY_OP for every
    to_dtype on pinned main 60f4226c)."""
    try:
        from torch_spyre._C import DataFormats, encode_constant  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def _bits_f32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _bits_bf16(value: float) -> int:
    """bf16 bits of a float (truncated fp32 upper half — the exact bf16
    representable set; inputs are constructed to be bf16-exact)."""
    f32 = _bits_f32(value)
    return f32 >> 16


def dl16_decode(word: int) -> float:
    """Decode a DL16 (1-6-9) word as fp32.

    Mirrors the pinned deeptools decoder ``Fp16BinToFloatInternal``
    (util/sen_data_convert.cpp:249-262 at KB pin e7b8ce6f): sign [15],
    exponent [14:9] bias 31, fraction [8:0]; exp==0 && frac==0 is zero;
    there is NO subnormal interpretation. This is the model the garbage
    values below are computed from — never assert a hand-written constant.
    """
    s = (word & 0x8000) >> 15
    x = (word & 0x7E00) >> 9
    y = word & 0x01FF
    if x == 0 and y == 0:
        return 0.0
    i = (s << 31) | (((x - 31 + 127) & 0xFF) << 23) | (y << (23 - 9))
    return _f32_from_bits(i)


class TestBitMoveGarbage(unittest.TestCase):
    """bf16 bits reinterpreted as DL16 words decode to garbage — computed
    from the decoder model above. Documents the defect's magnitude."""

    # (bf16-exact value, note)
    CORPUS = [
        (1.0, "unity"),
        (-1.0, "negative unity"),
        (0.5, "power of two"),
        (3.0, "non-power-of-two normal"),
        (6.0, "mantissa-carrying normal"),
    ]

    def test_bit_moved_bf16_decodes_to_wrong_value(self):
        for value, note in self.CORPUS:
            with self.subTest(value=value, note=note):
                word = _bits_bf16(value)
                decoded = dl16_decode(word)
                self.assertNotEqual(
                    decoded,
                    value,
                    f"bf16 {value} (bits 0x{word:04X}) bit-moved to DL16 "
                    f"decodes to {decoded} — expected garbage, got equality",
                )

    def test_bit_move_is_spot_check_proof(self):
        # ±2.0 SURVIVES the bit-move by coincidence: bf16 0x4000 slices
        # into DL16 exp field 32 (= exponent 1) with zero fraction — the
        # bf16 exponent field's top 6 bits happen to equal the DL16 field
        # for this one binade alignment. A spot check with 2.0 would pass
        # and hide the defect; 1.0 (one binade down) decodes to garbage.
        # This is why the conviction corpus must sweep, not sample.
        for value in (2.0, -2.0):
            with self.subTest(value=value):
                word = _bits_bf16(value)
                self.assertEqual(dl16_decode(word), value)
        self.assertNotEqual(dl16_decode(_bits_bf16(1.0)), 1.0)
        self.assertNotEqual(dl16_decode(_bits_bf16(4.0)), 4.0)

    def test_bit_move_loses_sign_of_zero(self):
        # bf16 -0.0 is 0x8000: DL16 exp==0 && frac==0 -> +0.0. The sign of
        # zero is silently dropped by the bit-move.
        word = _bits_bf16(-0.0)
        self.assertEqual(word, 0x8000)
        decoded = dl16_decode(word)
        self.assertEqual(decoded, 0.0)
        self.assertNotEqual(
            _bits_f32(decoded),
            _bits_f32(-0.0),
            "bit-moved -0.0 must lose its sign bit (conviction of the "
            "sign-loss facet)",
        )

    def test_bit_moved_bf16_nan_becomes_large_finite(self):
        # A bf16 NaN (0x7FC0) has DL16 exponent field 0x3F (all ones) and a
        # nonzero fraction — the decoder maps it to a large FINITE value:
        # NaN silently becomes a number.
        import math

        word = 0x7FC0
        decoded = dl16_decode(word)
        self.assertFalse(math.isnan(decoded))
        self.assertFalse(math.isinf(decoded))
        self.assertGreater(decoded, 1e9)


class TestBf16Dl16Boundaries(unittest.TestCase):
    """The refuse-loud / flush-count contract for the bf16->DL16 fp32-pivot
    conversion. RED on current main: ``torch_spyre._inductor.dtype_convert``
    does not exist yet; each test fails on import until S3 lands it."""

    def _convert_module(self):
        # Deferred import: the module is the S3 deliverable. Red on current
        # main via ModuleNotFoundError — the final-form API the fix must
        # provide.
        import torch_spyre._inductor.dtype_convert as dtype_convert

        return dtype_convert

    def test_just_past_dl16_max_refuses_typed(self):
        dtype_convert = self._convert_module()
        # DL16 max is derived in the module from the store-grid model
        # (never asserted here); just-past-max must raise the typed error
        # naming symbol/boundary/value/index.
        just_past = dtype_convert.DL16_MAX_NEXT_UP_BF16
        with self.assertRaises(dtype_convert.Bf16Dl16RangeError):
            dtype_convert.bf16_to_dl16_words(
                [just_past], symbol_id="test_symbol"
            )

    def test_subnormal_band_flushes_with_count(self):
        dtype_convert = self._convert_module()
        # Values below the DL16 format floor flush to signed zero and the
        # flushed-element COUNT is surfaced (warn-with-count policy,
        # USER-ruled 2026-11-05; see wiki/decisions/bf16-dl16-underflow-policy).
        small = 1e-30  # below the DL16 floor; bf16-representable via fp32 truncation
        result = dtype_convert.bf16_to_dl16_words(
            [small, 1.0, small], symbol_id="test_symbol"
        )
        self.assertEqual(result.flushed_count, 2)
        self.assertEqual(result.words[0], 0x0000)
        # The in-range element converts through the pivot (encoded by the
        # module's own encoder — derived, not hand-asserted).
        self.assertEqual(result.words[1], dtype_convert.f32_to_dl16_word(1.0))
        self.assertEqual(result.words[2], 0x0000)

    def test_roundtrip_within_pivot_tolerance(self):
        dtype_convert = self._convert_module()
        # In-range round-trip bf16 -> DL16 words -> fp32 readback must be
        # within the fp32->DL16 rounding tolerance (mantissa 23->9). The
        # tolerance floor is derived in the module, not hand-picked here.
        corpus = [1.0, -2.0, 0.5, 3.0, 100.0, -0.75]
        result = dtype_convert.bf16_to_dl16_words(corpus, symbol_id="rt")
        for i, value in enumerate(corpus):
            readback = dtype_convert.dl16_word_to_f32(result.words[i])
            self.assertLessEqual(
                abs(readback - value),
                dtype_convert.PIVOT_TOLERANCE * max(abs(value), 1.0),
                f"round-trip element {i} ({value}) exceeds the derived "
                f"pivot tolerance",
            )

    # ------------------------------------------------------------------
    # S3 contract pins (landed with the module; the three tests above
    # were the red-first convictions these now pin green).
    # ------------------------------------------------------------------

    def test_boundaries_derived_and_exact(self):
        import math

        dtype_convert = self._convert_module()
        # The floor is 2**-30 (biased field 1; no subnormal encoding) and
        # the max is (2-2**-9)*2**32 — both DERIVED from the 1-6-9
        # format parameters in the module. These assertions pin the
        # derivation, they do not hand-write the constants into it.
        self.assertEqual(dtype_convert.DL16_MIN_NORMAL, 2.0 ** -30)
        self.assertEqual(
            dtype_convert.DL16_MAX, (2.0 - 2.0 ** -9) * 2.0 ** 32
        )
        self.assertEqual(dtype_convert.DL16_MAX_WORD, 0x7FFF)
        # The floor value itself is in range and survives exactly.
        word = dtype_convert.f32_to_dl16_word(dtype_convert.DL16_MIN_NORMAL)
        self.assertEqual(
            dtype_convert.dl16_word_to_f32(word), dtype_convert.DL16_MIN_NORMAL
        )
        # The max value encodes to the max word and decodes back exactly.
        self.assertEqual(
            dtype_convert.f32_to_dl16_word(dtype_convert.DL16_MAX),
            dtype_convert.DL16_MAX_WORD,
        )
        self.assertEqual(
            dtype_convert.dl16_word_to_f32(dtype_convert.DL16_MAX_WORD),
            dtype_convert.DL16_MAX,
        )
        # The next-up boundary is the first bf16 grid point above
        # DL16_MAX: strictly greater, and the largest bf16 at or below
        # DL16_MAX (its truncation) is still in range.
        self.assertGreater(
            dtype_convert.DL16_MAX_NEXT_UP_BF16, dtype_convert.DL16_MAX
        )
        largest_in_range_bf16 = _f32_from_bits(
            (_bits_f32(dtype_convert.DL16_MAX) >> 16) << 16
        )
        self.assertLessEqual(
            largest_in_range_bf16, dtype_convert.DL16_MAX
        )
        result = dtype_convert.bf16_to_dl16_words(
            [largest_in_range_bf16], symbol_id="edge"
        )
        self.assertEqual(result.flushed_count, 0)
        self.assertLessEqual(
            abs(
                dtype_convert.dl16_word_to_f32(result.words[0])
                - largest_in_range_bf16
            ),
            dtype_convert.PIVOT_TOLERANCE * largest_in_range_bf16,
        )

    def test_floor_edge_flushes_below_not_at(self):
        import math

        dtype_convert = self._convert_module()
        # The flush boundary is the RAW exponent: values with raw
        # exponent < -30 flush (counted); 2**-30 itself (raw exponent
        # exactly -30) is the smallest in-range normal. Mirrors
        # FloatToFp16BinInternal's pre-rounding `exponent < -30` check.
        # The fp32 predecessor of the floor, built by bit decrement (a
        # float64 nextafter would round back up at the fp32 cast).
        just_below = _f32_from_bits(
            _bits_f32(dtype_convert.DL16_MIN_NORMAL) - 1
        )
        result = dtype_convert.bf16_to_dl16_words(
            [just_below, dtype_convert.DL16_MIN_NORMAL, -just_below],
            symbol_id="floor",
        )
        self.assertEqual(result.flushed_count, 2)
        self.assertEqual(result.words[0], 0x0000)
        self.assertEqual(result.words[2], 0x8000)  # signed zero keeps sign
        self.assertNotEqual(result.words[1], 0x0000)
        self.assertEqual(
            dtype_convert.dl16_word_to_f32(result.words[1]),
            dtype_convert.DL16_MIN_NORMAL,
        )

    def test_zero_passes_uncounted_with_sign(self):
        dtype_convert = self._convert_module()
        result = dtype_convert.bf16_to_dl16_words(
            [0.0, -0.0], symbol_id="z"
        )
        self.assertEqual(result.flushed_count, 0)
        self.assertEqual(result.words, [0x0000, 0x8000])

    def test_nonfinite_refuses_typed(self):
        import math

        dtype_convert = self._convert_module()
        for bad in (float("inf"), float("nan"), float("-inf")):
            with self.subTest(value=bad):
                with self.assertRaises(dtype_convert.Bf16Dl16RangeError) as ctx:
                    dtype_convert.bf16_to_dl16_words([bad], symbol_id="nf")
                # Non-finite is its own boundary class, never "max".
                self.assertEqual(ctx.exception.boundary, "non-finite")

    def test_error_fields_name_symbol_value_index(self):
        dtype_convert = self._convert_module()
        with self.assertRaises(dtype_convert.Bf16Dl16RangeError) as ctx:
            dtype_convert.bf16_to_dl16_words(
                [1.0, dtype_convert.DL16_MAX_NEXT_UP_BF16],
                symbol_id="fields",
            )
        err = ctx.exception
        self.assertEqual(err.symbol_id, "fields")
        self.assertEqual(err.boundary, "max")
        self.assertEqual(err.value, dtype_convert.DL16_MAX_NEXT_UP_BF16)
        self.assertEqual(err.linear_index, 1)

    def test_saturation_opt_in_counts(self):
        dtype_convert = self._convert_module()
        just_past = dtype_convert.DL16_MAX_NEXT_UP_BF16
        with unittest.mock.patch.dict(
            "os.environ", {"SENTIENT_BF16_DL16_SATURATE": "1"}
        ):
            result = dtype_convert.bf16_to_dl16_words(
                [just_past, -just_past, 1.0], symbol_id="sat"
            )
        self.assertEqual(result.saturated_count, 2)
        self.assertEqual(result.flushed_count, 0)
        self.assertEqual(result.words[0], dtype_convert.DL16_MAX_WORD)
        self.assertEqual(
            result.words[1], 0x8000 | dtype_convert.DL16_MAX_WORD
        )
        self.assertEqual(
            result.words[2], dtype_convert.f32_to_dl16_word(1.0)
        )

    def test_random_bf16_corpus_roundtrip_max_diff(self):
        # Plan §8 metric 4: round-trip max_diff over an in-range random
        # corpus must stay within the derived tolerance floor. The corpus
        # is bf16-exact by construction (fp32 truncation), covering the
        # full in-range exponent span including both boundaries.
        import math
        import random

        dtype_convert = self._convert_module()
        rng = random.Random(20261105)
        corpus = []
        # Deterministic sweep of the exponent range plus random mantissas.
        for raw_exp in range(-30, 33):
            for _ in range(16):
                m = 1.0 + rng.random()
                v = math.ldexp(m, raw_exp)
                if v <= dtype_convert.DL16_MAX:
                    corpus.append(v)
                    corpus.append(-v)
        corpus.extend(
            [dtype_convert.DL16_MIN_NORMAL, dtype_convert.DL16_MAX, 1.0]
        )
        # Snap every value to the bf16 grid (the input contract).
        corpus = [_f32_from_bits((_bits_f32(v) >> 16) << 16) for v in corpus]
        result = dtype_convert.bf16_to_dl16_words(corpus, symbol_id="sweep")
        self.assertEqual(result.flushed_count, 0)
        # S6 refit: the comparison goes through the fp32-referenced
        # utility (plan §5.4) — for this bf16-grid corpus it demands
        # EXACT equality (the tolerance arm is for fp32 intermediates,
        # not the input contract). The max_diff figure below is the
        # receipt metric, computed from the same verdict.
        comparison = dtype_convert.compare_dl16_readback(
            corpus, result.words, symbol_id="sweep"
        )
        self.assertEqual(comparison.mismatches, [])
        self.assertEqual(comparison.exact_count, len(corpus))
        self.assertEqual(comparison.tolerance_count, 0)
        self.assertEqual(comparison.max_abs_diff, 0.0)
        self.assertGreater(len(corpus), 1000)


class TestDl16ComparisonUtility(unittest.TestCase):
    """S6 (plan §5.4): the fp32-referenced compare utility.

    The tolerance is ASYMMETRIC by derivation, not blanket:

    * bf16-grid expected values (the input contract) are DL16-EXACT —
      bf16's 7-bit mantissa fits DL16's 9-bit grid (S3 Receipt J:
      relative max_diff 0.0 over the 1008-value sweep). The comparison
      demands EXACT equality for them; a blanket 2^-9 tolerance would
      excuse a real 1-ulp DL16 corruption on the input path.
    * The 2^-9 PIVOT_TOLERANCE is load-bearing ONLY for off-grid fp32
      intermediates — values that were never bf16 and pay the
      fp32->DL16 mantissa hop (23 -> 9).
    * Below-floor expected values flush to signed zero (the USER-ruled
      warn-with-count policy); the utility verdicts them "flushed" with
      sign preserved, not as mismatches.
    """

    def _convert_module(self):
        # Same deferred-import pattern as TestBf16Dl16Boundaries: the
        # S6 utility is the deliverable these tests convict into
        # existence (red via AttributeError until it lands).
        import torch_spyre._inductor.dtype_convert as dtype_convert

        return dtype_convert

    def test_bf16_grid_corpus_compares_exact(self):
        # The input contract: every in-range bf16 value must compare
        # EXACT — no tolerance spent. This is the arm a blanket
        # tolerance would silently weaken.
        import math
        import random

        dtype_convert = self._convert_module()
        rng = random.Random(20261106)
        corpus = []
        for raw_exp in range(-30, 33):
            for _ in range(8):
                v = math.ldexp(1.0 + rng.random(), raw_exp)
                if v <= dtype_convert.DL16_MAX:
                    corpus.append(v)
        corpus = [
            _f32_from_bits((_bits_f32(v) >> 16) << 16) for v in corpus
        ]
        result = dtype_convert.bf16_to_dl16_words(corpus, "exact-arm")
        comparison = dtype_convert.compare_dl16_readback(
            corpus, result.words, symbol_id="exact-arm"
        )
        self.assertEqual(comparison.mismatches, [])
        self.assertEqual(comparison.exact_count, len(corpus))
        self.assertEqual(comparison.tolerance_count, 0)
        self.assertEqual(comparison.flushed_count, 0)
        self.assertEqual(comparison.max_abs_diff, 0.0)

    def test_length_mismatch_refuses_typed(self):
        # Gauntlet r1 P2-1: zip() stops at the shorter sequence, so a
        # short/stick-miscounted word array compared only its prefix and
        # returned a false green — the exact layout-bug class this oracle
        # exists to catch, silenced by the oracle's own blind spot. A
        # length mismatch must refuse typed, never compare a prefix.
        dtype_convert = self._convert_module()
        expected = [1.0, 2.0, 4.0, 8.0]
        full = dtype_convert.bf16_to_dl16_words(expected, "len-arm")
        with self.assertRaises(ValueError) as ctx:
            dtype_convert.compare_dl16_readback(
                expected, full.words[:-1], symbol_id="len-arm"
            )
        message = str(ctx.exception)
        self.assertIn("len-arm", message)
        self.assertIn("length mismatch", message)
        # Control: equal lengths still compare (the prefix refusal did
        # not break the honest path).
        comparison = dtype_convert.compare_dl16_readback(
            expected, full.words, symbol_id="len-arm"
        )
        self.assertEqual(comparison.mismatches, [])
        self.assertEqual(comparison.exact_count, 4)

    def test_offgrid_fp32_intermediates_use_pivot_tolerance(self):
        # Off-grid fp32 values pay the 23->9 mantissa hop: within
        # PIVOT_TOLERANCE relative, counted in the tolerance arm —
        # never in the exact arm.
        import math
        import random

        dtype_convert = self._convert_module()
        rng = random.Random(20261107)
        corpus = []
        for raw_exp in range(-25, 30):
            for _ in range(8):
                # Full fp32 mantissa — deliberately NOT bf16-snapped.
                v = math.ldexp(1.0 + rng.random(), raw_exp)
                if v <= dtype_convert.DL16_MAX:
                    corpus.append(v)
        words = [dtype_convert.f32_to_dl16_word(v) for v in corpus]
        comparison = dtype_convert.compare_dl16_readback(
            corpus, words, symbol_id="tolerance-arm"
        )
        self.assertEqual(comparison.mismatches, [])
        self.assertEqual(comparison.exact_count, 0)
        self.assertGreater(comparison.tolerance_count, 0)
        self.assertEqual(
            comparison.tolerance_count + comparison.exact_count,
            len(corpus),
        )

    def test_one_ulp_corruption_on_bf16_grid_value_detected(self):
        # THE ASYMMETRY CONVICTION: a 1-DL16-ulp corruption of a
        # bf16-grid value is a MISMATCH under this utility — a blanket
        # 2^-9 relative tolerance would excuse it (1 ulp at 9 mantissa
        # bits IS 2^-9 relative). Exactness for the input contract is
        # load-bearing.
        dtype_convert = self._convert_module()
        value = 1.5
        word = dtype_convert.f32_to_dl16_word(value)
        self.assertEqual(dtype_convert.dl16_word_to_f32(word), 1.5)
        corrupted = word + 1  # one DL16 ulp up
        comparison = dtype_convert.compare_dl16_readback(
            [value], [corrupted], symbol_id="ulp-conviction"
        )
        self.assertEqual(len(comparison.mismatches), 1)
        index, expected, decoded, reason = comparison.mismatches[0]
        self.assertEqual(index, 0)
        self.assertEqual(expected, 1.5)
        self.assertEqual(decoded, dtype_convert.dl16_word_to_f32(corrupted))
        self.assertIn("exact", reason)

    def test_bit_moved_readback_detected(self):
        # The defect the lane exists to close, made DETECTABLE: bf16
        # bits reinterpreted as a DL16 word compare as garbage — the
        # utility catches what the old dtype-blind comparison could not.
        dtype_convert = self._convert_module()
        value = 1.0  # bf16 word 0x3F80
        bit_moved = 0x3F80
        comparison = dtype_convert.compare_dl16_readback(
            [value], [bit_moved], symbol_id="bit-move-conviction"
        )
        self.assertEqual(len(comparison.mismatches), 1)

    def test_below_floor_flushes_with_sign_verdict(self):
        # Warn-with-count policy at the comparison surface: a below-
        # floor expected value against a signed-zero readback is a
        # "flushed" verdict (counted, surfaced), not a mismatch — but
        # the WRONG-SIGN zero is a mismatch.
        dtype_convert = self._convert_module()
        small = dtype_convert.DL16_MIN_NORMAL / 2.0
        comparison = dtype_convert.compare_dl16_readback(
            [small, -small], [0x0000, 0x8000], symbol_id="flush"
        )
        self.assertEqual(comparison.mismatches, [])
        self.assertEqual(comparison.flushed_count, 2)
        # Wrong sign on the flushed zero: detected.
        bad = dtype_convert.compare_dl16_readback(
            [-small], [0x0000], symbol_id="flush-sign"
        )
        self.assertEqual(len(bad.mismatches), 1)

    def test_out_of_range_expected_is_named_mismatch(self):
        # The comparison never silently excuses an expected value the
        # conversion itself would have refused.
        dtype_convert = self._convert_module()
        comparison = dtype_convert.compare_dl16_readback(
            [dtype_convert.DL16_MAX * 2.0],
            [dtype_convert.DL16_MAX_WORD],
            symbol_id="range",
        )
        self.assertEqual(len(comparison.mismatches), 1)
        self.assertIn("range", comparison.mismatches[0][3])

    def test_nonfinite_expected_refuses_typed(self):
        dtype_convert = self._convert_module()
        with self.assertRaises(dtype_convert.Bf16Dl16RangeError):
            dtype_convert.compare_dl16_readback(
                [float("nan")], [0x0000], symbol_id="nan"
            )


def _can_import_seam_guard_surface() -> bool:
    """The seam-guard tests need torch_spyre._C (dl16_h2d_seam_boundary /
    dl16_h2d_seam_check / SpyreTensorLayout). In environments without the
    built extension they are skipped; the boundary itself is additionally
    pinned source-side by the derivation in dl16_seam_guard.h (decode of
    word 0x7DFF in the 1-6-9 format)."""
    try:
        import torch_spyre._C as _C  # noqa: F401

        _C.dl16_h2d_seam_boundary  # noqa: B018
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False
    return True


@unittest.skipUnless(
    _can_import_seam_guard_surface(),
    "torch_spyre._C with the T-DC-17 seam guard not built",
)
class TestH2DSeamGuard(unittest.TestCase):
    """The T-DC-17 contract: the H2D converter-in seam's silent saturation
    (0x7DFF, silicon-witnessed 2026-11-05, claim
    bf16pivot-sixitem-20261105-002 item 2) is converted to a typed loud
    refusal on the host, before the DMA.

    All boundary values are DERIVED from the DL16 decoder model above —
    never hand-asserted. Two numbers, two sites: the seam ceiling
    (decode(0x7DFF), the converter-in's usable limit) is distinct from
    the DL16 format max (decode(0x7FFF), refused by the fp32 pivot in
    dtype_convert) — the tests pin both, and their separation."""

    def _c(self):
        import torch_spyre._C as _C

        return _C

    def test_seam_ceiling_derived_from_saturation_word(self):
        _C = self._c()
        # The ceiling is the decode of the witnessed saturation word
        # 0x7DFF — derived through the decoder model, and it sits one
        # binade below the format max decode(0x7FFF).
        seam = dl16_decode(0x7DFF)
        self.assertEqual(seam, 4290772992.0)
        self.assertLess(seam, dl16_decode(0x7FFF))
        # The ceiling itself is admitted; the format max is NOT (it is
        # above the seam — the honest narrowing this guard implements).
        self.assertEqual(_C.dl16_h2d_seam_boundary(seam), "ok")
        self.assertEqual(
            _C.dl16_h2d_seam_boundary(dl16_decode(0x7FFF)),
            "over-seam-ceiling",
        )

    def test_witnessed_saturators_refuse(self):
        _C = self._c()
        # The box-run divergence corpus: over-range inputs AND the largest
        # in-range grid value (module word 0x7FF8) all saturate at 0x7DFF
        # on device — every one of them must refuse on the host now.
        in_range_grid_max = dl16_decode(0x7FF8)  # 8522825728, derived
        self.assertGreater(in_range_grid_max, 4290772992.0)
        for value in (in_range_grid_max, 2.0**33, 1e10, 2.0**40):
            with self.subTest(value=value):
                self.assertEqual(
                    _C.dl16_h2d_seam_boundary(value), "over-seam-ceiling"
                )
                self.assertEqual(
                    _C.dl16_h2d_seam_boundary(-value), "over-seam-ceiling"
                )

    def test_just_above_ceiling_refuses(self):
        _C = self._c()
        import math

        seam = dl16_decode(0x7DFF)
        just_above = math.nextafter(seam, math.inf)
        self.assertEqual(_C.dl16_h2d_seam_boundary(seam), "ok")
        self.assertEqual(
            _C.dl16_h2d_seam_boundary(just_above), "over-seam-ceiling"
        )

    def test_in_band_values_pass(self):
        _C = self._c()
        # The entire witnessed-correct range, including the fp16 format
        # max (eight binades below the ceiling) and the real-model census
        # max (SmolLM2-135M max |w| = 5.25).
        for value in (0.0, -0.0, 1.0, -1.0, 5.25, 65504.0, 2.0**31, -2.0**31):
            with self.subTest(value=value):
                self.assertEqual(_C.dl16_h2d_seam_boundary(value), "ok")

    def test_nonfinite_refuses(self):
        _C = self._c()
        # No DL16 inf/nan word exists; the converter's field-63 marker
        # decodes as a large finite value (silent corruption) — refuse.
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                self.assertEqual(
                    _C.dl16_h2d_seam_boundary(value), "non-finite"
                )

    def _sen169_layout(self, numel):
        _C = self._c()
        return _C.SpyreTensorLayout(
            [numel], [numel], _C.DataFormats.SEN169_FP16
        )

    def test_tensor_guard_refuses_bad_bf16_element(self):
        import torch

        _C = self._c()
        # A bf16 tensor with one over-seam element refuses loudly
        # (RuntimeError from TORCH_CHECK), naming the seam.
        t = torch.tensor([1.0, 2.0**33, 3.0], dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "seam"):
            _C.dl16_h2d_seam_check(t, self._sen169_layout(3))

    def test_tensor_guard_refuses_nonfinite_bf16_element(self):
        import torch

        _C = self._c()
        t = torch.tensor([1.0, float("inf")], dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _C.dl16_h2d_seam_check(t, self._sen169_layout(2))

    def test_tensor_guard_passes_clean_bf16_tensor(self):
        import torch

        _C = self._c()
        t = torch.tensor([1.0, -5.25, 65504.0], dtype=torch.bfloat16)
        self.assertIsNone(
            _C.dl16_h2d_seam_check(t, self._sen169_layout(3))
        )

    def test_tensor_guard_scans_strided_view(self):
        # The DMA converts the logical view (stride-aware), so the guard
        # must too: a bad element reachable only through a stride > 1
        # still refuses — the odometer, not a contiguous memcpy.
        import torch

        _C = self._c()
        base = torch.tensor(
            [1.0, 1.0, 2.0**33, 1.0, 1.0, 1.0], dtype=torch.bfloat16
        )
        strided = base.as_strided((3,), (2,), storage_offset=0)
        with self.assertRaisesRegex(RuntimeError, "seam"):
            _C.dl16_h2d_seam_check(strided, self._sen169_layout(3))
        # The same base tensor viewed without the bad element (indices
        # 3 and 5, skipping the 2**33 at index 2) passes.
        clean = base.as_strided((2,), (2,), storage_offset=3)
        self.assertIsNone(
            _C.dl16_h2d_seam_check(clean, self._sen169_layout(2))
        )

    def test_tensor_guard_noop_for_non_dl16_destination(self):
        # A non-SEN169_FP16 destination never routes through the
        # converting seam: the guard is a no-op even for wild values.
        import torch

        _C = self._c()
        t = torch.tensor([2.0**40, float("inf")], dtype=torch.bfloat16)
        layout = _C.SpyreTensorLayout(
            [2], [2], _C.DataFormats.IEEE_FP32
        )
        self.assertIsNone(_C.dl16_h2d_seam_check(t, layout))

    def test_tensor_guard_noop_for_fp32_source(self):
        # fp32 H2D is a pure bit-move (silicon-witnessed, claim item 3):
        # no converter-in numerics, no seam exposure — no-op.
        import torch

        _C = self._c()
        t = torch.tensor([2.0**40, float("inf")], dtype=torch.float32)
        self.assertIsNone(
            _C.dl16_h2d_seam_check(t, self._sen169_layout(2))
        )

    def test_tensor_guard_covers_fp16_source(self):
        # fp16's max (65504) cannot cross the ceiling, but the guard
        # scans the whole 16-bit-float dtype class (derived from the
        # seam's dtype class, not a guess): a synthetic fp16 tensor is
        # scanned and passes; the guard surface does not skip it.
        import torch

        _C = self._c()
        t = torch.tensor([1.0, 65504.0], dtype=torch.float16)
        self.assertIsNone(
            _C.dl16_h2d_seam_check(t, self._sen169_layout(2))
        )

    # ------------------------------------------------------------------
    # Hook-exercise tests (gauntlet r1 finding 1): the guard's PRODUCTION
    # call site is SpyreStream::copyAsync (spyre_stream.cpp) — these drive
    # the real H2D copy path through _C.dl16_h2d_seam_copy_probe, which
    # fabricates a SEN169_FP16 destination and calls copyAsync exactly as
    # spyre_copy_from does. The refusal must fire from the real hook
    # (before generate_dci / any flex call), and a clean element must
    # PASS the hook (stopping only at the runtime-free environment's
    # stream-pool lookup) — not just at the oracle classifier.
    # ------------------------------------------------------------------

    def test_copy_hook_over_seam_element_refuses_from_real_path(self):
        import torch

        _C = self._c()
        t = torch.tensor([1.0, 2.0**33], dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "converter-in seam"):
            _C.dl16_h2d_seam_copy_probe(t)

    def test_copy_hook_nonfinite_element_refuses_from_real_path(self):
        import torch

        _C = self._c()
        t = torch.tensor([1.0, float("inf")], dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _C.dl16_h2d_seam_copy_probe(t)

    def test_copy_hook_at_ceiling_element_passes_the_hook(self):
        # The largest bf16 grid point at or below the seam ceiling
        # (truncation of decode(0x7DFF) onto the bf16 grid — derived, and
        # strictly below the ceiling because decode(0x7DFF) needs 9
        # mantissa bits): it must be ADMITTED by the real copyAsync hook.
        # The copy then proceeds past the guard into DMA setup and stops
        # at the stream-pool lookup — the runtime-free environment's own
        # refusal, NOT the seam guard. That error shape is the witness
        # that the hook (not just the oracle) let the value through.
        import torch

        _C = self._c()
        seam = dl16_decode(0x7DFF)
        at_ceiling_bf16 = _f32_from_bits((_bits_f32(seam) >> 16) << 16)
        self.assertLessEqual(at_ceiling_bf16, seam)
        self.assertGreater(at_ceiling_bf16, 0.0)
        t = torch.tensor([1.0, at_ceiling_bf16], dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "flex handle") as raised:
            _C.dl16_h2d_seam_copy_probe(t)
        self.assertNotIn(
            "seam",
            str(raised.exception),
            "the at-ceiling element must pass the T-DC-17 guard; the "
            "RuntimeError must come from the stream-pool lookup, not the "
            "guard",
        )

    def test_copy_hook_witnessed_saturator_refuses_from_real_path(self):
        # The box-run's in-range-but-seam-saturated grid value (module
        # word 0x7FF8, derived through the decoder) refuses at the real
        # hook too — the exact class the silicon witness saturated.
        import torch

        _C = self._c()
        in_range_grid_max = dl16_decode(0x7FF8)
        t = torch.tensor([in_range_grid_max], dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "converter-in seam"):
            _C.dl16_h2d_seam_copy_probe(t)


if __name__ == "__main__":
    unittest.main()
