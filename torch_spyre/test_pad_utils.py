"""T-TB-23 C2 unit tests for the generic load-time pad primitives.

Torch-only (no torch_spyre._C, no device): ``pad_utils`` is pure
host-side shape arithmetic, so these pin the boundary arithmetic, the
loud-refusal classes, and the semantic contract of ``end_pad`` on a
plain host. Red-first: every behavior asserted here is absent on the
parent commit (the module does not exist there).

Run: python3 -m pytest torch_spyre/test_pad_utils.py -q
"""

import math
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from torch_spyre.pad_utils import (  # noqa: E402
    EAR_ADDRESSABLE_BYTES,
    EAR_WIDTH_BITS,
    STICK_BYTES,
    EARSpanUncomputableError,
    InvalidVocabSizeError,
    NotATensorError,
    PadPrimitiveError,
    PadTargetBelowOriginalError,
    UnsupportedPadDtypeError,
    elements_per_stick,
    end_pad,
    pad_length_to_multiple,
    padded_head_dim,
    padded_vocab_size,
    vocab_stick_span_bytes,
)


# --- Stick geometry / constants (KB C1 authority, 2c72d393) ----------------


class TestBoundaryArithmetic:
    def test_stick_constants_match_c1_closed_form(self):
        assert STICK_BYTES == 128
        assert EAR_WIDTH_BITS == 21
        # 2^21 sticks x 128 B = 256 MiB, the closed form not a magic number.
        assert EAR_ADDRESSABLE_BYTES == (1 << 21) * 128 == 256 * 1024 * 1024

    def test_elements_per_stick_derives_from_itemsize(self):
        # Never a hardcoded 64: the count derives from stick / itemsize.
        assert elements_per_stick(torch.float16) == 64
        assert elements_per_stick(torch.bfloat16) == 64
        assert elements_per_stick(torch.float32) == 32
        assert elements_per_stick(torch.int8) == 128

    def test_elements_per_stick_refuses_non_dtype_instance(self):
        # The dtype class itself is not a dtype instance (nor is a str).
        with pytest.raises(UnsupportedPadDtypeError):
            elements_per_stick(torch.dtype)

    def test_elements_per_stick_whitelist_refuses_unsupported(self):
        # bool (128 "elements" of flag -- the trap class) and float64
        # have no known device representation at this granularity (F4).
        with pytest.raises(UnsupportedPadDtypeError):
            elements_per_stick(torch.bool)
        with pytest.raises(UnsupportedPadDtypeError):
            elements_per_stick(torch.float64)

    def test_elements_per_stick_refuses_non_dtype(self):
        with pytest.raises(UnsupportedPadDtypeError):
            elements_per_stick("fp16")

    def test_pad_length_to_multiple_rounds_up(self):
        assert pad_length_to_multiple(100, 64) == 128
        assert pad_length_to_multiple(128, 64) == 128  # identity
        assert pad_length_to_multiple(1, 64) == 64

    def test_pad_length_to_multiple_refuses_non_positive(self):
        with pytest.raises(PadPrimitiveError):
            pad_length_to_multiple(100, 0)

    def test_pad_length_to_multiple_refuses_negative_length(self):
        # A negative length is a config defect, not a pad target --
        # silently returning 0 (or any value) is the nonsense class (F2).
        with pytest.raises(PadPrimitiveError):
            pad_length_to_multiple(-5, 64)


# --- end_pad: the generic end-pad primitive --------------------------------


class TestEndPad:
    def _distinct(self, *shape):
        """Tensor whose values are position-unique along the last dim."""
        return (
            torch.arange(math.prod(shape), dtype=torch.float32)
            .reshape(shape)
            .remainder(97)
        )

    def test_end_pad_grows_dim_with_constant_tail(self):
        t = self._distinct(2, 5)
        out = end_pad(t, dim=1, target_length=8)
        assert out.shape == (2, 8)
        # Original data keeps its positions...
        assert torch.equal(out[:, :5], t)
        # ...and the tail is the constant.
        assert torch.equal(out[:, 5:], torch.zeros(2, 3))

    def test_end_pad_first_dim(self):
        t = self._distinct(3, 4)
        out = end_pad(t, dim=0, target_length=5, fill_value=7.0)
        assert out.shape == (5, 4)
        assert torch.equal(out[:3], t)
        assert torch.equal(out[3:], torch.full((2, 4), 7.0))

    def test_end_pad_identity_when_already_at_target(self):
        t = self._distinct(2, 6)
        out = end_pad(t, dim=1, target_length=6)
        assert out is t  # identity, not a copy that silently "worked"

    def test_end_pad_refuses_non_tensor(self):
        with pytest.raises(NotATensorError):
            end_pad([1, 2, 3], dim=0, target_length=5)

    def test_end_pad_refuses_target_below_original(self):
        t = self._distinct(2, 8)
        with pytest.raises(PadTargetBelowOriginalError):
            end_pad(t, dim=1, target_length=5)

    def test_end_pad_refuses_negative_dim(self):
        t = self._distinct(2, 8)
        with pytest.raises(PadPrimitiveError):
            end_pad(t, dim=-1, target_length=10)

    def test_end_pad_refuses_dim_out_of_range(self):
        # Raw IndexError from tensor.shape[dim] is not a loud refusal (F3).
        t = self._distinct(2, 8)
        with pytest.raises(PadPrimitiveError):
            end_pad(t, dim=2, target_length=10)

    def test_end_pad_refuses_zero_dim_tensor(self):
        with pytest.raises(PadPrimitiveError):
            end_pad(torch.tensor(3.0), dim=0, target_length=4)

    def test_end_pad_refuses_non_int_target(self):
        # Raw TypeError is not a loud refusal (F3).
        t = self._distinct(2, 8)
        with pytest.raises(PadPrimitiveError):
            end_pad(t, dim=1, target_length=None)


# --- Head-dim rule (hf-adapters hf_common.py:3006-3017) -------------------


class TestHeadDimRule:
    def test_head_dim_rounds_to_twice_elements_per_stick(self):
        # The RoPE [2, D/2] reshape needs each half >= one stick:
        # head_dim rounds up to a multiple of 2 x 64 = 128 at fp16.
        assert padded_head_dim(48, torch.float16) == 128
        assert padded_head_dim(64, torch.float16) == 128  # 64 VIOLATES it
        assert padded_head_dim(128, torch.float16) == 128  # already aligned

    def test_head_dim_rule_is_dtype_aware(self):
        assert padded_head_dim(48, torch.float32) == 64  # 2 x 32


# --- Vocab smoothness / EAR span -------------------------------------------


class TestVocabEarSpan:
    def test_non_positive_vocab_refuses_loud(self):
        # vocab 0 is a config defect AND lpf(0) would never terminate
        # (the infinite loop axis-B reproduced, timeout 124) -- both
        # vocab entry points refuse by name BEFORE any loop runs (F1).
        with pytest.raises(InvalidVocabSizeError):
            vocab_stick_span_bytes(0, 2048, torch.float16)
        with pytest.raises(InvalidVocabSizeError):
            padded_vocab_size(0, 2048, torch.float16)
        with pytest.raises(InvalidVocabSizeError):
            vocab_stick_span_bytes(-32000, 2048, torch.float16)

    def test_non_positive_vocab_hang_proof(self):
        # Self-bound hang proof (r2 MINOR rider): the vocab=0 call must
        # REFUSE within a subprocess timeout, not hang -- kills the
        # guard-removal mutant without relying on an external timeout
        # wrapper around the whole pytest run.
        code = (
            "from torch_spyre.pad_utils import padded_vocab_size; "
            "import torch; "
            "padded_vocab_size(0, 2048, torch.float16)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=10,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert b"InvalidVocabSizeError" in result.stderr

    def test_span_arithmetic_matches_c1_formula(self):
        # vocab 151936, hidden 2048, fp16: ceil(151936/64) = 2374 sticks,
        # lpf(2374) = 1187 -> 1187 * 2048 * 64 * 2 = 311,164,928 bytes > 256 MiB
        # (the KB C4 probe's crossing config).
        span = vocab_stick_span_bytes(151936, 2048, torch.float16)
        assert span == 1187 * 2048 * 64 * 2
        assert span > EAR_ADDRESSABLE_BYTES

    def test_span_ceil_pinned_on_non_multiple_vocab(self):
        # vocab 32001 is NOT a multiple of eps=64: ceil(32001/64) = 501
        # sticks, 501 = 3 * 167, lpf = 167 -> 167 * 2048 * 64 * 2. A floor
        # instead of the ceil would see 500 sticks (lpf 5) and under-report
        # the span 33x -- the silently-wrong class on this seam (F1).
        span = vocab_stick_span_bytes(32001, 2048, torch.float16)
        assert span == 167 * 2048 * 64 * 2 == 43_778_048

    def test_padded_vocab_from_non_multiple_vocab(self):
        # Padding from a non-multiple vocab still lands on a multiple of
        # eps whose span is under the bound: ceil(32001/64) = 501 sticks,
        # lpf 167 -> span 43,778,048 B <= 256 MiB, so 32064 is admitted.
        padded = padded_vocab_size(32001, 2048, torch.float16)
        assert padded == 32064
        assert padded % elements_per_stick(torch.float16) == 0
        assert (
            vocab_stick_span_bytes(padded, 2048, torch.float16) <= EAR_ADDRESSABLE_BYTES
        )

    def test_smooth_vocab_stays_under_ear(self):
        # vocab 32000, hidden 2048, fp16: 500 sticks, lpf 5 -> 1,310,720 B.
        assert (
            vocab_stick_span_bytes(32000, 2048, torch.float16)
            == 5 * 2048 * 64 * 2
            < EAR_ADDRESSABLE_BYTES
        )

    def test_padded_vocab_size_pads_up_to_a_multiple_of_eps(self):
        padded = padded_vocab_size(151936, 2048, torch.float16)
        assert padded >= 151936
        assert padded % elements_per_stick(torch.float16) == 0
        assert (
            vocab_stick_span_bytes(padded, 2048, torch.float16) <= EAR_ADDRESSABLE_BYTES
        )

    def test_padded_vocab_identity_when_already_smooth(self):
        assert padded_vocab_size(32000, 2048, torch.float16) == 32000

    def test_uncomputable_span_refuses_loud(self):
        # hidden so large that even lpf=2 exceeds the EAR bound: no vocab
        # padding can fix it -- a hidden/dtype problem, not a padding one.
        # Closed form: uncomputable iff hidden > EAR_BYTES / (2*eps*itemsize).
        eps = elements_per_stick(torch.float16)
        too_large_hidden = EAR_ADDRESSABLE_BYTES // (2 * eps * 2) + 1
        with pytest.raises(EARSpanUncomputableError):
            padded_vocab_size(32000, too_large_hidden, torch.float16)
        # ...and the boundary itself stays computable (equal, not over).
        assert padded_vocab_size(32000, too_large_hidden - 1, torch.float16) >= 32000
