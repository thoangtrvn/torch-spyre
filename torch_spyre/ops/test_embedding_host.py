"""Torch-free unit tests for the embedding launch planner. Run with sentient_codegen
on PYTHONPATH; no torch / torch_spyre import required."""
import math
import pytest

from torch_spyre.ops._embedding_host import build_embedding_launches


def test_multistick_hardware_blocked():
    """d_model>elements_per_stick (multi-stick) is HARDWARE-BLOCKED on AIU 1.0.

    The >1-stick-per-token gather deposit produces wrong output on silicon
    (hardware-confirmed: d=128 max_diff=9; d=2048 device hang). Until that
    embedding-specific deposit defect is fixed, the host helper must reject
    multi-stick loudly rather than ship a wrong/hanging binary. (The codegen
    capacity math and wide-row pointwise dataflow are separately verified; this
    fence is specifically for the embedding deposit defect.)
    """
    idx = list(range(4))
    with pytest.raises(NotImplementedError, match="[Mm]ulti-stick"):
        build_embedding_launches(vocab=128, d_model=4096, element_bits=16, flat_idx=idx)


def test_two_stick_also_blocked():
    """Even the smallest multi-stick case (d=128, spt=2) is blocked — the deposit
    defect is in the basic >1-stick gather, not only wide/chunked rows."""
    idx = list(range(8))
    with pytest.raises(NotImplementedError, match="[Mm]ulti-stick"):
        build_embedding_launches(vocab=1000, d_model=128, element_bits=16, flat_idx=idx)


def test_single_stick_unchanged():
    """d_model=64 (spt=1, C=1) keeps tokens_per_core=8 ceiling."""
    idx = list(range(10))
    launches, tokens_per_launch = build_embedding_launches(
        vocab=64, d_model=64, element_bits=16, flat_idx=idx)
    assert tokens_per_launch == min(32, math.ceil(10 / 8)) * 8  # 2 cores × 8 = 16
