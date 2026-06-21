"""Torch-free unit tests for the embedding launch planner. Run with sentient_codegen
on PYTHONPATH; no torch / torch_spyre import required."""
import math
import pytest

from torch_spyre.ops._embedding_host import build_embedding_launches, _tokens_per_core


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
    """d_model=64 (spt=1) keeps tokens_per_core=8 ceiling (PM-T3: unchanged).

    Plane-major ceiling for spt=1: floor(8/1)=8. Same as old model.
    """
    idx = list(range(10))
    launches, tokens_per_launch = build_embedding_launches(
        vocab=64, d_model=64, element_bits=16, flat_idx=idx)
    assert tokens_per_launch == min(32, math.ceil(10 / 8)) * 8  # 2 cores × 8 = 16


# ---------------------------------------------------------------------------
# PM-Task 3: host ceiling tests
# ---------------------------------------------------------------------------

def test_pm3_d1024_raises_not_implemented():
    """d_model=1024 (spt=16) raises NotImplementedError on the multi-stick fence.

    PM-T3: spt=16 > 1, so the hardware fence fires before the EBR ceiling check.
    The match confirms it is the multi-stick hardware fence, not an unrelated error.
    (If the fence were removed, the EBR ceiling — _tokens_per_core(8,16,16)==0 — would
    also raise NotImplementedError, so this path is double-gated.)
    """
    idx = list(range(4))
    with pytest.raises(NotImplementedError, match="[Mm]ulti-stick"):
        build_embedding_launches(vocab=128, d_model=1024, element_bits=16, flat_idx=idx)


@pytest.mark.parametrize("spt,expected", [
    (1,  8),  # spt=1  → floor(8/1)=8 (single-stick, unchanged from old model)
    (2,  4),  # spt=2  → floor(8/2)=4 (d=128 at 16-bit)
    (4,  2),  # spt=4  → floor(8/4)=2 (d=256 at 16-bit)
    (8,  1),  # spt=8  → floor(8/8)=1 (d=512, max spt under this model)
    (16, 0),  # spt=16 → floor(8/16)=0 (d=1024 exceeds EBR file; caller raises)
])
def test_pm3_tokens_per_core(spt, expected):
    """PM-T3: _tokens_per_core exercises the REAL production formula for all spt values.

    ebr_count=8, lar_count=16 (AIU 1.0 hardware constants via gen registry).
    These tests call _tokens_per_core directly, so a bug in the production formula
    (e.g. using max instead of min, wrong integer-division, wrong register count)
    will immediately cause a failure — unlike the previous inline-arithmetic versions
    which were tautological (they reimplemented the formula, not called it).

    spt>1 coverage is possible here because _tokens_per_core has no hardware fence;
    the fence lives in build_embedding_launches and only blocks the end-to-end path.
    The codegen guard (test_pm3_guard_* in codegen tests) provides additional spt>1
    coverage through the scheduler path.
    """
    assert _tokens_per_core(8, 16, spt) == expected, (
        f"PM-T3: _tokens_per_core(8, 16, {spt}) = "
        f"{_tokens_per_core(8, 16, spt)} != {expected}"
    )


def test_pm3_spt1_tokens_per_core_unchanged():
    """PM-T3: spt=1 ceiling is floor(8/1)=8 — byte-identical to old model.

    Verifies via build_embedding_launches that tokens_per_core=8 for spt=1.
    """
    # 8 exact tokens → 1 launch with 1 core × 8 tokens
    launches, tokens_per_launch = build_embedding_launches(
        vocab=64, d_model=64, element_bits=16, flat_idx=list(range(8)),
    )
    assert tokens_per_launch == 8, (
        f"PM-T3: spt=1, 8 tokens → tokens_per_launch=8, got {tokens_per_launch}"
    )
