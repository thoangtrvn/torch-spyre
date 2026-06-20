"""Torch-free unit tests for the embedding launch planner. Run with sentient_codegen
on PYTHONPATH; no torch / torch_spyre import required."""
import math
import pytest

from torch_spyre.ops._embedding_host import build_embedding_launches


def test_multistick_no_longer_fenced():
    """d_model=4096 (spt=64, C=2) must plan launches, not raise NotImplementedError."""
    idx = list(range(4))
    launches, tokens_per_launch = build_embedding_launches(
        vocab=128, d_model=4096, element_bits=16, flat_idx=idx)
    assert launches, "expected at least one launch binary"
    # C=2 → tokens_per_core=min(8, 16//2)=8 → ceiling 256 for 32 cores; here N=4 small.
    assert tokens_per_launch >= 4


def test_ceiling_drops_when_lar_binds():
    """d_model=8192 (spt=128, C=4) → tokens_per_core=min(8,16//4)=4."""
    N = 300
    idx = [i % 100 for i in range(N)]
    launches, tokens_per_launch = build_embedding_launches(
        vocab=100, d_model=8192, element_bits=16, flat_idx=idx)
    # tokens_per_core capped at 4 by LAR; with 32 cores tokens_per_launch=128.
    assert tokens_per_launch == 128
    assert len(launches) == math.ceil(N / 128)


def test_single_stick_unchanged():
    """d_model=64 (spt=1, C=1) keeps tokens_per_core=8 ceiling."""
    idx = list(range(10))
    launches, tokens_per_launch = build_embedding_launches(
        vocab=64, d_model=64, element_bits=16, flat_idx=idx)
    assert tokens_per_launch == min(32, math.ceil(10 / 8)) * 8  # 2 cores × 8 = 16
