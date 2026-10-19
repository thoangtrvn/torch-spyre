"""T-SR-50: dtype-aware elements-per-stick in SpyreHeuristics.

`choices.py` hardcoded elements_per_stick = 64 — the 16-bit value (128-byte
stick / 2 bytes). Every non-16-bit kernel (FP8: 128, INT32/FP32: 32,
BOOL: 128) was silently mistiled. These tests pin the dtype derivation
seam: `SpyreHeuristics._kernel_dtype` resolves the kernel's (single) dtype
from its scheduler nodes and refuses on ambiguity, and
`torch_spyre._C.get_elem_in_stick` supplies the per-dype stick width.
"""

import pytest
import torch

from torch_spyre._C import get_elem_in_stick
from torch_spyre._inductor.choices import SpyreHeuristics


class _FakeBuffer:

    def __init__(self, dtype):
        self._dtype = dtype

    def get_dtype(self):
        return self._dtype


class _FakeNode:

    def __init__(self, dtype):
        self.node = _FakeBuffer(dtype) if dtype is not None else object()


class _FakeFeatures:

    def __init__(self, *dtypes):
        self._nodes = [_FakeNode(d) for d in dtypes]

    def scheduler_nodes(self):
        return self._nodes


# --- get_elem_in_stick: the dtype → stick-width table -----------------------

@pytest.mark.parametrize("dtype,expected", [
    (torch.float16, 64),
    (torch.bfloat16, 64),
    (torch.float32, 32),
    (torch.int32, 32),
    (torch.int64, 16),
    (torch.bool, 128),
])
def test_get_elem_in_stick_table(dtype, expected):
    assert get_elem_in_stick(dtype) == expected


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="torch lacks float8_e4m3fn"
)
def test_get_elem_in_stick_fp8():
    assert get_elem_in_stick(torch.float8_e4m3fn) == 128


# --- _kernel_dtype: the derivation seam -------------------------------------

def test_kernel_dtype_single_node():
    features = _FakeFeatures(torch.float16)
    assert SpyreHeuristics._kernel_dtype(features) == torch.float16


def test_kernel_dtype_agreeing_nodes():
    features = _FakeFeatures(torch.float16, torch.float16, None)
    assert SpyreHeuristics._kernel_dtype(features) == torch.float16


def test_kernel_dtype_mixed_refuses():
    """A node disagreement must refuse, not silently pick one."""
    features = _FakeFeatures(torch.float16, torch.float32)
    with pytest.raises(NotImplementedError, match="mixed-dtype"):
        SpyreHeuristics._kernel_dtype(features)


def test_kernel_dtype_unresolvable_refuses():
    """No resolvable dtype must refuse, not fall back to a default."""
    features = _FakeFeatures(None)
    with pytest.raises(NotImplementedError, match="could not resolve"):
        SpyreHeuristics._kernel_dtype(features)


def test_no_hardcoded_sixty_four_at_the_eps_site():
    """Red-first pin for the original defect: the eps site must route through
    get_elem_in_stick, not a bare 64 literal (dtype-dependent block sizing)."""
    import inspect
    from torch_spyre._inductor import choices

    src = inspect.getsource(choices)
    assert "elements_per_stick = get_elem_in_stick(" in src
    assert "elements_per_stick = 64" not in src
