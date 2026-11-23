# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The ATen exposure mapping for SDPA on spyre — off-box pins.

The mapping (``torch_spyre.ops.attention.sdpa_attention_mapping``) is the
DELEGATING surface: it derives (rows, head_dim, kv_len, mask_mode,
gqa_group, dtype_bits, scale) per (b, h) from the ATen call and delegates
admission to codegen's attention predicate — it invents NO refusal of its
own beyond the two ATen-flag refusals (dropout, debug mask) that are not
envelope questions.

Every envelope refusal here must carry the PREDICATE'S OWN clause name
([row-budget], [head-dim], [dtype], [mask-mode], [gqa-group], [kv-block],
[kv-len]) — an EncodingError, a bare ValueError, or a silent CPU fallback
at an admitted shape is a defect (the op-route contract).
"""

import pytest
import torch


def _mapping():
    try:
        from torch_spyre.ops.attention import sdpa_attention_mapping
    except ImportError as e:
        pytest.skip(f"mapping not importable: {e}")
    return sdpa_attention_mapping


def _qkv(batch=1, heads=1, kv_heads=None, rows=8, kv=64, d=64, dtype=torch.float16):
    kv_heads = kv_heads if kv_heads is not None else heads
    g = torch.Generator().manual_seed(20261123)
    q = torch.randn(batch, heads, rows, d, generator=g).to(dtype)
    k = torch.randn(batch, kv_heads, kv, d, generator=g).to(dtype)
    v = torch.randn(batch, kv_heads, kv, d, generator=g).to(dtype)
    return q, k, v


def test_mapping_admits_the_witnessed_cell():
    mapping = _mapping()
    q, k, v = _qkv()
    calls = mapping(q, k, v, None, 0.0, True, False, None)
    assert len(calls) == 1  # B=1, H=1
    call = calls[0]
    assert call["rows"] == 8
    assert call["head_dim"] == 64
    assert call["kv_len"] == 64
    assert call["mask_mode"] == "causal"
    assert call["gqa_group"] == 1
    assert call["dtype_bits"] == 16
    assert call["scale"] == pytest.approx(1.0 / 8.0)
    assert call["strategy"] == "container"


def test_mapping_multiblock_and_batch_iteration():
    mapping = _mapping()
    q, k, v = _qkv(batch=2, heads=3, kv=128)
    calls = mapping(q, k, v, None, 0.0, False, False, None)
    assert len(calls) == 6  # B*H per-(b,h) launches
    assert all(c["kv_len"] == 128 for c in calls)
    assert all(c["strategy"] == "host_serial" for c in calls)
    assert all(c["mask_mode"] == "none" for c in calls)


def test_mapping_explicit_scale():
    mapping = _mapping()
    q, k, v = _qkv()
    calls = mapping(q, k, v, None, 0.0, True, False, 0.5)
    assert calls[0]["scale"] == 0.5


# ---------------------------------------------------------------------------
# The refusal sweep — the predicate's own clause names, never invented
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_kwargs,clause",
    [
        (dict(rows=64), "row-budget"),
        (dict(d=100), "head-dim"),  # not a whole-stick multiple
        (dict(d=576), "head-dim"),  # > 512
        (dict(kv=100), "kv-len"),  # ragged multiblock
        (dict(kv=32), "kv-block"),  # sub-block
    ],
)
def test_refusal_sweep_envelope(make_kwargs, clause):
    mapping = _mapping()
    q, k, v = _qkv(**make_kwargs)
    with pytest.raises(NotImplementedError) as ei:
        mapping(q, k, v, None, 0.0, True, False, None)
    assert f"[{clause}]" in str(ei.value)


def test_refusal_fp32():
    mapping = _mapping()
    q, k, v = _qkv(dtype=torch.float32)
    with pytest.raises(NotImplementedError) as ei:
        mapping(q, k, v, None, 0.0, True, False, None)
    assert "[dtype]" in str(ei.value)


def test_refusal_bf16():
    mapping = _mapping()
    q, k, v = _qkv(dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError) as ei:
        mapping(q, k, v, None, 0.0, True, False, None)
    assert "[dtype]" in str(ei.value)


def test_refusal_explicit_mask():
    mapping = _mapping()
    q, k, v = _qkv()
    bias = torch.zeros(1, 1, 8, 64, dtype=torch.float16)
    with pytest.raises(NotImplementedError) as ei:
        mapping(q, k, v, bias, 0.0, False, False, None)
    assert "[mask-mode]" in str(ei.value)


def test_refusal_gqa():
    mapping = _mapping()
    q, k, v = _qkv(heads=4, kv_heads=2)
    with pytest.raises(NotImplementedError) as ei:
        mapping(q, k, v, None, 0.0, True, False, None)
    assert "[gqa-group]" in str(ei.value)


def test_refusal_dropout():
    mapping = _mapping()
    q, k, v = _qkv()
    with pytest.raises(NotImplementedError):
        mapping(q, k, v, None, 0.1, False, False, None)


def test_refusal_debug_mask():
    mapping = _mapping()
    q, k, v = _qkv()
    with pytest.raises(NotImplementedError):
        mapping(q, k, v, None, 0.0, False, True, None)


def test_refusal_ragged_head_count():
    """H=3 over Hkv=2 is not an integral expansion — the gqa-group
    clause owns the refusal (3 // 2 = 1 would be a LIE if it admitted)."""
    mapping = _mapping()
    q, k, v = _qkv(heads=3, kv_heads=2)
    with pytest.raises(NotImplementedError) as ei:
        mapping(q, k, v, None, 0.0, True, False, None)
    assert "[gqa-group]" in str(ei.value)


# ---------------------------------------------------------------------------
# The custom op exists with a fake impl (compile-path)
# ---------------------------------------------------------------------------


def test_spyre_attention_op_registered():
    import torch_spyre  # noqa: F401 — device autoload

    assert hasattr(torch.ops.spyre, "attention")


def test_spyre_attention_fake():
    import torch_spyre  # noqa: F401

    q = torch.empty(8, 64, dtype=torch.float16, device="meta")
    k = torch.empty(64, 64, dtype=torch.float16, device="meta")
    v = torch.empty(64, 64, dtype=torch.float16, device="meta")
    out = torch.ops.spyre.attention(q, k, v, None)
    assert out.shape == (8, 64)
    assert out.dtype == torch.float16
