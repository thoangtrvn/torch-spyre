# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""1x1 convolution -> matmul alias (decomposition gate + refuse-loud).

A 1x1 convolution with stride 1, no padding, no dilation and a single
group is a pure per-spatial-position channel mix — i.e. exactly a
matmul over the channel dimension. The hardware has no convolution
datapath; every conv must bottom out in the (silicon-witnessed) matmul
route. This decomposition rewrites the gated form at the PyTorch level
so Inductor never emits a conv extern kernel for it:

    out = w.view(O, C) @ x.view(N, C, H*W)       # (N, O, H*W)
    out = out.view(N, O, H, W)
    out = out + bias.view(1, O, 1, 1)            # SEPARATE pointwise add

Everything outside the gate (strided / padded / dilated / grouped /
transposed 1x1, any k>1 conv, conv1d/conv3d, misaligned or
non-contiguous layouts) must raise ``errors.Unsupported`` — no CPU
fallback, no extern kernel, no silent contiguity assumption.

Layout contract (refuse-loud, never assumed):
  - NCHW stick alignment: the channel extent C must be a multiple of
    the stick size in elements (128B / element_size — 64 elements at
    DL16/fp16, 32 at fp32). A sub-stick C would place the mm's K
    dimension off stick boundaries.
  - Contiguity: ``input`` and ``weight`` must be dense-contiguous
    NCHW/OIHW. Reachable non-contiguous inputs (slices, permutes) are
    REFUSED with the observed strides in the message; the decomposition
    never calls ``.contiguous()`` (which would silently pay a copy and
    hide the layout from the compiler).

These tests are offline (CPU tensors + tracing): the decomposition is
pure PyTorch ops, so its numerics and its gate are testable without
the Spyre runtime. The end-to-end TTIR routing receipt (the rewritten
graph classifying as matmul through the native-matmul route) is
captured separately against the local generation stack.
"""

import pytest
import torch
import torch.nn.functional as F

from torch_spyre._inductor.decompositions import (
    conv2d_alias_decomp,
    conv2d_decomp,
    spyre_decompositions,
)


def _aligned_inputs(n=2, c=64, h=8, w=8, o=64, dtype=torch.float32):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn(n, c, h, w, dtype=dtype)
    weight = torch.randn(o, c, 1, 1, dtype=dtype)
    return x, weight


class TestOneByOneHappyPath:
    def test_matches_conv2d_reference_without_bias(self):
        x, weight = _aligned_inputs()
        out = conv2d_decomp(
            x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        ref = F.conv2d(x, weight, None, stride=1, padding=0, dilation=1)
        torch.testing.assert_close(out, ref)

    def test_matches_conv2d_reference_with_bias(self):
        # Bias is a SEPARATE pointwise add (the default D-disposition) —
        # numerically it must still match the fused reference exactly on
        # fp32 CPU.
        x, weight = _aligned_inputs()
        bias = torch.randn(weight.shape[0])
        out = conv2d_decomp(
            x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        ref = F.conv2d(x, weight, bias, stride=1, padding=0, dilation=1)
        torch.testing.assert_close(out, ref)

    def test_conv2d_overload_signature_matches(self):
        # aten.conv2d calls with the 7-argument signature (no
        # transposed/output_padding); both overloads must reach the same
        # gated rewrite.
        x, weight = _aligned_inputs()
        out = conv2d_alias_decomp(x, weight, None, [1, 1], [0, 0], [1, 1], 1)
        ref = F.conv2d(x, weight, None)
        torch.testing.assert_close(out, ref)

    def test_matches_reference_at_non_coincident_spatial(self):
        # Guard the coincidence trap: at h=w=8, C=64 the spatial extent
        # equals the channel extent (P == C == 64) and a wrong operand
        # order can still type-check. This shape keeps them distinct.
        x, weight = _aligned_inputs(n=2, c=64, h=4, w=6, o=96)
        bias = torch.randn(96)
        out = conv2d_decomp(
            x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        ref = F.conv2d(x, weight, bias)
        torch.testing.assert_close(out, ref)

    def test_output_shape_is_nchw(self):
        x, weight = _aligned_inputs(n=3, c=64, h=4, w=6, o=128)
        out = conv2d_decomp(
            x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        assert out.shape == (3, 128, 4, 6)

    def test_len1_aten_parameter_lists_accepted(self):
        # A traced RAW op call (torch.ops.aten.conv2d(...)) delivers the
        # parameters exactly as written — len-1 lists like [1]/[0]/[1]
        # reach the decomposition un-normalized. Eager F.conv2d normalizes
        # via CompositeImplicitAutograd, but the traced path must not
        # false-refuse on the ATen shorthand: [1] == [1, 1] at rank 2.
        x, weight = _aligned_inputs()
        out = conv2d_decomp(x, weight, None, [1], [0], [1], False, [0], 1)
        ref = F.conv2d(x, weight, None)
        torch.testing.assert_close(out, ref)

    def test_traced_raw_aten_conv2d_len1_lists_rewrite_to_matmul(self):
        # The delivery path for the len-1 lists above, verified end to
        # end: make_fx on a function calling the RAW aten op (not the
        # Python F.conv2d wrapper) under enable_spyre_decompositions must
        # rewrite to matmul with no conv node surviving.
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch_spyre._inductor.decompositions import (
            enable_spyre_decompositions,
        )

        def fn(x, w):
            return torch.ops.aten.conv2d(x, w, None, [1], [0], [1], 1)

        x, weight = _aligned_inputs()
        table = {}
        with enable_spyre_decompositions(table):
            gm = make_fx(fn, decomposition_table=table)(x, weight)
        graph_ops = [str(node.target) for node in gm.graph.nodes]
        assert not any("conv" in op for op in graph_ops), graph_ops
        assert any("matmul" in op or "mm" in op for op in graph_ops), (
            graph_ops)


class TestGateRefusesLoud:
    """Every un-gated variant raises errors.Unsupported — the alias must
    never silently catch a conv form it cannot express as this matmul."""

    def _refuse(self, **kwargs):
        x, weight = _aligned_inputs()
        args = dict(
            stride=[1, 1], padding=[0, 0], dilation=[1, 1],
            transposed=False, output_padding=[0, 0], groups=1,
        )
        args.update(kwargs)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(x, weight, None, **args)
        return str(excinfo.value)

    def test_stride_2_refused(self):
        msg = self._refuse(stride=[2, 1])
        assert "stride" in msg

    def test_padding_refused(self):
        msg = self._refuse(padding=[1, 0])
        assert "padding" in msg

    def test_dilation_refused(self):
        msg = self._refuse(dilation=[1, 2])
        assert "dilation" in msg

    def test_groups_refused(self):
        msg = self._refuse(groups=2)
        assert "groups" in msg

    def test_transposed_refused(self):
        msg = self._refuse(transposed=True)
        assert "transposed" in msg

    def test_k3_kernel_refused(self):
        # Post im2col-v0: a 3x3 kernel routes through the im2col core,
        # and at C=64 (fp32, K=576 -> ks=18) it refuses on the conv
        # ENVELOPE (ks >= 9 needs chunked-K, not landed) — still
        # refuse-loud, now named at the envelope boundary instead of
        # the 1x1 alias's kernel guard. In-envelope 3x3 shapes are
        # covered in test_conv2d_im2col.py.
        x, _ = _aligned_inputs()
        weight = torch.randn(64, 64, 3, 3)
        with pytest.raises(Exception, match="does not support"):
            conv2d_decomp(
                x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

    def test_conv3d_refused(self):
        # aten.convolution is the funnel for conv1d/conv2d/conv3d — a
        # 5D input must not sneak through the 1x1 gate.
        x = torch.randn(2, 64, 4, 4, 4)
        weight = torch.randn(64, 64, 1, 1, 1)
        with pytest.raises(Exception, match="does not support"):
            conv2d_decomp(
                x, weight, None, [1] * 3, [0] * 3, [1] * 3, False,
                [0] * 3, 1)

    def test_sub_stick_channels_refused(self):
        # NCHW stick alignment: C=48 at fp32 (stick = 32 elements) is
        # aligned, C=24 is not — the misaligned form must refuse loud.
        x = torch.randn(2, 24, 8, 8)
        weight = torch.randn(64, 24, 1, 1)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        assert "stick" in str(excinfo.value)

    def test_aligned_sub_multiple_stick_channels_pass(self):
        # C=32 at fp32 is exactly one stick — the alignment boundary the
        # gate encodes (positive control for the refusal above).
        x = torch.randn(2, 32, 8, 8)
        weight = torch.randn(64, 32, 1, 1)
        out = conv2d_decomp(
            x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        ref = F.conv2d(x, weight, None)
        torch.testing.assert_close(out, ref)

    def test_non_contiguous_input_refused_with_strides_named(self):
        x, weight = _aligned_inputs()
        x = x.permute(0, 1, 3, 2)  # shape-preserving (h==w), non-contiguous
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        assert "contiguous" in str(excinfo.value)
        assert "stride" in str(excinfo.value)

    def test_non_contiguous_weight_refused(self):
        x, _ = _aligned_inputs()
        weight = torch.randn(64, 64, 1, 1).transpose(0, 1)
        with pytest.raises(Exception, match="does not support"):
            conv2d_decomp(
                x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

    def test_conv3d_refusal_names_rank_not_stride(self):
        # A 5D (conv3d-shaped) input carrying len-3 parameter lists must
        # get the RANK refusal ("only 4D NCHW"), not a stride refusal
        # naming [1, 1, 1] — the rank check runs before the parameter
        # equality gate so conv1d/conv3d callers are told what is
        # actually wrong.
        x = torch.randn(2, 64, 4, 4, 4)
        weight = torch.randn(64, 64, 1, 1, 1)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, None, [1] * 3, [0] * 3, [1] * 3, False,
                [0] * 3, 1)
        assert "4D NCHW" in str(excinfo.value)
        assert "stride" not in str(excinfo.value)

    def test_wrong_length_parameter_list_refused(self):
        # A len-3 list against a 4D input is malformed for conv2d (the
        # ATen shorthand covers only len-1); it must refuse loud naming
        # the observed length, never index-error.
        x, weight = _aligned_inputs()
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, None, [1, 1, 1], [0, 0], [1, 1], False,
                [0, 0], 1)
        assert "stride" in str(excinfo.value)
        assert "length" in str(excinfo.value)

    def test_bias_dtype_mismatch_refused(self):
        # fp32 input/weight with an f64 bias silently promotes under
        # plain tensor ops (the reference conv2d refuses it); the alias
        # must refuse loud naming both dtypes instead.
        x, weight = _aligned_inputs()
        bias = torch.randn(weight.shape[0], dtype=torch.float64)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        msg = str(excinfo.value)
        assert "bias" in msg and "dtype" in msg
        assert "float32" in msg and "float64" in msg

    def test_weight_dtype_mismatch_refused(self):
        # Mixed input/weight dtypes currently die as a bare RuntimeError
        # inside the matmul; the gate must refuse typed, naming both.
        x, _ = _aligned_inputs(dtype=torch.float32)
        weight = torch.randn(64, 64, 1, 1, dtype=torch.float16)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        msg = str(excinfo.value)
        assert "dtype" in msg
        assert "float32" in msg and "float16" in msg

    def test_bfloat16_refused_as_1p5_format(self):
        # Dtype IDENTITY, not just uniformity: a bfloat16 1x1 conv is
        # internally uniform so the equality gates pass, and torch_spyre
        # maps bf16 to DataFormats.BFLOAT16 (torch_spyre/_C.py:113) —
        # the 1p5 format. On a 1p0 target that is the BF16-bits-read-as-
        # DL16 silent-wrong class, so the alias must refuse bfloat16
        # NAMED, following the softmax seam precedent
        # (torch_spyre/ops/softmax.py: "bfloat16 is 1p5"). Nothing
        # downstream independently refuses bf16 — the mapping is the
        # exposure — so the gate at this layer is load-bearing.
        x, weight = _aligned_inputs(dtype=torch.bfloat16)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        msg = str(excinfo.value)
        assert "bfloat16" in msg
        assert "1p5" in msg

    def test_non_1d_bias_refused(self):
        # A 2D bias of matching numel ((O, 1)) is leniently accepted by
        # the .view() but refused by the conv2d reference ("Expected bias
        # to be 1-dimensional"); the alias must match the reference —
        # refuse, naming the observed shape.
        x, weight = _aligned_inputs()
        bias = torch.randn(weight.shape[0], 1)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        assert "1-dimensional" in str(excinfo.value)
        assert str(tuple(bias.shape)) in str(excinfo.value)

    def test_wrong_length_bias_refused(self):
        # A 1D bias of the wrong length currently dies as a bare
        # RuntimeError inside .view(); it must refuse typed, naming the
        # observed and expected lengths.
        x, weight = _aligned_inputs(o=64)
        bias = torch.randn(65)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        msg = str(excinfo.value)
        assert "bias" in msg
        assert "65" in msg and "64" in msg


class TestRegistryWiring:
    def test_both_conv_overloads_registered(self):
        # The compile path funnels conv2d -> convolution, but eager and
        # pre-decomposition paths can present either overload; both
        # must be wired to the same gated implementation.
        assert (
            spyre_decompositions[torch.ops.aten.convolution.default]
            is conv2d_decomp
        )
        assert (
            spyre_decompositions[torch.ops.aten.conv2d.default]
            is conv2d_alias_decomp
        )

    def test_make_fx_rewrites_conv2d_to_matmul(self):
        # The registration arm `decomp.register_decomposition(...,
        # spyre_decompositions)` targets exactly this: under
        # enable_spyre_decompositions a traced conv2d is replaced by the
        # matmul (+ optional add) chain, with NO conv node surviving.
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch_spyre._inductor.decompositions import (
            enable_spyre_decompositions,
        )

        def fn(x, w, b):
            return F.conv2d(x, w, b)

        x, weight = _aligned_inputs()
        bias = torch.randn(weight.shape[0])
        table = {}
        with enable_spyre_decompositions(table):
            gm = make_fx(fn, decomposition_table=table)(x, weight, bias)
        graph_ops = [str(node.target) for node in gm.graph.nodes]
        assert not any("conv" in op for op in graph_ops), graph_ops
        assert any("matmul" in op or "mm" in op for op in graph_ops), (
            graph_ops)
        assert any("add" in op for op in graph_ops), graph_ops

    def test_make_fx_refuses_un_gated_variant(self):
        # The refusal must fire INSIDE the trace too (a strided 1x1 conv
        # under torch.compile fails loud at decomposition time, not as a
        # silent extern kernel downstream).
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch_spyre._inductor.decompositions import (
            enable_spyre_decompositions,
        )

        def fn(x, w):
            return F.conv2d(x, w, None, stride=2)

        x, weight = _aligned_inputs()
        table = {}
        with enable_spyre_decompositions(table):
            with pytest.raises(Exception, match="does not support"):
                make_fx(fn, decomposition_table=table)(x, weight)
