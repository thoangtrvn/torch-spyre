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

"""General convolution -> host im2col + matmul (decomposition gate +
refuse-loud envelope).

The hardware has no convolution datapath and the device-side gather is
vetoed twice over (L3 EAR strides are stick-units, so a 1-2 element
sliding-window hop is inexpressible; and the analyzer's dot-operand
provenance guard refuses gathered dot operands). A general
stride-1/dilation-1/groups-1 convolution therefore rewrites at the
PyTorch level to:

    cols = im2col(x, (kh, kw), padding=(ph, pw))    # (N, C*kh*kw, L)
    cols = pad(cols, K -> K_pad)                     # whole-stick K
    w    = pad(weight.view(O, K), K -> K_pad)        # zero-pad bill
    out  = w @ cols                                  # (N, O, L)
    out  = out.view(N, O, H_out, W_out)
    out  = out + bias.view(1, O, 1, 1)               # SEPARATE pointwise add

HOST-VS-DEVICE FALLBACK LEDGER (mandatory disposition, on record):
  - im2col (the unfold) runs on HOST. The device-side gather is
    hardware-vetoed (see module docstring) — there is no device im2col
    to fall back to, so this is a documented host disposition, not a
    silent CPU fallback of a device-capable op.
  - The K zero-pad (weight rows AND im2col columns) is structural
    zeros: numerically exact, billed as `K_pad - K` elements per weight
    row / im2col column (the im2col amplification term).
  - The mm rides the silicon-witnessed native-matmul route; the bias
    stays a SEPARATE pointwise add (the matmul_bias composite admission
    is witnessed only at the census geometry M=32/K=64/N=64).

ENVELOPE (fail-loud BEFORE emission, owned by the conv seam — never
silently inherited from the matmul route's downstream shape checks):
  - ks >= 9 refuses (single-ImpSync clean-K caps at ks=8; chunked-K is
    not landed) — every real 3x3 layer at C >= 64 lands here and must
    refuse NAMED. The bound lives in
    sentient_codegen.scheduler.conv_predicate (single source of truth);
    this decomposition delegates to it.
  - ks*ns > 127 refuses (14-bit L3 LBR single-kernel fit bound) — the
    matmul route would NOT refuse this (its FLAT single-buffer band
    serves up to ~192), which is exactly why the conv seam owns it.

Everything outside the gate (stride != 1, dilation != 1, groups != 1,
transposed, padding not 0 or kernel//2 per dim, conv1d/conv3d,
degenerate windows, non-contiguous layouts) raises
``errors.Unsupported`` — no CPU fallback, no extern kernel, no silent
contiguity assumption.

These tests are offline (CPU tensors + tracing). The end-to-end TTIR
routing receipt is captured separately; the known boundary is that
batch>1 offline TTIR capture dies at an Inductor KeyError (ZBLOCK) in
triton-spyre (third-party, repair lane in flight), so any capture
receipt below this file is batch-1 only.
"""

import pytest
import torch
import torch.nn.functional as F

from torch_spyre._inductor.decompositions import (
    conv2d_alias_decomp,
    conv2d_decomp,
)


def _require_codegen():
    """The envelope boundary delegates to sentient_codegen's conv
    predicate (single source of truth). Skip loud when it is not
    importable rather than false-red — but note the DECOMPOSITION
    itself refuses loud in that case (no silent skip in production)."""
    return pytest.importorskip(
        "sentient_codegen",
        reason="conv im2col envelope predicate lives in sentient_codegen "
        "(source env.sh / PYTHONPATH to the codegen tree)")


def _im2col_inputs(n=2, c=8, h=8, w=6, o=64, kh=3, kw=3,
                   dtype=torch.float32):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn(n, c, h, w, dtype=dtype)
    weight = torch.randn(o, c, kh, kw, dtype=dtype)
    return x, weight


class TestIm2colHappyPath:
    def test_matches_conv2d_reference_without_bias(self):
        _require_codegen()
        x, weight = _im2col_inputs()
        out = conv2d_decomp(
            x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        ref = F.conv2d(x, weight, None, stride=1, padding=0, dilation=1)
        torch.testing.assert_close(out, ref)

    def test_matches_conv2d_reference_with_bias(self):
        # Bias is a SEPARATE pointwise add — numerically it must still
        # match the fused reference exactly on fp32 CPU.
        _require_codegen()
        x, weight = _im2col_inputs()
        bias = torch.randn(weight.shape[0])
        out = conv2d_decomp(
            x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        ref = F.conv2d(x, weight, bias)
        torch.testing.assert_close(out, ref)

    def test_same_padding_matches_reference(self):
        # p = kernel//2 symmetric-valid padding (3x3 -> p=1), the
        # shape-class of the first planned conv HW witness.
        _require_codegen()
        x, weight = _im2col_inputs(h=6, w=5)
        bias = torch.randn(weight.shape[0])
        out = conv2d_decomp(
            x, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        ref = F.conv2d(x, weight, bias, padding=1)
        torch.testing.assert_close(out, ref)
        assert out.shape == (2, 64, 6, 5)

    def test_matches_reference_at_non_coincident_geometry(self):
        # Guard the coincidence trap: keep C, O, K, L and the spatial
        # extents pairwise distinct so a wrong operand order or a wrong
        # im2col column ordering cannot still type-check.
        _require_codegen()
        x, weight = _im2col_inputs(n=2, c=5, h=7, w=4, o=96, kh=3, kw=2)
        bias = torch.randn(96)
        out = conv2d_decomp(
            x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        ref = F.conv2d(x, weight, bias)
        torch.testing.assert_close(out, ref)

    def test_ragged_k_zero_pad_is_numerically_exact(self):
        # 3x3 over C=3: K=27 pads to a whole stick (32 at fp32 / 64 at
        # DL16). The structural zeros must not move a single result
        # bit — this is the ragged-K zero-pad-bill receipt.
        _require_codegen()
        x, weight = _im2col_inputs(n=2, c=3, h=5, w=5, o=64)
        bias = torch.randn(64)
        out = conv2d_decomp(
            x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        ref = F.conv2d(x, weight, bias)
        torch.testing.assert_close(out, ref)

    def test_dl16_dtype_routes_with_dl16_stick_width(self):
        # float16 input = DL16 on 1p0: elements_per_stick is 64, so the
        # same K=72 geometry pads differently (to 128, not 96) — the
        # envelope must be evaluated at the request's own stick width.
        # Tolerance note: the rewrite adds the bias as a SEPARATE fp16
        # pointwise add after the mm, while the fused CPU reference
        # folds it before the final rounding — a 1-ulp fp16 difference
        # is the expected bias-ordering effect, not a layout bug (the
        # no-bias form is bit-exact; the DL16 device-accumulation
        # ruling is a separate lane's deliverable).
        _require_codegen()
        x, weight = _im2col_inputs(dtype=torch.float16)
        bias = torch.randn(weight.shape[0], dtype=torch.float16)
        out = conv2d_decomp(
            x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        ref = F.conv2d(x, weight, bias)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        # The no-bias form is bit-exact (structural zero-pad proof at
        # the DL16 stick width).
        out_nb = conv2d_decomp(
            x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        ref_nb = F.conv2d(x, weight, None)
        torch.testing.assert_close(out_nb, ref_nb)

    def test_conv2d_overload_signature_matches(self):
        # aten.conv2d's 7-argument signature (no transposed /
        # output_padding) must reach the same gated rewrite.
        _require_codegen()
        x, weight = _im2col_inputs()
        out = conv2d_alias_decomp(x, weight, None, [1, 1], [0, 0], [1, 1], 1)
        ref = F.conv2d(x, weight, None)
        torch.testing.assert_close(out, ref)

    def test_5x5_kernel_same_padding_matches(self):
        # kh != kw == 5 with p = 5//2 = 2: the per-dim kernel//2 rule.
        _require_codegen()
        x, weight = _im2col_inputs(n=1, c=4, h=7, w=7, o=32, kh=5, kw=5)
        out = conv2d_decomp(
            x, weight, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1)
        ref = F.conv2d(x, weight, None, padding=2)
        torch.testing.assert_close(out, ref)


class TestEnvelopeRefusesLoud:
    """The conv seam's OWN refusal boundary — named, BEFORE any tensor
    is built or anything emitted. Delegated to
    sentient_codegen.scheduler.conv_predicate (single source of truth).
    """

    def test_3x3_c64_refuses_named_ks(self):
        # The red-first pin: 3x3 over C=64 at DL16 is K=576 = ks=9 —
        # every real 3x3 layer at C >= 64 lands here. Must refuse with
        # the offending ks named, before any binary emission.
        _require_codegen()
        x, weight = _im2col_inputs(c=64, o=64, dtype=torch.float16)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        msg = str(excinfo.value)
        assert "ks=9" in msg
        assert "chunked" in msg

    def test_lbr_product_refuses_named(self):
        # ks=3 (3x3 over C=8 at fp32, K=72 -> 96) with O=1408
        # (ns=44): ks*ns=132 > 127. The matmul route would NOT refuse
        # this (FLAT single-buffer band serves up to ~192) — the conv
        # seam must.
        _require_codegen()
        x, weight = _im2col_inputs(c=8, o=1408)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        msg = str(excinfo.value)
        assert "ks*ns" in msg or "ks·ns" in msg
        assert "132" in msg

    def test_envelope_refusal_fires_inside_the_trace(self):
        # The refusal must fire at decomposition time under torch.compile
        # too, not as a silent extern kernel or a downstream scheduler
        # error.
        _require_codegen()
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch_spyre._inductor.decompositions import (
            enable_spyre_decompositions,
        )

        def fn(x, w):
            return F.conv2d(x, w, None)

        x, weight = _im2col_inputs(c=64, o=64)
        table = {}
        with enable_spyre_decompositions(table):
            with pytest.raises(Exception, match="does not support"):
                make_fx(fn, decomposition_table=table)(x, weight)

    def test_in_envelope_3x3_does_not_refuse(self):
        # Positive control for the two refusals above: the smallest
        # in-envelope 3x3 (C=8 at fp32: ks=3, ns=2, ks*ns=6) admits.
        _require_codegen()
        x, weight = _im2col_inputs()
        out = conv2d_decomp(
            x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        assert out.shape == (2, 64, 6, 4)


class TestGateRefusesLoud:
    """Every un-gated variant raises errors.Unsupported — the rewrite
    must never silently catch a conv form it cannot express."""

    def _refuse(self, weight=None, x=None, **kwargs):
        _require_codegen()
        if x is None and weight is None:
            x, weight = _im2col_inputs()
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

    def test_dilation_2_refused(self):
        msg = self._refuse(dilation=[1, 2])
        assert "dilation" in msg

    def test_groups_refused(self):
        msg = self._refuse(groups=2)
        assert "group" in msg

    def test_transposed_refused(self):
        msg = self._refuse(transposed=True)
        assert "transposed" in msg

    def test_asymmetric_padding_refused(self):
        # p=(2,2) with a 3x3 kernel: 2 != 3//2 — outside the
        # {0, kernel//2} per-dim rule.
        msg = self._refuse(padding=[2, 2])
        assert "padding" in msg

    def test_conv3d_refused(self):
        # aten.convolution is the funnel for conv1d/conv2d/conv3d — a
        # 5D input must not sneak through the im2col gate.
        _require_codegen()
        x = torch.randn(2, 8, 4, 4, 4)
        weight = torch.randn(64, 8, 3, 3, 3)
        with pytest.raises(Exception, match="does not support"):
            conv2d_decomp(
                x, weight, None, [1] * 3, [0] * 3, [1] * 3, False,
                [0] * 3, 1)

    def test_degenerate_window_refused(self):
        # 3x3 kernel over a 2x2 input with p=0: the window does not
        # fit even once — refuse named instead of letting unfold error
        # opaquely downstream.
        x, weight = _im2col_inputs(h=2, w=2)
        msg = self._refuse(x=x, weight=weight)
        assert "window" in msg or "extent" in msg

    def test_channel_mismatch_refused(self):
        x, _ = _im2col_inputs(c=8)
        weight = torch.randn(64, 16, 3, 3)
        msg = self._refuse(x=x, weight=weight)
        assert "channel" in msg

    def test_non_contiguous_input_refused_with_strides_named(self):
        x, weight = _im2col_inputs(h=6, w=6)
        x = x.permute(0, 1, 3, 2)  # shape-preserving (h==w), non-contiguous
        msg = self._refuse(x=x, weight=weight)
        assert "contiguous" in msg
        assert "stride" in msg

    def test_non_contiguous_weight_refused(self):
        x, _ = _im2col_inputs()
        weight = torch.randn(64, 8, 3, 3).transpose(0, 1)
        self._refuse(x=x, weight=weight)


class TestFoldConventionParity:
    """The T-CN-1 gauntlet-r1 fold conventions (ATen len-1 list
    canonicalization, rank-first refusals, dtype/bias gates) apply to
    the im2col core too — same funnel, same contract."""

    def test_conv3d_refusal_names_rank_not_stride(self):
        # A 5D input must be told it is the wrong rank, not get a
        # stride refusal naming the rank-3 list it legitimately carries
        # (the fold's rank-first rule).
        _require_codegen()
        x = torch.randn(2, 8, 4, 4, 4)
        weight = torch.randn(64, 8, 3, 3, 3)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, None, [1] * 3, [0] * 3, [1] * 3, False,
                [0] * 3, 1)
        assert "4D NCHW" in str(excinfo.value)

    def test_traced_raw_aten_len1_shorthand_admits(self):
        # A traced RAW op call delivers the parameter lists un-normalized
        # (eager F.conv2d normalizes via CompositeImplicitAutograd): the
        # ATen len-1 shorthand ([1]/[0]/[1]) must canonicalize to the
        # spatial rank and admit, not false-refuse.
        _require_codegen()
        x, weight = _im2col_inputs()
        out = torch.ops.aten.conv2d.default(
            x, weight, None, [1], [0], [1], 1)
        ref = F.conv2d(x, weight, None)
        torch.testing.assert_close(out, ref)

    def test_wrong_list_length_refuses_named(self):
        # A 4D input carrying a rank-3 parameter list: the rank is fine,
        # the list length is not — refuse naming the observed length.
        # (The mirror case — a 5D input, rank-first naming — is pinned
        # by test_conv3d_refusal_names_rank_not_stride above.)
        _require_codegen()
        x, weight = _im2col_inputs()
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, None, [1, 1, 1], [0, 0], [1, 1], False,
                [0, 0], 1)
        assert "length" in str(excinfo.value)

    def test_mixed_weight_dtype_refuses_naming_both(self):
        _require_codegen()
        x, weight = _im2col_inputs()
        w64 = weight.double()
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, w64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        msg = str(excinfo.value)
        assert "dtype" in msg
        assert "float64" in msg and "float32" in msg

    def test_mixed_bias_dtype_refuses_naming_both(self):
        _require_codegen()
        x, weight = _im2col_inputs()
        bias = torch.randn(weight.shape[0], dtype=torch.float64)
        with pytest.raises(Exception, match="does not support"):
            conv2d_decomp(
                x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

    def test_non_1d_bias_refuses_named(self):
        _require_codegen()
        x, weight = _im2col_inputs()
        bias = torch.randn(weight.shape[0], 1)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        assert "1-dimensional" in str(excinfo.value)

    def test_wrong_length_bias_refuses_named(self):
        _require_codegen()
        x, weight = _im2col_inputs()
        bias = torch.randn(weight.shape[0] + 1)
        with pytest.raises(Exception, match="does not support") as excinfo:
            conv2d_decomp(
                x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        assert "length" in str(excinfo.value)


class TestTraceReceipt:
    def test_make_fx_rewrites_conv2d_to_im2col_and_matmul(self):
        # The registration arm's whole point: under
        # enable_spyre_decompositions a traced general conv is replaced
        # by the im2col (+ pad) + matmul (+ optional add) chain, with
        # NO conv node surviving.
        _require_codegen()
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch_spyre._inductor.decompositions import (
            enable_spyre_decompositions,
        )

        def fn(x, w, b):
            return F.conv2d(x, w, b)

        x, weight = _im2col_inputs()
        bias = torch.randn(weight.shape[0])
        table = {}
        with enable_spyre_decompositions(table):
            gm = make_fx(fn, decomposition_table=table)(x, weight, bias)
        graph_ops = [str(node.target) for node in gm.graph.nodes]
        assert not any("conv" in op for op in graph_ops), graph_ops
        assert any("im2col" in op or "unfold" in op for op in graph_ops), (
            graph_ops)
        assert any("matmul" in op or "mm" in op for op in graph_ops), (
            graph_ops)
        assert any("add" in op for op in graph_ops), graph_ops
        # The ragged-K zero-pad shows up as the structural-zero block
        # (new_zeros + slice/copy — backend-neutral by design, see the
        # decomposition's comment on why not constant_pad_nd).
        assert any("new_zeros" in op for op in graph_ops), graph_ops

    def test_make_fx_refuses_un_gated_variant(self):
        # The refusal must fire INSIDE the trace too.
        _require_codegen()
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch_spyre._inductor.decompositions import (
            enable_spyre_decompositions,
        )

        def fn(x, w):
            return F.conv2d(x, w, None, stride=2)

        x, weight = _im2col_inputs()
        table = {}
        with enable_spyre_decompositions(table):
            with pytest.raises(Exception, match="does not support"):
                make_fx(fn, decomposition_table=table)(x, weight)


class TestOneByOneRoutingUnchanged:
    def test_1x1_still_routes_to_the_alias_not_im2col(self):
        # The 1x1 form keeps its cheaper direct-matmul alias — no
        # unfold node may appear for it.
        _require_codegen()
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch_spyre._inductor.decompositions import (
            enable_spyre_decompositions,
        )

        def fn(x, w):
            return F.conv2d(x, w, None)

        x = torch.randn(2, 64, 8, 8)
        weight = torch.randn(64, 64, 1, 1)
        table = {}
        with enable_spyre_decompositions(table):
            gm = make_fx(fn, decomposition_table=table)(x, weight)
        graph_ops = [str(node.target) for node in gm.graph.nodes]
        assert not any("im2col" in op or "unfold" in op for op in graph_ops), (
            graph_ops)
