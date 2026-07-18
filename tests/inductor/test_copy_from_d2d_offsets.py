# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for device-to-device copies of offset / strided views.

Guards against the silent-wrong-data bug where copying different slices of a
Spyre tensor D2D more than once returned the *first* call's data. The slice
position lives only in the tensor's storage_offset, and a graph input's
storage_offset is dropped by the Inductor backend (its FixedLayout.offset is
0 and SpyreTensorLayout has no offset field), so the compiled kernel bound the
storage base pointer and read from element 0.

The fix has two parts:
  * Row-major, stick-aligned operands (the common row-offset path) stay on
    device: lower_spyre_from_d2d re-introduces the dropped storage_offset
    in-graph via a ReinterpretView so it lands in the coordinate superdsc
    bakes into the SDSC binary.
  * Any other view — non-row-major (transpose / permute-of-stick / stepped
    slice) or an offset that is sub-stick / not stick-aligned — is routed by
    spyre__copy_from (eager.py) to a CPU round-trip, which honors
    storage_offset via the H2D/D2H DMA path. Correct for arbitrary strided
    views, at a performance cost for those exotic copies.
"""

import unittest

import torch

import torch_spyre  # noqa: F401


DEVICE = "spyre"
DTYPE = torch.float16


class TestCopyFromD2DContiguousOffsets(unittest.TestCase):
    """Contiguous slices at varying offsets — the core reproducer."""

    def test_multi_offset_clone(self):
        x = torch.arange(4 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 64)
        a = x.narrow(0, 0, 1).clone()
        b = x.narrow(0, 2, 1).clone()
        torch.testing.assert_close(a.cpu(), x.cpu()[0:1])
        torch.testing.assert_close(b.cpu(), x.cpu()[2:3])

    def test_loop_varying_offsets(self):
        x = torch.arange(8 * 64, dtype=DTYPE, device=DEVICE).reshape(8, 64)
        for r in [0, 2, 4, 6, 7]:
            out = x.narrow(0, r, 1).clone()
            torch.testing.assert_close(
                out.cpu(),
                x.cpu()[r : r + 1],
                msg=f"row {r}: got {out.cpu()[0, 0].item()}",
            )

    def test_revisit_offset(self):
        x = torch.arange(6 * 64, dtype=DTYPE, device=DEVICE).reshape(6, 64)
        for r in [1, 3, 1, 5, 3]:
            out = x.narrow(0, r, 1).clone()
            torch.testing.assert_close(out.cpu(), x.cpu()[r : r + 1])

    def test_copy_into_sliced_dst(self):
        """dst is itself a narrow (nonzero dst storage_offset)."""
        x = torch.arange(4 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 64)
        dst = torch.full((4, 64), -1.0, dtype=DTYPE, device=DEVICE)
        dst.narrow(0, 0, 1).copy_(x.narrow(0, 0, 1))
        dst.narrow(0, 2, 1).copy_(x.narrow(0, 3, 1))
        out = dst.cpu()
        torch.testing.assert_close(out[0:1], x.cpu()[0:1])
        torch.testing.assert_close(out[2:3], x.cpu()[3:4])
        torch.testing.assert_close(out[1:2], torch.full((1, 64), -1.0, dtype=DTYPE))
        torch.testing.assert_close(out[3:4], torch.full((1, 64), -1.0, dtype=DTYPE))

    def test_column_slice_inner_offset(self):
        """Offset along the last (stick) dim: narrow columns at an offset.

        A stick-dim offset cannot be baked as a sub-stick base offset (the
        device layout has no such field), so this routes to the CPU
        round-trip fallback, which honors storage_offset."""
        x = torch.arange(2 * 128, dtype=DTYPE, device=DEVICE).reshape(2, 128)
        # columns [64:128) -> nonzero offset within a row
        out = x.narrow(1, 64, 64).clone()
        torch.testing.assert_close(out.cpu(), x.cpu()[:, 64:128])

    def test_partial_stick_offset(self):
        """Offset in the stick dim that is NOT a whole-stick multiple."""
        x = torch.arange(2 * 128, dtype=DTYPE, device=DEVICE).reshape(2, 128)
        out = x.narrow(1, 32, 64).clone()  # columns [32:96), sub-stick offset
        torch.testing.assert_close(out.cpu(), x.cpu()[:, 32:96])


class TestCopyFromD2DStridedViews(unittest.TestCase):
    """Non-contiguous / non-stick-aligned views: transpose / permute / stepped
    slice / select / stick-dim offset.

    These cannot be represented by the compiled d2d kernel, so spyre__copy_from
    routes them to a CPU round-trip (see _d2d_kernel_can_bake in eager.py).
    They emit a FallbackWarning but must produce correct data.
    """

    def test_transpose_clone(self):
        x = torch.arange(4 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 64)
        out = x.t().clone()  # (64, 4), non-contiguous
        torch.testing.assert_close(out.cpu(), x.cpu().t())

    def test_transpose_then_offset_clone(self):
        """Transpose AND a nonzero offset along the transposed dim."""
        x = torch.arange(4 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 64)
        xt = x.t()  # (64, 4)
        out = xt.narrow(1, 2, 2).clone()  # rows of original [2:4], transposed
        torch.testing.assert_close(out.cpu(), x.cpu().t()[:, 2:4])

    def test_permute_clone(self):
        x = torch.arange(2 * 3 * 64, dtype=DTYPE, device=DEVICE).reshape(2, 3, 64)
        out = x.permute(1, 0, 2).clone()  # (3, 2, 64)
        torch.testing.assert_close(out.cpu(), x.cpu().permute(1, 0, 2))

    def test_permute_with_offset_clone(self):
        x = torch.arange(4 * 3 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 3, 64)
        v = x.permute(1, 0, 2).narrow(1, 1, 2)  # offset along a permuted dim
        out = v.clone()
        torch.testing.assert_close(out.cpu(), x.cpu().permute(1, 0, 2)[:, 1:3])

    def test_select_clone(self):
        """select drops a dim and introduces an offset."""
        x = torch.arange(4 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 64)
        out = x.select(0, 2).clone()  # row 2 as 1-D (64,), storage_offset=128
        torch.testing.assert_close(out.cpu(), x.cpu()[2])

    def test_stepped_slice_clone(self):
        """Strided (step>1) slice — non-unit stride plus offset."""
        x = torch.arange(8 * 64, dtype=DTYPE, device=DEVICE).reshape(8, 64)
        out = x[1::2].clone()  # rows 1,3,5,7 ; offset=64, stride[0]=128
        torch.testing.assert_close(out.cpu(), x.cpu()[1::2])

    def test_transpose_varying_offsets_loop(self):
        """Multiple distinct offsets on a transposed view in one process."""
        x = torch.arange(8 * 64, dtype=DTYPE, device=DEVICE).reshape(8, 64)
        xt = x.t()  # (64, 8)
        for c in [0, 2, 5, 7]:
            out = xt.narrow(1, c, 1).clone()
            torch.testing.assert_close(
                out.cpu(),
                x.cpu().t()[:, c : c + 1],
                msg=f"transpose col {c}",
            )


if __name__ == "__main__":
    unittest.main()
