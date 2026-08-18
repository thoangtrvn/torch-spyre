# Copyright 2025 The Torch-Spyre Authors.
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

The fix re-introduces the offset in-graph in lower_spyre_from_d2d
(torch_spyre/_inductor/lowering.py) via a ReinterpretView, so the offset lands
in the coordinate that superdsc bakes into the SDSC binary.

The transpose / permute / strided cases below deliberately exercise
NON-contiguous views. They probe whether re-injecting only the scalar
storage_offset onto the input's own (size, stride) layout is sufficient
("Option 2"), or whether the full view (size + stride + offset) must be
reconstructed from the base tensor ("Option 1"). If any strided case fails on
hardware while the contiguous cases pass, Option 2 is insufficient.
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

    def test_wide_row_slice_multi_stick(self):
        """dim-0 (row) narrow of a WIDE tensor (>1 stick per row).

        Regression for the compile-crash sibling of issue #3264:
        ``narrow(0, 2, 2)`` of an [8, 4096] fp16 tensor (4096 = 64 sticks/row)
        reaches codegen as a [2, 4096] VIEW whose interior device dim extent
        already equals the iteration size, so superdsc's
        ``dev_dim_size > it_dim_size`` gate never fired and the re-injected flat
        storage_offset (flat 2*4096=8192) never landed as a per-dim offset. The
        interior coordinate ``d0 + 2`` then addressed a tile outside the trimmed
        dim, aborting the dxp_standalone compile with "There must be at least
        one valid candidate" (L3DlOpsScheduler.cpp:1075). VALUE-checked, not
        just non-crash: rows [2:4) must match the source rows."""
        x = torch.arange(8 * 4096, dtype=DTYPE, device=DEVICE).reshape(8, 4096)
        out = x.narrow(0, 2, 2).clone()  # rows [2:4), each spanning 64 sticks
        torch.testing.assert_close(out.cpu(), x.cpu()[2:4])

    def test_wide_row_slice_multi_stick_offsets(self):
        """dim-0 narrow of a WIDE tensor at multiple nonzero offsets.

        Companion to test_wide_row_slice_multi_stick that exercises the
        offsets the first fix attempt missed (off=4 and off=6). The
        allreduce_2d_compose path stacks SRC clones at off=4/6, so a
        single-offset test did not cover it. The proven
        ``dev_dim_size > it_dim_size`` gate (superdsc.py) must bake the
        per-dim device offset (off * 64 device-memory stride) for each. The
        SRC arg carries the PARENT interior device extent (8), inherited
        unchanged from the [8,4096] parent through the eager view, so the
        gate fires for every offset. VALUE-checked."""
        x = torch.arange(8 * 4096, dtype=DTYPE, device=DEVICE).reshape(8, 4096)
        for off in (4, 6):
            out = x.narrow(0, off, 2).clone()
            torch.testing.assert_close(
                out.cpu(),
                x.cpu()[off : off + 2],
                msg=f"wide row slice off={off}",
            )

    def test_wide_narrowed_dst_multi_stick(self):
        """Write a wide multi-stick block into a narrowed-DST slice.

        The reassembly step of allreduce_2d_compose does
        ``tensor.narrow(0, k, len).copy_(block)`` where both operands span
        >1 stick per row. This exercises the DST side of copy_from_d2d with a
        nonzero dst_off on a wide tensor: a [2,4096] block is written into
        rows [4:6) of an [8,4096] tensor. Rows outside [4:6) must be
        untouched. VALUE-checked."""
        dst = torch.full((8, 4096), -1.0, dtype=DTYPE, device=DEVICE)
        block = torch.arange(2 * 4096, dtype=DTYPE, device=DEVICE).reshape(2, 4096)
        dst.narrow(0, 4, 2).copy_(block)
        out = dst.cpu()
        torch.testing.assert_close(out[4:6], block.cpu())
        torch.testing.assert_close(
            out[0:4], torch.full((4, 4096), -1.0, dtype=DTYPE)
        )
        torch.testing.assert_close(
            out[6:8], torch.full((2, 4096), -1.0, dtype=DTYPE)
        )

    @unittest.expectedFailure
    def test_column_slice_inner_offset(self):
        """Offset along the last (stick) dim: narrow columns at an offset.

        KNOWN LIMITATION (silent wrong data). A storage_offset that falls in
        the innermost / stick dimension is not correctly baked: the fix
        re-introduces the flat offset onto the layout, but superdsc decomposes
        per-dim offsets against device_size and does not split a stick-dim
        offset correctly, so the read is off by the stick-dim component.
        Tracked for the follow-up PR (stick-dim offset handling), distinct from
        the row-offset bug fixed here."""
        x = torch.arange(2 * 128, dtype=DTYPE, device=DEVICE).reshape(2, 128)
        # columns [64:128) -> nonzero offset within a row
        out = x.narrow(1, 64, 64).clone()
        torch.testing.assert_close(out.cpu(), x.cpu()[:, 64:128])


class TestCopyFromD2DStridedViews(unittest.TestCase):
    """Non-contiguous views: transpose / permute / step slices / select.

    permute and select work (the offset fix carries them). transpose and
    stepped-slice cases are marked expectedFailure: they fail in the Spyre
    restickify layout pass ("no mechanism to resolve stick incompatibility" /
    "scatter elements from one stick to multiple sticks"), NOT in offset
    handling. Verified to fail identically on the pre-fix baseline (offset==0
    transpose reduces lower_spyre_from_d2d to the original mutate_to), so these
    are pre-existing backend limitations, not regressions from this fix, and
    reconstructing size+stride explicitly (Option 1) would not resolve them.
    Tracked for the follow-up PR.
    """

    @unittest.expectedFailure
    def test_transpose_clone(self):
        x = torch.arange(4 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 64)
        out = x.t().clone()  # (64, 4), non-contiguous
        torch.testing.assert_close(out.cpu(), x.cpu().t())

    @unittest.expectedFailure
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

    @unittest.expectedFailure
    def test_stepped_slice_clone(self):
        """Strided (step>1) slice — non-unit stride plus offset."""
        x = torch.arange(8 * 64, dtype=DTYPE, device=DEVICE).reshape(8, 64)
        out = x[1::2].clone()  # rows 1,3,5,7 ; offset=64, stride[0]=128
        torch.testing.assert_close(out.cpu(), x.cpu()[1::2])

    @unittest.expectedFailure
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
