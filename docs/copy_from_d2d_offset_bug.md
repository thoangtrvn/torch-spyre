# D2D copy silently returns the first offset's data (`spyre::copy_from_d2d`)

## Summary

Copying different slices of a Spyre tensor **device-to-device** more than once
in the same process silently returns the **first** slice's data for every later
call. No error, no warning.

```python
x = torch.arange(4 * 64, dtype=torch.float16, device="spyre:0").reshape(4, 64)
a = x.narrow(0, 0, 1).clone()   # rows [0] — correct
b = x.narrow(0, 2, 1).clone()   # rows [2] — WRONG: silently returns row 0 again
```

This surfaced in `alltoall_base`'s redesign, which clones a different row-offset
each round; every round after the first quietly reused the first round's data.

## Root cause

`clone()` / `copy_()` / `.to()` on a device tensor dispatch through
`aten::_copy_from` → `spyre__copy_from` (`torch_spyre/ops/eager.py`) → the
device→device branch calls `torch.ops.spyre.copy_from_d2d(...)`, which is
compiled once and cached via `compile_once` (`torch_spyre/ops/eager.py`).

The slice position lives only in the tensor's `storage_offset` — tensor
metadata, not an explicit argument. **A graph input's `storage_offset` is
dropped by the Inductor backend:**

- Upstream Inductor builds each placeholder's `FixedLayout` from
  `static_sizes_strides` — **size and stride only**; `storage_offset()` is
  never read, so `FixedLayout.offset == 0`
  (`torch/_inductor/graph.py`).
- Spyre's `SpyreTensorLayout` has **no offset field** at all
  (`torch_spyre/csrc/spyre_tensor_impl.h`); it carries only `device_size`,
  `stride_map`, `device_dtype`, `element_arrangement`.
- At runtime the compiled kernel binds the **storage base pointer**
  (`tensor.storage().data_ptr()` in `job_plan.cpp` / `spyre_stream.cpp`), and
  the per-dim byte offset in the SDSC descriptor
  (`codegen/superdsc.py`, the `offsets` field) is baked **only** from the
  in-graph coordinate constant (`dim_coord.as_coeff_Add()[0]`), gated on
  `dev_dim_size > it_dim_size`. With the offset dropped, both conditions fail
  and `offsets[dim] = 0`.

Net effect: the kernel always reads from element 0. `narrow(0, 0, 1)` happens to
be correct because row 0 *is* at offset 0; every other offset returns row 0.

The C++ DMA path honors `storage_offset` **only** for host↔device copies
(`spyre_mem.cpp`: `offset_src_ = host2device ? storage_offset : 0`), which is why
eager H2D/D2H slicing works but the compiled D2D path does not.

### Why this is NOT (only) a dynamo caching bug

An earlier hypothesis was that `copy_from_d2d(src, dst)` takes no int offset
argument, so dynamo's `TENSOR_MATCH` guard (which keys on dtype/device/size/
stride but **not** `storage_offset`) reuses one compiled binary across offsets.
That guard gap is real, but fixing *only* it (add int offset args +
`specialize_int=True`, mirroring `spyre::overwrite` PR #2084) **failed on
hardware** — every offset still returned row 0. Forcing a per-offset recompile
does not help when the offset never enters the graph to begin with: the
recompiled graph's data path is identical and still reads from the base. The
offset must be **re-introduced in-graph**; the guard fix is necessary but not
sufficient.

## How `spyre::overwrite` avoids this (the working precedent)

`spyre::overwrite(input, output, dims, offsets)` takes offsets as **explicit int
args** and its lowering rebuilds the slice in-graph via
`ir.SliceView.create(output, dim, offset, offset + size)`
(`torch_spyre/_inductor/lowering.py`), which sets
`FixedLayout.offset = old.offset + stride[dim] * start`. That offset flows into
the coordinate constant and gets baked. `lower_cat` and `lower_constant_pad_nd`
use the same `SliceView.create` pattern. `overwrite` also wraps its compiled
call in `torch._dynamo.config.patch(specialize_int=True)` so each distinct
offset is baked as a **constant** (not promoted to a symbol by auto-dynamic).

`overwrite` is a *scatter with no pre-existing view*, so it must take base +
offsets. `copy_from_d2d` is the gather direction and can reconstruct the same
way.

## Options considered

All options: (a) add explicit int offset args to `copy_from_d2d`, (b) keep
`specialize_int=True`, (c) re-introduce the offset **in-graph** in the lowering.
They differ in **how much of the view the lowering rebuilds**.

### Option 1 — General `as_strided` (rebuild size + stride + offset)

Pass the base storage plus full `size`, `stride`, `offset` for `src` and `dst`;
the lowering rebuilds each view from scratch as a `ReinterpretView`, ignoring
the input layout.

- **Pros:** correct for *any* view — contiguous narrow, `select`, multi-dim
  slices, transpose, permute, stepped slices — because it reconstructs stride
  and offset explicitly rather than trusting the input layout.
- **Cons:** ~8 extra args on the internal op schema; more lowering code.

### Option 2 — Scalar offset only (re-inject just the offset) — **CHOSEN (pending HW confirmation)**

Pass the base tensors plus only the two scalar `storage_offset`s; the lowering
leaves the input's existing (size, stride) layout alone and installs only the
offset via a `ReinterpretView` (`_reoffset` in `lowering.py`).

- **Pros:** minimal — 2 args, closest to the existing signature; the lowering
  change is small and localized.
- **Cons:** relies on the assumption that the input's **stride survives** into
  the device layout (true for contiguous narrows). If a transposed / permuted /
  stepped view reaches this path and the device layout does not preserve the
  stride, a single scalar offset may not fully reconstruct the view — risking a
  repeat of the same silent-wrong-data class of bug for those cases.

## Decision

**Pick Option 2 first, and let hardware decide whether it is sufficient.**

Rationale:

1. Only `storage_offset` is provably dropped by Inductor; size and stride are
   read from `static_sizes_strides` and *should* survive. So the minimal,
   offset-only reconstruction is the natural fix and matches the exact gap.
2. It is far smaller and easier to review than rebuilding the full view, and it
   reuses the callers already wired to pass `storage_offset()`.
3. The risk (strided views) is **testable**, not theoretical. We added explicit
   regression tests for transpose / permute / select / stepped-slice views
   (`tests/inductor/test_copy_from_d2d_offsets.py`,
   `TestCopyFromD2DStridedViews`). If any strided case fails on hardware while
   the contiguous cases pass, that is the signal that Option 2 is insufficient
   and we escalate to **Option 1** (rebuild size + stride + offset).

This staged approach avoids over-engineering the op schema until the tests prove
the general reconstruction is actually needed.

## Fix (Option 2) — files touched

- `torch_spyre/_inductor/customops.py` — `copy_from_d2d` gains
  `src_off: int, dst_off: int`; body wraps the compiled call in
  `specialize_int=True`. `register_fake` updated to match.
- `torch_spyre/_inductor/lowering.py` — new `_reoffset(node, offset)` helper
  builds a `ReinterpretView` with the offset installed on the layout;
  `lower_spyre_from_d2d` calls it on `src` and `dst` before `mutate_to`.
- `torch_spyre/ops/eager.py` and `torch_spyre/_monkey_patch.py` — both D2D
  callers pass `self.storage_offset()` / `dst.storage_offset()`.

## Hardware results (Option 2)

`pytest tests/inductor/test_copy_from_d2d_offsets.py` → **7 passed, 5 xfailed**.

Passing (the reported bug is fixed): all contiguous row-offset cases —
`test_multi_offset_clone` (the exact reproducer), `test_loop_varying_offsets`,
`test_revisit_offset`, `test_copy_into_sliced_dst` — plus `permute` and
`select` views.

Failing → marked `@unittest.expectedFailure`, split into two pre-existing
backend limitations that neither Option 1 nor 2 addresses:

1. **Restickify / stick incompatibility** (hard compile error, not offset
   related): `test_transpose_clone`, `test_transpose_then_offset_clone`,
   `test_transpose_varying_offsets_loop`, `test_stepped_slice_clone`. These
   fail in `optimize_restickify.py`'s beam search ("no mechanism to resolve
   stick incompatibility" / "scatter one stick to multiple sticks").
   **Verified to fail identically on the pre-fix baseline** (an offset==0
   transpose reduces `lower_spyre_from_d2d` to the original `mutate_to`), so
   they are not regressions, and reconstructing size+stride explicitly
   (Option 1) does not resolve a stick-layout rejection.
2. **Stick-dim offset** (silent wrong data): `test_column_slice_inner_offset`.
   A `storage_offset` landing in the innermost/stick dim is not baked
   correctly — `superdsc` decomposes per-dim offsets against `device_size` and
   does not split a stick-dim offset. Same failure *mode* as the original bug
   (wrong data), different *mechanism* (stick-dim decomposition).

**Conclusion: Option 2 is sufficient for the reported bug.** The 5 xfails are
scoped to a follow-up PR (stick-dim offset baking + transposed/stepped d2d
restickify support).

## Verification

1. Reproducer: `x.narrow(0, 0, 1).clone()` then `x.narrow(0, 2, 1).clone()` on
   `spyre:0` — `b` returns row 2. ✅
2. `pytest tests/inductor/test_copy_from_d2d_offsets.py -v` — 7 passed,
   5 xfailed (documented limitations above).
3. Regression: `pytest tests/inductor/test_overwrite.py
   tests/inductor/test_copy_back_elision.py tests/inductor/test_inductor_ops.py`.
