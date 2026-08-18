# Support device-to-device copy of a sliced/mutated ≥2-D tiled tensor (`narrow(0).clone()` / `narrow(0).copy_()`)

**Labels:** bug, inductor
**Related:** #3236 (row-narrow D2D silent-wrong — fixed by #3237/`05ed893`), #3264 (stick-dim/column-narrow D2D silent-wrong — open). This issue is the **wide-tensor (>1 stick/row) generalization** and the **crash-manifesting** member of that family. Supersedes the earlier draft `copy_from_d2d-dim0-narrow-wide-tensor-no-valid-candidate.md` in this dir.

## The torch ops that fail

On a Spyre device tensor whose inner dim spans **more than one stick** (fp16 stick = 64 elems, so any `hidden ≥ 128` — e.g. the ubiquitous `[tokens, 4096]` activation), these standard PyTorch ops fail:

```python
import torch, torch_spyre
DEV = torch.device("spyre:0")
x = torch.arange(8 * 4096, dtype=torch.float16, device=DEV).reshape(8, 4096)

# (1) SRC-offset clone of a dim-0 slice
a = x.narrow(0, 2, 2).clone()          # rows [2:4)  -> COMPILE CRASH
b = x[4:6].clone()                     # equivalent  -> COMPILE CRASH

# (2) DST-offset copy into a dim-0 slice (mutating a sub-region)
y = torch.empty_like(x)
y.narrow(0, 4, 2).copy_(x[0:2])        # write rows into [4:6) -> COMPILE CRASH
```

**Failure mode:** hard compile-time crash, not silent-wrong:

```
terminate called after throwing an instance of 'DtException'
  what():  DtException: There must be at least one valid candidate.,
  file deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 1075
```

thrown inside the `dxp_standalone` compile of `torch.ops.spyre.copy_from_d2d`
(the Python traceback dead-ends at `torch_spyre/ops/eager.py` `spyre__copy_from`
→ `customops.py` `copy_from_d2d` → `subprocess.run`, because it is a C++
`terminate` in the compile subprocess).

### What currently works (the boundary)

- `x.narrow(0, 0, N).clone()` — **offset 0** (no re-injection needed).
- 1-D tensors of any size — genuinely contiguous, flat offset is valid.
- `[N, 64]` tensors (exactly **one stick per row**) at any row offset — this is the
  case #3236/#3237 fixed and `test_copy_from_d2d_offsets.py` covers.
- Whole-tensor D2D copy (`x.clone()`, `y.copy_(x)`) — offset 0, layout-preserving.

The gap is precisely: **a nonzero-offset slice (as source of a clone, or as
mutated destination of a copy) of a ≥2-D tensor with >1 stick per row.**

## Why it fails

A ≥2-D tensor is stored in a 4-D **tiled/stickified** `SpyreTensorLayout`
(`device_size` e.g. `[64, 2, 64]` for `[2,4096]` fp16; the 64-elem stick is
innermost). Logical dim-0 (the row/token dim we slice) maps to an **interior**
device dimension, and a single logical row is scattered across the device buffer
as many discrete stick-sized segments (64 sticks/row for hidden=4096), **not** a
contiguous byte range.

The compiled D2D path re-injects the dropped `storage_offset` (see #3236) as a
single flat host-space scalar via `_reoffset`
(`torch_spyre/_inductor/lowering.py:741`); `compute_coordinates`
(`torch_spyre/_inductor/views.py`) distributes it against the device `stride_map`,
and `_create_sdsc_tensors` (`torch_spyre/_inductor/codegen/superdsc.py:353-379`)
bakes a per-device-dim byte offset into the SDSC. deeptools' scheduler then
enumerates, per primary dim, split params that must be both divisible and a
stick-multiple (`L3DlOpsScheduler.cpp:1018-1071`); when the interior-dim
offset/extent is inconsistent, **no** candidate survives → the assert at
`L3DlOpsScheduler.cpp:1075`.

**Open investigative point (must be settled before the fix):** two independent
code investigations disagreed on the exact malformed value, and neither produced a
working fix:

1. One held that the trimmed slice VIEW loses its parent interior extent, so
   superdsc sees the interior dim as size = slice-length (e.g. 2) rather than the
   parent (8); then the proven `dev_dim_size > it_dim_size` gate
   (`superdsc.py:373-379`, which installs offset **and** `backGap` **and** the
   stride-trim together — the #3236 path) never fires, and baking the offset
   *alone* yields a coordinate addressing tiles outside the declared extent.
2. The other verified from C++ (`spyre_views.cpp:115-143`) that `narrow` inherits
   the **parent** `SpyreTensorLayout` verbatim (so `device_size` row extent = the
   parent 8, and the gate *should* fire), which contradicts (1) and means the real
   trimming/inconsistency site is elsewhere.

Crucially, a change that only re-baked the offset made an isolated `off=2` clone
compile correctly **but the same operation inside a larger graph (a collective
that stacks `off=2,4,6` clones plus a narrowed-dst copy) still crashed** — and
reverting to the gate-only baseline reproduces the original crash. So the precise
mechanism is **not yet pinned**; the decisive missing evidence is the actual
`dev_dim_size`/`it_dim_size`/`offsets`/`backGap` superdsc emits, and the exact
scheduler-rejected coordinate, **for the failing op inside the real graph** (all
analysis so far is either C++-source reasoning or isolated single-op unit tests).

## What we want to add

Full support for a device-to-device copy of a **nonzero-offset dim-0 slice** of a
tiled ≥2-D tensor — both directions:
- **source:** `slice.clone()` produces a correct fresh contiguous tensor of the
  sliced rows;
- **destination:** `dst.narrow(0, k, len).copy_(src)` writes `src` into rows
  `[k, k+len)` of `dst` without disturbing other rows.

value-correct (not just non-crash), for any `hidden` that is a stick multiple, at
any stick-aligned row offset. This is the primitive that a **bandwidth-optimal
tensor-parallel `all_reduce`** (built as `reduce_scatter + all_gather`, which must
chunk `[tokens, hidden]` along dim-0) depends on; without it, TP≥4 must fall back
to a whole-tensor all_gather+sum (correct but ~2× the data movement).

### Acceptance criteria

- `x.narrow(0, k, len).clone()` on `[N, hidden]` fp16 (hidden a stick multiple),
  for k ∈ {0, and several stick-aligned nonzero offsets} — value-checked against
  `x.cpu()[k:k+len]`.
- `y.narrow(0, k, len).copy_(src)` (wide-tensor narrowed **destination**) —
  value-checked; other rows of `y` unchanged.
- Existing `[N, 64]` (1-stick/row) cases in `test_copy_from_d2d_offsets.py` still
  pass — **no regression** to the #3236/#3237 path.
- The `#3264` stick-dim/column-narrow case may remain a separate tracked
  limitation, but this fix should not make it worse.
- New regression tests in `tests/inductor/test_copy_from_d2d_offsets.py` covering
  the SRC-clone at multiple offsets and the wide narrowed-DST copy.

### Suggested first step (diagnosis before fix)

Add temporary instrumentation in `_create_sdsc_tensors` (dump
`dev_dim_size`/`it_dim_size`/`offsets[dim]`/`backGap[dim]` per dim) and identify
which copy op (clone vs narrowed-dst) and which offset first trips the scheduler,
on the real failing graph — to settle the two contradictory root-cause theories
before committing a fix. The fix is expected to be localized to the torch/codegen
layer (`superdsc.py` and/or `lowering.py` `_reoffset` and/or `views.py`
`compute_coordinates`); deeptools already consumes per-primary-dim offsets, so no
scheduler change is anticipated.

## Environment

- PyTorch 2.11, latest torch-spyre, branch `phase-1-overlap-communicate-compute`.
- Reproduces single-rank (world_size 1); the compile crash is not
  distribution-dependent.
