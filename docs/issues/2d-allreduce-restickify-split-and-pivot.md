# 2-D TP all_reduce: layout-transposing restickify work-division crash, and the whole-tensor pivot

> **RESOLUTION (2026-08-18) — read this first.** The real root cause was NOT a
> restickify/work-division/codegen defect. torch-spyre lacked `aten::ones`/
> `aten::full` kernels, so `torch.ones([tokens,hidden])` (used to build the
> collective's test buffers) got a TRANSPOSED stickified layout; every crash
> below was a symptom of that. Fix: register `aten::ones`/`aten::full`
> (`torch_spyre/ops/eager.py`) so they build via `torch.empty`+fill and get the
> normal layout. With that fix, the NATIVE LIBCOLL Ring path (`ctx.allreduce`)
> reduces 2-D correctly at 4 ranks across even/uneven/decode/prefill + subgroup
> -- HW-verified. So the whole-tensor pivot AND the chunked compose were both
> REMOVED; 2-D all_reduce now takes the same native Ring path as 1-D, with no
> special 2-D code, no env guards. The narrow/clone/restickify investigation
> below is retained as history (all symptoms). See memory
> `spyre-ones-full-transposed-layout-rootcause`. The genuine per-op latency
> floor found while benchmarking is tracked separately
> (`spyre-allreduce-per-op-latency-floor`).

**Status:** RESOLVED by the aten::ones/aten::full layout fix; 2-D uses native
LIBCOLL Ring (compose + pivot removed). History below.
**Labels:** bug, inductor, distributed
**Related:** #3236 (fixed), #3264 (open, stick-dim column narrow), and the
sibling doc `copy_from_d2d-sliced-2d-tiled-view.md`.

## What we were building

A tensor-parallel `all_reduce(SUM)` on the Spyre c10d backend for the
`[num_tokens, hidden]` activation (fired twice per transformer layer). The
bandwidth-optimal implementation composes `reduce_scatter + all_gather` and
chunks dim-0 across ranks. On Spyre that chunking uses
`tensor.narrow(0, off, len).clone()` at the torch layer (the `alltoall_base`
pattern) to produce per-rank chunk tensors.

## The codegen bug this hit (now filed as follow-up)

The `narrow(0).clone()` of a `[8,4096]`-derived chunk compiles a **fused**
`ReStickifyOpHBM` + `identity` on a `[2,4096]` tensor that the deeptools
work-division scheduler rejects:

```
DtException: There must be at least one valid candidate.
  deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1075/1180
```

**Root cause (precisely characterized from the emitted SDSC / inductor module):**
it is a **layout-transposing restickify**, not an offset problem. The fused
op's two tensors have *different stick dims*:

- INPUT: `device_size=[1,4096,64]`, `device_coordinates=[floor(c0/64), c1, Mod(c0,64)]`
  → the stick coordinate is `Mod(c0, 64)`, and **c0 is the 2-row dim**.
- OUTPUT: `device_size=[64,2,64]`, `device_coordinates=[floor(c1/64), c0, Mod(c1,64)]`
  → the stick coordinate is `Mod(c1, 64)`, and **c1 is the 4096 dim**.
- `iteration_space`: `c0=(2, split 1)`, `c1=(4096, split 32)`.

Work-division splits `c1` (=4096) across all 32 cores, but for the INPUT
tensor `c1` is a *non-stick middle* dim while its stick is `c0` (only 2 rows).
Splitting `c1` 32 ways is inconsistent with the input's stick structure, so the
scheduler finds no valid corelet parameter and aborts. Every affine `beta_` is
0 — there is **no** re-injected storage offset involved.

Our tree crashes where the newer reference tree
(`torch-spyre@async-host-engine`) does not, because our coordinate/work-division
machinery is materially older: it lacks the `indirect_sizes`-threaded
`compute_coordinates`/`device_coordinates`, the offset-aware stick predicate,
`normalize_coordinates` (#3079), and the core-mapping refactor (#3268). Porting
those is a deep multi-module coordinate-API migration (proven by a copy-and-see
experiment: dropping in the reference files breaks 5 call sites via a signature
change and pulls 4 missing symbols) — not a surgical fix, and not something to
land build-blind for one collective.

Failed intermediate fix attempts (recorded so they are not retried): an
`offset_baked_blocked_vars` guard (no offset exists → never fires) and a
`single_stick_blocked_vars` guard (shares the `len(free_symbols)!=1`
recognition blind spot that causes the bug). Both were reverted. Only the
reference-faithful #3340 `coordinate_mask_blocked_vars` + `align_tensors`
stick-count-gcd port was kept as legitimate hardening.

## The pivot (what actually ships)

`allreduce_2d_compose` was rewritten to **avoid narrow/clone/restickify
entirely**: every rank `all_gather`s the whole `[tokens, hidden]` contribution
from all ranks and sums them on-device. Whole-tensor transfers are layout-safe
(the tiling reconstructs identically on the receiver), and the on-device
`add_` operates on whole tensors — no sliced operand, so the transposing
restickify never occurs.

- Correct for **all** token counts, including `tokens < world` (no chunking) —
  this subsumes the previously-planned decode fallback.
- Cost: moves ~`world ×` bytes, roughly 2× a bandwidth-optimal
  reduce_scatter+all_gather. Accepted as correctness-first; a bandwidth-optimal
  chunked path is a tracked follow-up, blocked on the codegen fix above.

## Example PyTorch code

### 1. The collective under test (what the fix makes correct)

```python
import os, torch, torch_spyre  # noqa: F401
import torch.distributed as dist

torch.spyre._impl._lazy_init()
DEVICE = torch.device(f"spyre:{os.environ['RANK']}")
dist.init_process_group("cpu:gloo,spyre:spyreccl")

n = dist.get_world_size()
# TP activation shape: [num_tokens, hidden]. hidden=4096; tokens varies.
t = torch.ones((8, 4096), dtype=torch.float16, device=DEVICE)
dist.all_reduce(t, op=dist.ReduceOp.SUM)          # -> whole-tensor allgather+sum
# every element must equal world_size (all-ones summed across n ranks)
assert bool((t.to("cpu") == float(n)).all()), t.to("cpu").flatten()[:4].tolist()
print(f"rank {dist.get_rank()}: 2-D all_reduce OK, all == {n}")
```

Run: `torchrun --nproc-per-node 4 this_file.py` (needs ≥4 Spyre cards).

### 2. Minimal repro of the underlying codegen bug (single card)

The crash is at **compile time** in `dxp_standalone`, so it reproduces on one
card with no collectives. The trigger is a dim-0 slice-clone of a wide
(>1 stick/row) tensor that fuses a layout-transposing restickify:

```python
import torch, torch_spyre  # noqa: F401
torch.spyre._impl._lazy_init()
DEV = torch.device("spyre:0")

# [8,4096] fp16: 4096 = 64 sticks/row. A dim-0 chunk of this, cloned inside a
# graph that restickifies it, fuses ReStickifyOpHBM+identity whose work-division
# splits the 4096 (c1) dim 32 ways while the input's stick dim is the 2-row (c0)
# dim -> deeptools "There must be at least one valid candidate".
x = torch.arange(8 * 4096, dtype=torch.float16, device=DEV).reshape(8, 4096)
chunk = x.narrow(0, 2, 2).clone()   # rows [2:4), each row = 64 sticks
print(chunk.to("cpu").flatten()[:4].tolist())
```

Note: whether this exact snippet crashes depends on how the surrounding graph
fuses the restickify — in isolation an eager `narrow().clone()` may compile,
while the same op inside the collective's fused graph crashes. The reliable
reproduction is the 4-rank collective before the pivot; the standalone snippet
is the shape/op that the codegen fix must eventually handle. `SENCORES=1`
(single-core, no work-division split) makes the crash disappear, confirming the
split is the trigger.

## Acceptance (for the codegen follow-up)

- A dim-0 slice-clone of a wide `[N, hidden]` (hidden a stick multiple) that
  fuses a layout-transposing restickify compiles and is value-correct across
  cores (not just single-core).
- No regression to the `[N,64]` single-stick cases (#3236/#3237) or pointwise
  ops.
- Then the bandwidth-optimal chunked `reduce_scatter+all_gather` allreduce can
  replace the whole-tensor pivot.
