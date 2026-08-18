# Distributed collectives (SpyreCCL c10d backend)

The `"spyreccl"` c10d backend registers Spyre as a `torch.distributed` backend
for multi-accelerator (torchrun, one process per card) inference — tensor
parallel (TP), data parallel (DP), and their subgroups. Initialize with a hybrid
backend so CPU-side coordination uses gloo and device collectives use SpyreCCL:

```python
dist.init_process_group("cpu:gloo,spyre:spyreccl")
```

## Supported collectives

| Op | Support | Notes |
|----|---------|-------|
| `all_reduce(SUM)` | ✅ 1-D and ≥2-D | The TP activation reduce. SUM only. |
| `reduce_scatter(SUM)` | ✅ list form | Pairwise sendrecv shuffle + on-device accumulate. |
| `all_gather` | ✅ uniform + variable | Variable (per-rank sizes) via `exchange_uneven`. |
| `broadcast` | ✅ | |
| `gather` | ✅ | |
| `send` / `recv` / batched P2P | ✅ | Pre-resolved local CB path. |
| `barrier` | ✅ | |
| `all_reduce` non-SUM | ❌ | Only SUM is implemented. |
| `allreduce_coalesced` | ❌ | No public interface. |

Collectives dispatch asynchronously through a process-global FIFO progress
worker (see the async progress-thread design); `dist.all_reduce(..., async_op=
True)` returns a real `Work` whose `wait()` blocks until completion. Multiple
process groups (world + TP/DP subgroups) share the one worker.

## 2-D `all_reduce` (the TP activation reduce)

A tensor-parallel `all_reduce` on the `[num_tokens, hidden]` activation is the
hottest collective in TP inference (fired twice per transformer layer). It uses
the **native LIBCOLL Ring path** — identical to the 1-D path: the tensor is
flattened to `{numel}`, reduce-scattered across ranks (per-chunk device add
kernel), then all-gathered. There is **no special-case 2-D code**.

Correct for any token count including uneven splits (`tokens % world != 0`) and
decode (`tokens < world`), on the world group and on subgroups. HW-verified at
4 ranks across `[8,4096]`/`[6,4096]`/`[1,4096]`/`[32,4096]` on world + a `[0,1]`
TP subgroup (`tests/distributed/test_allreduce_2d.py`).

### Layout requirement (important)

Device tensors on Spyre use a tiled/stickified `SpyreTensorLayout` (128-byte
stick = 64 fp16 elems). A collective operates correctly only when all
participating tensors carry the **normal** stickified layout for their
`(shape, dtype)`. Tensor constructors that go through the registered Spyre
kernels (`torch.empty`, `zeros`, `ones`, `full`, and results of device ops)
produce that layout. Historically `torch.ones`/`torch.full` lacked dedicated
Spyre kernels and fell through to a generic path that produced a **transposed**
layout, which silently broke 2-D `all_reduce`; those kernels are now registered
(`torch_spyre/ops/eager.py`) so all constructors agree on layout. Do not
hand-construct device tensors via paths that bypass these kernels.

## Performance characteristics

Measured TP=4, fp16, p50 (`bench_allreduce_2d.sh`):

| shape | p50 (µs) |
|-------|----------|
| `[1,4096]` decode | ~4500 |
| `[8,4096]` | ~4650 |
| `[128,4096]` | ~11600 |
| `[512,4096]` | ~19000 |
| `[2048,4096]` | ~59000 |

**Known bottleneck — a ~4.5ms fixed per-op floor.** Even an 8 KB decode
`all_reduce` costs ~4.5ms, so latency at small/decode sizes is dominated by
fixed per-collective overhead (per-call OOB address re-exchange + wireup +
synchronous `start()`/`wait()` on the single comm stream), not the reduction
itself. For TP inference (2 reduces/layer) this floor — not bandwidth — is the
dominant cost. Eliminating it (address-exchange caching across stable-buffer
steps + async overlap) is a tracked follow-up; use `SPYRE_COMMS_TIMING=1` and
`prof_allreduce_floor.sh` to attribute the build-vs-execute split.

## Test / bench tooling

- `tests/distributed/test_allreduce_2d.py` — 2-D correctness regression
  (world + subgroup, even/uneven/decode/prefill). Runner:
  `run_allreduce_2d_verify.sh`.
- `bench_allreduce_2d.sh` — latency gate (p50/p95, prefill + decode).
- `prof_allreduce_floor.sh` — per-op floor attribution (SPYRE_COMMS_TIMING
  build breakdown vs total).
