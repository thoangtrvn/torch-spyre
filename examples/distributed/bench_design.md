# Runtime Scheduler & Distributed Communication Benchmark Design

## Overview

Two benchmark scripts measuring the flex runtime scheduler and spyre-comms stack
under realistic 70B LLM inference workloads:

1. **`bench_scheduler.py`** — single-rank scheduler microbenchmark
2. **`bench_distributed.py`** — multi-rank distributed benchmark (torchrun)

---

## Communication Ops Mapping

### `bench_scheduler.py` (single-rank)

| Scenario | Scheduler Op | Comms Op | Pipeline | Notes |
|----------|-------------|----------|----------|-------|
| `h2d_kv_head` | `schedule(H2D)` → `submitDma` | None | DMAI | Pure host→device DMA |
| `h2d_activation_small` | `schedule(H2D)` → `submitDma` | None | DMAI | |
| `h2d_activation_medium` | `schedule(H2D)` → `submitDma` | None | DMAI | |
| `h2d_activation_large` | `schedule(H2D)` → `submitDma` | None | DMAI | |
| `h2d_weight_shard` | `schedule(H2D)` → `submitDma` | None | DMAI | 460MB single transfer |
| `d2h_kv_head` | `schedule(D2H)` → `submitDma` | None | DMAO | Pure device→host DMA |
| `d2h_activation_large` | `schedule(D2H)` → `submitDma` | None | DMAO | |

**Gap:** Single-rank never hits the COMPUTE pipeline, `P2PDataExchange`,
`issueBarrier()`, or `pipelineTypeForOp()`. Only DMAI/DMAO are exercised.

### `bench_distributed.py` (multi-rank)

| Benchmark | Scheduler Ops | Comms Primitives | Pipelines Hit | Under the Hood |
|-----------|--------------|------------------|---------------|----------------|
| **AllReduce** | `schedule(P2PDataExchange)` × N | Ring/Tree allreduce via `spyre_comms::WorkSchedule` | COMPUTE | Each allreduce = multiple rounds of `P2PSendData` + `P2PRecvData` bundled into `P2PDataExchange`. All on COMPUTE pipeline. |
| **Send/Recv** | `schedule(P2PSendData)` + `schedule(P2PRecvData)` | Direct `dist.send()` / `dist.recv()` | COMPUTE | Single-rank-pair point-to-point. COMPUTE pipeline only. |
| **Layer Sim** | `schedule(H2D)` + `schedule(P2PDataExchange)` × 2 + `schedule(D2H)` | H2D → AllReduce(attn) → AllReduce(FFN) → D2H | DMAI → COMPUTE → DMAO | Exercises all three pipelines with STRICT_ORDERING barriers between pipeline switches. Hits `issueBarrier()`, `pipelineTypeForOp()`, fence wait/resolve. |

### Pipeline Coverage Summary

| Pipeline | `bench_scheduler.py` | `bench_distributed.py` |
|----------|---------------------|----------------------|
| DMAI (H2D) | ✓ all h2d_* scenarios | ✓ Layer Sim H2D step |
| DMAO (D2H) | ✓ all d2h_* scenarios | ✓ Layer Sim D2H step |
| COMPUTE (P2P) | ✗ not exercised | ✓ AllReduce, Send/Recv, Layer Sim |

### Scheduler Feature Coverage

| Feature | `bench_scheduler.py` | `bench_distributed.py` |
|---------|---------------------|----------------------|
| `schedule(H2D)` / `schedule(D2H)` | ✓ | ✓ (Layer Sim) |
| `schedule(P2PSendData)` | ✗ | ✓ (Send/Recv) |
| `schedule(P2PRecvData)` | ✗ | ✓ (Send/Recv) |
| `schedule(P2PDataExchange)` | ✗ | ✓ (AllReduce, Layer Sim) |
| `scheduleCommon()` code path | ✓ (DMA only) | ✓ (DMA + P2P) |
| `pipelineTypeForOp()` (P2P→Compute) | ✗ never triggered | ✓ (every P2P op) |
| `issueBarrier()` | ✗ no pipeline switches | ✓ (DMA↔COMPUTE switches in Layer Sim) |
| `fenceDequeForStream()` | ✓ (DMAI/DMAO only) | ✓ (all three maps) |
| `maybeSetBarrier()` (STRICT_ORDERING) | ✗ single pipeline only | ✓ (pipeline switches) |
| Async pipeline submission workers | ✓ (DMAI/DMAO) | ✓ (all three workers) |
| Backpressure warning | ✗ unlikely at these sizes | Possible with large AllReduce |

---

## 70B Model Tensor Size Reference

Llama-70B-like model, fp16, tensor parallelism = 4:

| Parameter | Value |
|-----------|-------|
| hidden_dim | 8192 |
| intermediate_size | 28672 |
| num_heads | 64 |
| num_kv_heads | 8 |
| Total weights | ~140 GB |
| Per-rank weights | ~35 GB |

Per-layer communication pattern (inference):

```
1. H2D:  activation shard [batch × seq_len, 8192]     → DMAI pipeline
2. AllReduce: attention output [batch × seq_len, 8192] → COMPUTE pipeline
3. AllReduce: FFN output [batch × seq_len, 8192]      → COMPUTE pipeline
4. D2H:  output readback (optional)                    → DMAO pipeline
```

---

## Scenario Details

### AllReduce Scenarios

| Name | Elements | Bytes | What it models |
|------|----------|-------|----------------|
| `ar_kv_short` | 64 × 8192 = 524K | 1 MB | KV cache, short sequence |
| `ar_activation_small` | 128 × 8192 = 1M | 2 MB | Short sequence activation |
| `ar_activation_medium` | 512 × 8192 = 4M | 8 MB | Medium sequence |
| `ar_activation_large` | 2048 × 8192 = 16M | 32 MB | Long sequence, batch=1 |
| `ar_batch4_long` | 4 × 2048 × 8192 = 67M | 128 MB | Batch=4, long sequence |
| `ar_weight_shard` | 8192 × 28672 = 234M | 460 MB | Full weight-sized transfer |

### Send/Recv Scenarios

| Name | Elements | Bytes | What it models |
|------|----------|-------|----------------|
| `sr_kv_head` | 64 × 8192 | 1 MB | Point-to-point KV cache |
| `sr_activation` | 2048 × 8192 | 32 MB | Point-to-point activation |

### H2D/D2H Scenarios (single-rank)

| Name | Shape | Bytes | What it models |
|------|-------|-------|----------------|
| `h2d_kv_head` | [64, 8192] | 1 MB | KV cache head load |
| `h2d_activation_small` | [128, 8192] | 2 MB | Small activation load |
| `h2d_activation_medium` | [512, 8192] | 8 MB | Medium activation load |
| `h2d_activation_large` | [2048, 8192] | 32 MB | Full-sequence activation |
| `h2d_weight_shard` | [8192, 28672] | 460 MB | FFN weight shard |
| `d2h_kv_head` | [64, 8192] | 1 MB | KV cache readback |
| `d2h_activation_large` | [2048, 8192] | 32 MB | Full activation readback |

---

## Model-Derived Scenario Generation

The hand-picked scenarios above (`ar_kv_short`, `ar_weight_shard`, etc.) are
round numbers, not sizes tied to any real model. `--model-bench` on
`bench_distributed.py` generates scenarios instead from real LLM
architecture parameters (`bench_model_configs.py`) crossed with a small set
of representative workload points (`bench_scenarios.py`):

- **26 dense models** (Qwen3, Granite, Llama, Mistral, Phi, OLMo, Falcon,
  DeepSeek-Coder, Yi, Gemma, GPT-2, GPT-Neo, Pythia, Ministral, and others —
  see `bench_model_configs.DENSE_MODELS`), each contributing an **AllReduce**
  scenario (attention-output/FFN-down-proj, sized `batch × seq_len ×
  hidden_size`) and an **AllGather** scenario (vocab-parallel logits, sized
  `batch × seq_len × ceil(vocab_size / world_size)` per rank).
- **5 workload points**: `decode_b1`/`decode_b8`/`decode_b32` (batch∈{1,8,32},
  seq_len=1 — continuous-batching serving latency) and
  `prefill_s2048`/`prefill_s4096` (batch=1, seq_len∈{2048,4096} —
  time-to-first-token / throughput).
- **2 MoE models** (Mixtral-8x7B-style, DeepSeek-V2-Lite-style — see
  `bench_model_configs.MOE_MODELS`), each contributing an **all-to-all
  PROXY** scenario per workload point (see "Known Gap" below).

26 models × 5 workload points × 2 dense ops = 260 AllReduce/AllGather
scenarios, plus 2 MoE models × 5 workload points = 10 proxy scenarios.
`--models`/`--workload-points` filter this down for a fast, targeted run;
`--model-bench all` with no filters runs everything.

Results append to a CSV history ledger (`bench_history.csv` by default,
next to this script) with a datetime and git commit per row, so performance
is comparable across code changes over time — see `bench_history.py` for
the schema. This is opt-in (`--model-bench` defaults to `none`) and
deliberately kept separate from `--bench all`, so a bare invocation's
runtime doesn't silently change.

Run at TP=2 and TP=4 the same way as every other benchmark here — via
`--nproc-per-node`, no script changes needed:

```bash
torchrun --nproc-per-node 2 bench_distributed.py --model-bench all
torchrun --nproc-per-node 4 bench_distributed.py --model-bench all

# Filtered:
torchrun --nproc-per-node 4 bench_distributed.py --model-bench allreduce \
    --models gpt2-124m,pythia-70m --workload-points decode_b1
```

---

## Usage

### Single-rank scheduler benchmark

```bash
python bench_scheduler.py
python bench_scheduler.py --iterations 20 --warmup 5
python bench_scheduler.py --scenario h2d_weight_shard
python bench_scheduler.py --json results_scheduler.json
```

### Multi-rank distributed benchmark

```bash
torchrun --nproc-per-node 2 bench_distributed.py
torchrun --nproc-per-node 4 bench_distributed.py
torchrun --nproc-per-node 4 bench_distributed.py --iterations 20 --warmup 5

# Specific benchmark suites
torchrun --nproc-per-node 4 bench_distributed.py --bench allreduce
torchrun --nproc-per-node 4 bench_distributed.py --bench sendrecv
torchrun --nproc-per-node 4 bench_distributed.py --bench layer

# Force specific allreduce algorithm
COLL_ALLREDUCE_ALGO=Ring torchrun --nproc-per-node 4 bench_distributed.py --bench allreduce
COLL_ALLREDUCE_ALGO=ReduceScatterAllGather torchrun --nproc-per-node 4 bench_distributed.py

# Save results
torchrun --nproc-per-node 4 bench_distributed.py --json results_distributed.json
```

---

## Output Format

Both scripts report results per-scenario and optionally write JSON:

```
h2d_activation_large            e2e=  1234.5us (p99=  1350.0)  sched=   45.2us     32.0MB   25.82 GB/s  | H2D: large activation
```

JSON format (per scenario):

```json
{
  "benchmark": "allreduce",
  "name": "ar_activation_large",
  "description": "AllReduce: activation (seq=2048, batch=1)",
  "rank": 0,
  "world_size": 4,
  "elements": 16777216,
  "dtype": "float16",
  "bytes": 33554432,
  "iterations": 10,
  "e2e_us": {"mean": 1234.5, "p50": 1200.0, "p99": 1350.0, "min": 1180.0, "max": 1400.0},
  "throughput_gbps": 25.82
}
```

---

## Known Gap: Single-Rank Compute Coverage

`bench_scheduler.py` does not exercise the COMPUTE pipeline, `issueBarrier()`,
or `pipelineTypeForOp()` because these require P2P operations which only exist
in multi-rank scenarios. A future addition could add a Compute scenario using
a pre-compiled add kernel to exercise the COMPUTE pipeline in single-rank mode,
but this would not test barrier logic (which requires pipeline switches between
DMAI/DMAO and COMPUTE).

## Known Gap: MoE All-to-All is a Proxy, Not a Native Benchmark

MoE expert-routing communication (`--model-bench alltoall_proxy`) has no
native collective to measure. `torch.distributed`'s `alltoall`/
`alltoall_base` are not implemented on this backend --
`SpyreCCLBackend::alltoall()`/`alltoall_base()` unconditionally throw
`SpyreCCLNotSupportedException` (`torch_spyre/csrc/distributed/
spyre_ccl.cpp`). Underneath, `spyre-comms` has an internal `AllToAll`
collective algorithm (`coll/alltoall.{hpp,cpp}`, `CollectiveType::AllToAll
= 3`), but no public `Context::all_to_all()` method exposes it (unlike
`allreduce`/`allgather`/`barrier`, which are all properly wired end-to-end).

`bench_alltoall_proxy()` works around this by orchestrating an
all-to-all-*shaped* traffic pattern via the P2P `send`/`recv` primitives
that ARE exposed (N-1 pairwise sendrecv exchanges per rank). This measures
real wire/HDMA traffic in the right shape, but is a **lower bound, not a
production measurement** -- it's missing everything a real all-to-all
needs:

- **Topology-aware routing**: flat pairwise exchange, not hierarchical
  (intra-switch/node first, then inter-node) the way a PCIe-topology-aware
  implementation should route it.
- **Two-phase variable-size protocol**: assumes uniform routing (`tokens ×
  top_k / num_experts` per pair) as an expected-value approximation, not the
  real data-dependent token distribution a router produces at runtime.
  A production implementation exchanges routing metadata first, then sizes
  receive buffers exactly.
- **Compute/comm overlap**: times the exchange in isolation; a real MoE
  serving path overlaps dispatch/combine all-to-all with other ranks'
  expert FFN compute.

Exposing a real `Context::all_to_all()` (variable split-size aware,
topology-routed) on top of the already-built internal algorithm is a
tracked follow-up -- not something this benchmark script should attempt to
build. Treat `alltoall_proxy` numbers as a floor for what's achievable
today, not a target for what's achievable once that exists.
