# Phase-1 async dispatch — performance analysis for real-world LLM inference on Sentient 1p5

**Date:** 2026-07-25
**Companion to:** `2026-07-25-async-progress-thread-design.md`
**Purpose:** Decide the C1 worker-scope question (one process-global worker vs
world-only single-PG) from what actually helps LLM inference on 1p5 hardware,
not from implementation convenience. Also bounds honestly what Phase 1 can and
cannot deliver, so it is not oversold.

---

## 1. The 1p5 performance envelope (hard constraints)

Everything below is bounded by these facts (senlib docs 35/36):

- **No concurrent kernel execution** (QGI compute queue processes 1 at a time).
  Two collectives — or a collective and a compute kernel — cannot both *execute*
  on the device simultaneously, EXCEPT across the distinct DMGR queues: DMA
  (DMAI/DMAO, 32-deep) can progress while compute (QGI, 4-deep) runs.
- **No device-side events / cross-stream wait.** Ordering is host-assisted; every
  collective's cross-rank coordination has a host component — BUILD's blocking OOB
  address pre-exchange plus START's HDMA/CBS rendezvous.
- **Fixed per-collective host overhead:** firmware dispatch 5-20 µs; per-kernel
  50-100 µs; plus the OOB round-trip.
- **Single global `comm_stream_`.** All collectives in a process serialize through
  the one `spyre_comms_global.comm_runtime_stream` regardless of which PG issued
  them.

**Critical implication:** Phase-1 async cannot create device-level parallelism the
hardware does not have. Its only lever is **hiding host-side latency** (OOB
exchange + dispatch + submission) behind device execution, by freeing the caller
thread. The entire performance question reduces to: **how much host-side,
off-critical-path latency is there to hide, per workload?**

---

## 2. Dense LLM (tensor-parallel)

Megatron-style TP forward, per layer:
`... → out_proj → all_reduce → residual+norm → MLP → all_reduce → residual+norm`.
Two all-reduces per layer, **both on the critical path** — the next op consumes the
result.

### Prefill (compute-bound, large activation tensors)
- All-reduce is bandwidth-dominated. The caller reaches `wait()` almost immediately
  (residual add needs the result). Essentially no independent compute to overlap
  against within the layer.
- Phase-1 overlap window ≈ **near zero on the critical path.** The only gains are
  contract-correctness and not stalling the host dispatch thread.
- **Real latency reduction: marginal.** The prefill lever is Phase 2 (chunked
  intra-collective compute/comm overlap), not Phase 1.

### Decode (memory/latency-bound, tiny tensors — dominant serving cost)
- Tensors are `batch × 1 × hidden` — tiny. All-reduce time is **dominated by fixed
  overhead** (OOB round-trip + dispatch + HDMA setup), not bandwidth.
- A decode step issues ~`2 × num_layers` all-reduces (≈64-160 for a 32-80 layer
  model), each carrying that fixed host cost.
- **This is where Phase 1 pays off.** Today each collective serializes its
  OOB+dispatch on the host before the device starts. With the caller unblocked, the
  host runs ahead — building/dispatching collective N+1's setup while collective N
  executes on the device. Across 100+ tiny collectives per token, hiding even
  10-30 µs of host overhead each is a meaningful fraction of per-token decode
  latency.
- **Caveat:** the single `comm_stream_` still serializes *device* execution, and
  BUILD(N+1)'s OOB is itself blocking on the worker — so the hidden quantity is
  "host overhead of N+1 overlapped with device execution of N," not full
  pipelining. Bounded, but real, and it compounds over layer count.

**Dense verdict:** modest win, concentrated entirely in decode, scaling with layer
count. Prefill barely benefits from Phase 1.

---

## 3. MoE (expert-parallel)

MoE layer adds expert-parallel **all-to-all** (token dispatch) + **all-to-all**
(combine) on top of the attention all-reduce.

- **all_to_all is EXCLUDED from Phase 1** (multi-leg inline-wait path, design
  §3.2). MoE's distinctive and most expensive communication gets **no async
  benefit** in Phase 1; it stays synchronous.
- What MoE *does* get is the same attention all-reduce benefit as dense (§2) —
  decode-concentrated, modest.
- MoE decode is even more latency-bound (small per-expert token counts), so the
  fixed-overhead-hiding argument applies with equal force to its all-reduces — but
  the all-to-alls dominate MoE comm time and are untouched.

**MoE verdict:** Phase-1 value for MoE is real but *partial* — it helps the dense
attention all-reduce, not the expert all-to-all. The big MoE win lives in Phase 2
(async + chunked all-to-all) and in completing the asymmetric/variable all-to-all
forms. **This is an argument for NOT over-investing Phase-1 complexity specifically
to chase MoE.**

---

## 4. How C1 (worker scope) is settled by this analysis

**On 1p5 there is no device concurrency to trade away by serializing PGs on one
worker.** All PGs share the single global `comm_stream_`; they already cannot
execute concurrently on the device. So a **single process-global FIFO worker**
(option a) serializing PG-A and PG-B loses **zero device throughput** — it only
imposes program order on work that was already device-serialized.

Deployment reality:
- DP×TP and vLLM subgroup serving is the multi-PG case. With torchrun
  one-process-per-card, a DP=2×TP=2 deployment puts multiple PGs (TP subgroup +
  DP group + world) in the same process → multiple `SpyreCCLBackend`s sharing the
  global stream/OOB/allocator.
- **Option (b) — world-only** — disables async for subgroups. That turns Phase 1
  OFF precisely in the DP×TP / MoE serving deployments where decode latency (the
  thing §2 says Phase 1 helps) matters most. A latency feature inactive in the
  latency-critical topology.
- **Option (a) — process-global worker** — keeps async on for all PGs, loses no
  device concurrency (there is none), and fixes the C1 race + C2 teardown
  regression by construction (one driver of the shared singletons).

**The performance evidence points cleanly to option (a).** The only reason to
prefer (b) was implementation simplicity; the analysis shows (b) sacrifices the
feature exactly where the workload benefits.

---

## 5. Bottom line

| Workload / phase | Phase-1 async benefit on 1p5 |
|---|---|
| Dense **prefill** | Marginal (critical-path all-reduce; needs Phase 2 chunking) |
| Dense **decode** | **Modest–meaningful** — hides fixed host overhead across ~2×layers small collectives |
| MoE attention all-reduce | Same as dense decode (modest) |
| MoE expert **all-to-all** | **None** (excluded from Phase 1; Phase 2 territory) |
| Multi-group (DP×TP) enablement | Only with **option (a)** process-global worker |

Two conclusions:
1. **Phase 1 is a decode-latency + contract-correctness change, not a throughput
   change.** Its ceiling is "hide host overhead," which is real but bounded. The
   larger latency wins (chunked intra-collective overlap, async all-to-all) are
   Phase 2. State this plainly so Phase 1 is not oversold.
2. **The performance evidence favors C1 option (a)** (process-global worker): on
   1p5 it costs no device concurrency, and it is the only option under which the
   decode/MoE serving deployments actually get the feature.

---

## 6. Load-bearing assumption to confirm before deciding

The §4 conclusion rests on **Spyre vLLM DP×TP instantiating multiple PGs per
process** (TP/DP subgroups via `new_group()`), rather than each replica being a
fully separate process with only a world PG. If it is the latter (separate
process per replica, world PG only), option (b) would suffice and C1 collapses.
A focused check of the Spyre vLLM / `spyre-inference` process-group setup will
confirm the subgroup topology before the decision is locked.

Related project memory: DP-rank derivation and subgroup work
([[spyre-dp-rank-derivation]], [[spyre-subgroup-failure-isolation-boundary]],
[[project-spyre-inference-local-fork-decision]]).
