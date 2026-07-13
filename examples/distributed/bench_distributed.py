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

"""
Flex Runtime & Spyre-Comms distributed benchmark -- multi-rank.

Measures allreduce, send/recv, and simulated transformer layer performance
at tensor sizes representative of a 70B LLM inference workload (TP=4).

Also supports model-derived scenarios sized from real LLM architectures
(see bench_model_configs.py/bench_scenarios.py) via --model-bench -- opt-in,
disabled by default so a bare invocation keeps today's fast smoke-test
behavior. Results from --model-bench runs are appended to a CSV history
ledger (bench_history.py) traceable to a git commit, so performance can be
compared across code changes over time.

Usage:
    torchrun --nproc-per-node 2 bench_distributed.py
    torchrun --nproc-per-node 4 bench_distributed.py
    torchrun --nproc-per-node 4 bench_distributed.py --iterations 20 --warmup 5
    torchrun --nproc-per-node 4 bench_distributed.py --json results.json
    COLL_ALLREDUCE_ALGO=Ring torchrun --nproc-per-node 4 bench_distributed.py

    # Model-derived scenarios (AllReduce + AllGather), all 26 dense models,
    # all workload points, at TP=2 and TP=4:
    torchrun --nproc-per-node 2 bench_distributed.py --model-bench all
    torchrun --nproc-per-node 4 bench_distributed.py --model-bench all

    # Filtered to specific models/workload points:
    torchrun --nproc-per-node 4 bench_distributed.py --model-bench allreduce \\
        --models gpt2-124m,pythia-70m --workload-points decode_b1

    # MoE all-to-all PROXY (see bench_alltoall_proxy() docstring for caveats):
    torchrun --nproc-per-node 4 bench_distributed.py --model-bench alltoall_proxy
"""

import argparse
import datetime
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from bench_history import (
    append_history_rows,
    capture_relevant_env_vars,
    get_deployed_state,
)
from bench_model_configs import DENSE_MODELS, MOE_MODELS
from bench_scenarios import (
    WORKLOAD_POINTS,
    filter_by_names,
    generate_dense_scenarios,
    generate_moe_scenarios,
)

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"

# ---------------------------------------------------------------------------
# 70B model tensor size reference (Llama-70B-like, fp16, TP=4)
#
#   hidden_dim = 8192, intermediate_size = 28672, num_kv_heads = 8
#   Per rank: ~35GB weights
#
# AllReduce happens twice per layer (after self-attn, after FFN).
# Shape: [batch * seq_len, hidden_dim]
# ---------------------------------------------------------------------------

ALLREDUCE_SCENARIOS = {
    "ar_kv_short": {
        "desc": "AllReduce: KV cache short (seq=64, batch=1)",
        "elements": 64 * 8192,
        "dtype": torch.float16,
    },
    "ar_activation_small": {
        "desc": "AllReduce: activation (seq=128, batch=1)",
        "elements": 128 * 8192,
        "dtype": torch.float16,
    },
    "ar_activation_medium": {
        "desc": "AllReduce: activation (seq=512, batch=1)",
        "elements": 512 * 8192,
        "dtype": torch.float16,
    },
    "ar_activation_large": {
        "desc": "AllReduce: activation (seq=2048, batch=1)",
        "elements": 2048 * 8192,
        "dtype": torch.float16,
    },
    "ar_batch4_long": {
        "desc": "AllReduce: batch=4, long seq (4x2048x8192)",
        "elements": 4 * 2048 * 8192,
        "dtype": torch.float16,
    },
    "ar_weight_shard": {
        "desc": "AllReduce: weight-sized (8192x28672) = 460MB",
        "elements": 8192 * 28672,
        "dtype": torch.float16,
    },
}

SENDRECV_SCENARIOS = {
    "sr_kv_head": {
        "desc": "Send/Recv: KV cache head (seq=64)",
        "elements": 64 * 8192,
        "dtype": torch.float16,
    },
    "sr_activation": {
        "desc": "Send/Recv: activation (seq=2048)",
        "elements": 2048 * 8192,
        "dtype": torch.float16,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device_tensor(elements, dtype, fill_value=None):
    """Create a CPU tensor, optionally fill it, then move to device."""
    t = torch.zeros(elements, dtype=dtype)
    if fill_value is not None:
        t.fill_(fill_value)
    return t.to(DEVICE)


def _percentile(data, pct):
    """Compute percentile from a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _element_bytes(dtype):
    """Bytes per element for the dtypes used by these benchmarks."""
    return {torch.float16: 2, torch.float32: 4}[dtype]


def _stats(values_us):
    """Return mean/p50/p99/min/max in microseconds."""
    if not values_us:
        return {"mean": 0, "p50": 0, "p99": 0, "min": 0, "max": 0}
    return {
        "mean": round(statistics.mean(values_us), 1),
        "p50": round(_percentile(values_us, 50), 1),
        "p99": round(_percentile(values_us, 99), 1),
        "min": round(min(values_us), 1),
        "max": round(max(values_us), 1),
    }


def _build_scenario_result(
    scenario,
    rank,
    e2e,
    message_bytes,
    aggregate_bytes_per_rank,
    iterations,
    warmup,
    run_ts,
    git_commit,
    git_dirty,
    flex_opensource_commit,
    flex_opensource_dirty,
    env_vars,
    is_proxy,
    notes,
):
    """Build the canonical per-scenario result dict for a model-derived
    AllReduce/AllGather/all-to-all-proxy run. One dict shape feeds both the
    --json output and (via _result_to_csv_row) the CSV history ledger."""
    throughput_gbps = (
        (aggregate_bytes_per_rank / (statistics.mean(e2e) / 1e6)) / 1e9 if e2e else 0
    )
    return {
        "benchmark": scenario.benchmark,
        "name": scenario.name,
        "description": scenario.description,
        "rank": rank,
        "world_size": scenario.world_size,
        "elements": scenario.elements,
        "dtype": str(scenario.dtype).split(".")[-1],
        "bytes": message_bytes,
        "iterations": iterations,
        "warmup": warmup,
        "e2e_us": _stats(e2e),
        "throughput_gbps": round(throughput_gbps, 2),
        "model_name": scenario.model_name,
        "workload_point": scenario.workload_point,
        "phase": scenario.phase,
        "batch": scenario.batch,
        "seq_len": scenario.seq_len,
        "hidden_size": scenario.hidden_size,
        "vocab_size": scenario.vocab_size,
        "num_layers": scenario.num_layers,
        "num_experts": scenario.num_experts,
        "top_k": scenario.top_k,
        "tokens_to_expert": scenario.tokens_to_expert,
        "message_bytes": message_bytes,
        "aggregate_bytes_per_rank": aggregate_bytes_per_rank,
        "is_proxy": is_proxy,
        "notes": notes,
        "timestamp_utc": run_ts,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "flex_opensource_commit": flex_opensource_commit,
        "flex_opensource_dirty": flex_opensource_dirty,
        "env_vars": env_vars,
        "coll_allreduce_algo": os.environ.get("COLL_ALLREDUCE_ALGO", ""),
    }


def _result_to_csv_row(result):
    """Flatten a canonical result dict (nested e2e_us stats, "name" key) into
    a flat dict matching bench_history.HISTORY_CSV_COLUMNS."""
    stats = result.get("e2e_us", {})
    return {
        "timestamp_utc": result.get("timestamp_utc", ""),
        "git_commit": result.get("git_commit", ""),
        "git_dirty": result.get("git_dirty", ""),
        "flex_opensource_commit": result.get("flex_opensource_commit", ""),
        "flex_opensource_dirty": result.get("flex_opensource_dirty", ""),
        "env_vars": result.get("env_vars", ""),
        "benchmark": result.get("benchmark", ""),
        "scenario_name": result.get("name", ""),
        "description": result.get("description", ""),
        "is_proxy": result.get("is_proxy", False),
        "notes": result.get("notes", ""),
        "model_name": result.get("model_name", ""),
        "workload_point": result.get("workload_point", ""),
        "phase": result.get("phase", ""),
        "batch": result.get("batch", ""),
        "seq_len": result.get("seq_len", ""),
        "hidden_size": result.get("hidden_size", ""),
        "vocab_size": result.get("vocab_size", ""),
        "num_layers": result.get("num_layers", ""),
        "num_experts": result.get("num_experts", ""),
        "top_k": result.get("top_k", ""),
        "tokens_to_expert": result.get("tokens_to_expert", ""),
        "world_size": result.get("world_size", ""),
        "rank": result.get("rank", ""),
        "iterations": result.get("iterations", ""),
        "warmup": result.get("warmup", ""),
        "dtype": result.get("dtype", ""),
        "elements": result.get("elements", ""),
        "message_bytes": result.get("message_bytes", ""),
        "aggregate_bytes_per_rank": result.get("aggregate_bytes_per_rank", ""),
        "e2e_us_mean": stats.get("mean", ""),
        "e2e_us_p50": stats.get("p50", ""),
        "e2e_us_p99": stats.get("p99", ""),
        "e2e_us_min": stats.get("min", ""),
        "e2e_us_max": stats.get("max", ""),
        "throughput_gbps": result.get("throughput_gbps", ""),
        "coll_allreduce_algo": result.get("coll_allreduce_algo", ""),
    }


# ---------------------------------------------------------------------------
# Benchmark: AllReduce
# ---------------------------------------------------------------------------


def bench_allreduce(elements, dtype, iterations, warmup):
    """Benchmark allreduce at a given tensor size."""
    rank = dist.get_rank()
    t = _device_tensor(elements, dtype, fill_value=float(rank + 1))

    # Warmup
    for _ in range(warmup):
        dist.all_reduce(t)
        torch.spyre.synchronize()

    e2e_times = []
    dispatch_times = []  # build + start() (host, up to all_reduce return)
    transfer_times = []  # synchronize() wait (device transfer)
    for _ in range(iterations):
        # Re-fill to ensure consistent input each iteration
        t.fill_(float(rank + 1))

        t0 = time.perf_counter()
        dist.all_reduce(t)
        t_disp = time.perf_counter()
        torch.spyre.synchronize()
        t1 = time.perf_counter()

        e2e_times.append((t1 - t0) * 1e6)
        dispatch_times.append((t_disp - t0) * 1e6)
        transfer_times.append((t1 - t_disp) * 1e6)

    # SPYRE_COMMS_TIMING split: dispatch (build+start) vs transfer (wait).
    # Pair with the C++ [SPYRE_COMMS_TIMING] build_us/oob_us lines to isolate
    # pure start() and pure synchronize(). No-op unless the env var is set.
    if os.environ.get("SPYRE_COMMS_TIMING") and rank == 0:
        d, x = _stats(dispatch_times), _stats(transfer_times)
        print(
            f"  [SPLIT] {elements}el dispatch_us(mean={d['mean']:.1f} p99={d['p99']:.1f}) "
            f"transfer_us(mean={x['mean']:.1f} p99={x['p99']:.1f})",
            flush=True,
        )

    return e2e_times


# ---------------------------------------------------------------------------
# Benchmark: Send/Recv pingpong
# ---------------------------------------------------------------------------


def bench_sendrecv(elements, dtype, iterations, warmup):
    """Benchmark send/recv pingpong between rank 0 and rank 1."""
    rank = dist.get_rank()
    size = dist.get_world_size()

    if size < 2:
        return None

    t = _device_tensor(elements, dtype, fill_value=float(rank + 1))

    # Warmup
    for _ in range(warmup):
        if rank == 0:
            dist.send(t, dst=1)
            dist.recv(t, src=1)
        elif rank == 1:
            dist.recv(t, src=0)
            dist.send(t, dst=0)
        torch.spyre.synchronize()

    e2e_times = []
    for _ in range(iterations):
        t.fill_(float(rank + 1))

        t0 = time.perf_counter()
        if rank == 0:
            dist.send(t, dst=1)
            dist.recv(t, src=1)
        elif rank == 1:
            dist.recv(t, src=0)
            dist.send(t, dst=0)
        torch.spyre.synchronize()
        t1 = time.perf_counter()

        if rank <= 1:
            e2e_times.append((t1 - t0) * 1e6)

    return e2e_times


# ---------------------------------------------------------------------------
# Benchmark: AllGather
# ---------------------------------------------------------------------------


def bench_allgather(elements, dtype, iterations, warmup):
    """Benchmark allgather at a given per-rank tensor size.

    Structural clone of bench_allreduce, using the output_list-of-tensors
    pattern from allgather.py: dist.all_gather(output_list, input_device).
    """
    rank = dist.get_rank()
    size = dist.get_world_size()
    t = _device_tensor(elements, dtype, fill_value=float(rank + 1))
    output_list = [torch.zeros_like(t) for _ in range(size)]

    for _ in range(warmup):
        dist.all_gather(output_list, t)
        torch.spyre.synchronize()

    e2e_times = []
    for _ in range(iterations):
        t.fill_(float(rank + 1))

        t0 = time.perf_counter()
        dist.all_gather(output_list, t)
        torch.spyre.synchronize()
        t1 = time.perf_counter()

        e2e_times.append((t1 - t0) * 1e6)

    return e2e_times


# ---------------------------------------------------------------------------
# Benchmark: MoE all-to-all PROXY
# ---------------------------------------------------------------------------


def _warmup_p2p_pairing(send_buf, recv_buf):
    """Force this rank's first-ever HDMA P2P call to be a synchronized,
    mutually-paired exchange with rank^1, before the real per-rank-
    independent alltoall_proxy traffic runs.

    Works around a flex-opensource deadlock (see flex-opensource/docs/
    hdma-barrier-vs-oob-exchange-deadlock.md): a rank's first-ever P2P op
    triggers a one-time, world-size-scoped barrier inside
    HdmaShmMgmt::hdma_cpu_init_ that needs every rank to arrive -- but a
    rank parked in that barrier can't service the per-pair OOB address
    exchange another, unrelated rank needs from it first. If ranks' first
    P2P partners are staggered (exactly what an all-to-all's independent
    per-rank partner ordering produces), this deadlocks. Pairing every rank
    with (rank XOR 1) is a perfect matching for any even world size: every
    rank's first P2P op is mutually matched with exactly one partner, and
    (thanks to the barrier() below) every pair attempts it at the same
    synchronized moment -- satisfying the global barrier and the per-pair
    handshake simultaneously.

    Reuses the caller's real send/recv buffers (same size as the actual
    scenario) rather than a smaller placeholder, so the one-time HDMA pool
    doesn't latch to a smaller size than what's about to be used -- the pool
    is sized once on the first HDMA batch and never regrown.

    No-op for odd world size (no clean perfect matching); harmless to repeat
    on every call, since after the first, hdma_cpu_init_'s is_hdma_allocated_
    latch makes the barrier a no-op and this just adds one already-cached
    OOB round-trip.

    Set BENCH_SKIP_WARMUP=1 to disable this and reproduce the underlying
    flex-opensource deadlock on demand (see hdma-barrier-vs-oob-exchange-
    deadlock.md) -- e.g. to verify a real fix there without this benchmark's
    own workaround masking it.
    """
    if os.environ.get("BENCH_SKIP_WARMUP") == "1":
        return
    size = dist.get_world_size()
    if size % 2 != 0:
        return
    rank = dist.get_rank()
    partner = rank ^ 1
    dist.barrier()
    if rank < partner:
        dist.send(send_buf, dst=partner)
        dist.recv(recv_buf, src=partner)
    else:
        dist.recv(recv_buf, src=partner)
        dist.send(send_buf, dst=partner)
    torch.spyre.synchronize()


def bench_alltoall_proxy(tokens_to_expert, hidden_size, dtype, iterations, warmup):
    """MoE expert-routing all-to-all PROXY via pairwise send/recv.

    *** LOWER-BOUND PROXY -- NOT a measurement of a real all-to-all. ***

    torch.distributed's alltoall/alltoall_base are not implemented on this
    backend (SpyreCCLBackend::alltoall{,_base} unconditionally throw
    SpyreCCLNotSupportedException -- see torch_spyre/csrc/distributed/
    spyre_ccl.cpp), and spyre-comms has no public Context::all_to_all()
    (only an internal, unexposed AllToAll collective algorithm). This proxy
    orchestrates an all-to-all-SHAPED traffic pattern using the send/recv
    primitives that ARE exposed, so it measures real wire/HDMA traffic in
    the right shape -- but provides none of what a production all-to-all
    needs: no topology-aware routing (flat pairwise, not hierarchical), no
    two-phase variable-size protocol (assumes uniform routing, not the real
    data-dependent token distribution), and no compute/comm overlap.
    Treat these numbers as a floor, not a target. Exposing a real
    Context::all_to_all() is a tracked follow-up, not something to build
    here -- see bench_design.md's "Known Gap" section.

    For N ranks, every rank does N-1 pairwise sendrecv exchanges, one with
    each other rank. Deadlock avoidance follows the same rule as
    bench_sendrecv: within each unordered pair (a, b), a < b sends-then-recvs,
    b recvs-then-sends. See _warmup_p2p_pairing() for why a synchronized
    warm-up pairing runs first -- this per-rank-independent ordering is
    exactly what exposes a flex-opensource HDMA init deadlock otherwise.
    """
    rank = dist.get_rank()
    size = dist.get_world_size()
    if size < 2:
        return None

    elements = max(1, round(tokens_to_expert)) * hidden_size
    partners = [r for r in range(size) if r != rank]

    send_buf = _device_tensor(elements, dtype, fill_value=float(rank + 1))
    recv_buf = _device_tensor(elements, dtype, fill_value=0.0)

    _warmup_p2p_pairing(send_buf, recv_buf)

    def _one_round():
        for partner in partners:
            if rank < partner:
                dist.send(send_buf, dst=partner)
                dist.recv(recv_buf, src=partner)
            else:
                dist.recv(recv_buf, src=partner)
                dist.send(send_buf, dst=partner)

    for _ in range(warmup):
        _one_round()
        torch.spyre.synchronize()

    e2e_times = []
    for _ in range(iterations):
        dist.barrier()  # bound stragglers before starting the timed round
        t0 = time.perf_counter()
        _one_round()
        torch.spyre.synchronize()
        t1 = time.perf_counter()
        e2e_times.append((t1 - t0) * 1e6)

    return e2e_times


# ---------------------------------------------------------------------------
# Benchmark: Simulated transformer layer
# ---------------------------------------------------------------------------


def bench_layer(iterations, warmup):
    """Simulate one transformer layer's communication pattern for a 70B model.

    Per-layer ops (TP=4):
      1. H2D: activation shard [batch*seq, hidden_dim]
      2. AllReduce: attention output [batch*seq, hidden_dim]
      3. AllReduce: FFN output [batch*seq, hidden_dim]
      4. D2H: output readback (optional)

    Uses seq=2048, batch=1 as the default realistic inference size.
    """
    hidden_dim = 8192
    seq_len = 2048
    elements = seq_len * hidden_dim

    # Pre-allocate tensors
    cpu_activation = torch.randn(elements, dtype=torch.float16)
    dev_activation = torch.zeros(elements, dtype=torch.float16, device=DEVICE)
    dev_reduce = _device_tensor(elements, torch.float16, fill_value=1.0)
    cpu_output = torch.zeros(elements, dtype=torch.float16)

    # Warmup
    for _ in range(warmup):
        dev_activation.copy_(cpu_activation)
        torch.spyre.synchronize()
        dev_reduce.fill_(1.0)
        dist.all_reduce(dev_reduce)
        torch.spyre.synchronize()
        dev_reduce.fill_(1.0)
        dist.all_reduce(dev_reduce)
        torch.spyre.synchronize()
        cpu_output.copy_(dev_activation)
        torch.spyre.synchronize()

    e2e_times = []
    h2d_times = []
    ar1_times = []
    ar2_times = []
    d2h_times = []

    for _ in range(iterations):
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        dev_activation.copy_(cpu_activation)
        torch.spyre.synchronize()
        t1 = time.perf_counter()
        h2d_times.append((t1 - t0) * 1e6)

        dev_reduce.fill_(1.0)
        t0 = time.perf_counter()
        dist.all_reduce(dev_reduce)
        torch.spyre.synchronize()
        t1 = time.perf_counter()
        ar1_times.append((t1 - t0) * 1e6)

        dev_reduce.fill_(1.0)
        t0 = time.perf_counter()
        dist.all_reduce(dev_reduce)
        torch.spyre.synchronize()
        t1 = time.perf_counter()
        ar2_times.append((t1 - t0) * 1e6)

        t0 = time.perf_counter()
        cpu_output.copy_(dev_activation)
        torch.spyre.synchronize()
        t1 = time.perf_counter()
        d2h_times.append((t1 - t0) * 1e6)

        e2e_times.append((t1 - t_total) * 1e6)

    return {
        "e2e_us": _stats(e2e_times),
        "h2d_us": _stats(h2d_times),
        "allreduce_attn_us": _stats(ar1_times),
        "allreduce_ffn_us": _stats(ar2_times),
        "d2h_us": _stats(d2h_times),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Flex Runtime & Spyre-Comms distributed benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of measured iterations"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--json", type=str, default=None, help="Write results to JSON file"
    )
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        choices=["allreduce", "sendrecv", "layer", "all", "none"],
        help=(
            "Which of the original hand-coded benchmarks to run (default: all). "
            "Pass 'none' to skip them entirely -- e.g. when only --model-bench "
            "scenarios are wanted, since --bench and --model-bench are independent "
            "flags and both run if --bench is left at its default."
        ),
    )
    parser.add_argument(
        "--model-bench",
        type=str,
        default="none",
        choices=["allreduce", "allgather", "alltoall_proxy", "all", "none"],
        help=(
            "Which model-derived benchmark to run, sized from real LLM architectures "
            "(see bench_model_configs.py). Opt-in, disabled by default so a bare "
            "invocation keeps today's fast smoke-test behavior unchanged."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model names to include in --model-bench, or 'all'",
    )
    parser.add_argument(
        "--workload-points",
        type=str,
        default="all",
        help="Comma-separated workload point names to include in --model-bench, or 'all'",
    )
    parser.add_argument(
        "--history-csv",
        type=str,
        default=None,
        help=(
            "Path to append --model-bench results to (default: bench_history.csv "
            "next to this script, regardless of invocation cwd)"
        ),
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip writing to the CSV history ledger (for throwaway local runs)",
    )
    args = parser.parse_args()

    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")

    dist.init_process_group(backend=C10D_BACKEND)

    rank = dist.get_rank()
    size = dist.get_world_size()

    if args.bench is None:
        args.bench = "all"

    # Captured once per run (not per scenario) -- a --model-bench run can take
    # minutes across hundreds of scenarios, and every row from one invocation
    # sharing one timestamp/commit makes "which run is this row from" trivial
    # without a separate run_id column.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_commit, git_dirty, flex_opensource_commit, flex_opensource_dirty = (
        get_deployed_state(script_dir)
    )
    env_vars = capture_relevant_env_vars()
    history_csv_path = args.history_csv or os.path.join(script_dir, "bench_history.csv")

    print(f"# Flex Distributed Benchmark: rank={rank} size={size}")
    print(f"# COLL_ALLREDUCE_ALGO={os.environ.get('COLL_ALLREDUCE_ALGO', '(default)')}")
    print(f"# Iterations: {args.iterations} (warmup: {args.warmup})")
    print()

    all_results = []
    history_rows = []

    # --models is a single flag applied to two different tables (dense models
    # for allreduce/allgather, MoE models for the all-to-all proxy). Validate
    # against the UNION of both tables' names, so a name valid in the OTHER
    # table doesn't raise here -- it should just yield zero matches from
    # whichever table it doesn't belong to, not crash the whole run.
    all_model_names = {m.name for m in DENSE_MODELS} | {m.name for m in MOE_MODELS}

    # --- AllReduce ---
    if args.bench in ("allreduce", "all"):
        for name, cfg in ALLREDUCE_SCENARIOS.items():
            elements = cfg["elements"]
            dtype = cfg["dtype"]
            element_bytes = _element_bytes(dtype)

            # Skip if tensor too small for number of ranks (stick-alignment)
            min_elements = 64 * size
            if elements < min_elements:
                if rank == 0:
                    print(
                        f"  SKIP {name}: {elements} elements < minimum {min_elements} for {size} ranks"
                    )
                continue

            total_bytes = elements * element_bytes

            e2e = bench_allreduce(elements, dtype, args.iterations, args.warmup)
            throughput_gbps = (
                (total_bytes / (statistics.mean(e2e) / 1e6)) / 1e9 if e2e else 0
            )

            result = {
                "benchmark": "allreduce",
                "name": name,
                "description": cfg["desc"],
                "rank": rank,
                "world_size": size,
                "elements": elements,
                "dtype": str(dtype).split(".")[-1],
                "bytes": total_bytes,
                "iterations": args.iterations,
                "e2e_us": _stats(e2e),
                "throughput_gbps": round(throughput_gbps, 2),
            }
            all_results.append(result)

            if rank == 0:
                s = result["e2e_us"]
                print(
                    f"  {name:30s}  e2e={s['mean']:8.1f}us (p99={s['p99']:8.1f})  "
                    f"{total_bytes / 1024 / 1024:7.1f}MB  "
                    f"{result['throughput_gbps']:6.2f} GB/s  | {cfg['desc']}"
                )

    # --- Send/Recv ---
    if args.bench in ("sendrecv", "all"):
        for name, cfg in SENDRECV_SCENARIOS.items():
            elements = cfg["elements"]
            dtype = cfg["dtype"]
            element_bytes = _element_bytes(dtype)
            total_bytes = elements * element_bytes

            e2e = bench_sendrecv(elements, dtype, args.iterations, args.warmup)

            if e2e is not None and rank <= 1:
                throughput_gbps = (
                    (total_bytes / (statistics.mean(e2e) / 1e6)) / 1e9 if e2e else 0
                )
                result = {
                    "benchmark": "sendrecv",
                    "name": name,
                    "description": cfg["desc"],
                    "rank": rank,
                    "world_size": size,
                    "elements": elements,
                    "dtype": str(dtype).split(".")[-1],
                    "bytes": total_bytes,
                    "iterations": args.iterations,
                    "e2e_us": _stats(e2e),
                    "throughput_gbps": round(throughput_gbps, 2),
                }
                all_results.append(result)

                if rank == 0:
                    s = result["e2e_us"]
                    print(
                        f"  {name:30s}  e2e={s['mean']:8.1f}us (p99={s['p99']:8.1f})  "
                        f"{total_bytes / 1024 / 1024:7.1f}MB  "
                        f"{result['throughput_gbps']:6.2f} GB/s  | {cfg['desc']}"
                    )
            elif rank > 1:
                if rank == 2:
                    print(
                        f"  {name}: ranks 2+ are non-participants in 2-rank send/recv"
                    )

    # --- Layer simulation ---
    if args.bench in ("layer", "all"):
        dist.barrier()
        if rank == 0:
            print()
            print("# Layer Simulation (70B, seq=2048, batch=1, TP=4)")
        layer_result = bench_layer(args.iterations, args.warmup)

        result = {
            "benchmark": "layer_simulation",
            "description": "Simulated 70B transformer layer (seq=2048, batch=1)",
            "rank": rank,
            "world_size": size,
            "iterations": args.iterations,
            **layer_result,
        }
        all_results.append(result)

        if rank == 0:
            print(f"  {'layer_total':30s}  e2e={layer_result['e2e_us']['mean']:8.1f}us")
            print(f"  {'  h2d':30s}  {layer_result['h2d_us']['mean']:8.1f}us")
            print(
                f"  {'  allreduce_attn':30s}  {layer_result['allreduce_attn_us']['mean']:8.1f}us"
            )
            print(
                f"  {'  allreduce_ffn':30s}  {layer_result['allreduce_ffn_us']['mean']:8.1f}us"
            )
            print(f"  {'  d2h':30s}  {layer_result['d2h_us']['mean']:8.1f}us")

    # --- Model-derived dense scenarios (AllReduce / AllGather) ---
    if args.model_bench in ("allreduce", "allgather", "all"):
        dense_ops = tuple(
            op for op in ("allreduce", "allgather") if args.model_bench in (op, "all")
        )
        models = filter_by_names(args.models, DENSE_MODELS, valid_names=all_model_names)
        workload_points = filter_by_names(args.workload_points, WORKLOAD_POINTS)
        scenarios = generate_dense_scenarios(
            models, workload_points, size, ops=dense_ops
        )

        if rank == 0:
            print(
                f"# Model-derived scenarios: {len(scenarios)} ({dense_ops}, "
                f"{len(models)} models x {len(workload_points)} workload points)"
            )

        for scenario in scenarios:
            element_bytes = _element_bytes(scenario.dtype)
            min_elements = 64 * size
            if scenario.elements < min_elements:
                if rank == 0:
                    print(
                        f"  SKIP {scenario.name}: {scenario.elements} elements < minimum "
                        f"{min_elements} for {size} ranks"
                    )
                continue

            message_bytes = scenario.elements * element_bytes
            if scenario.benchmark == "allreduce":
                e2e = bench_allreduce(
                    scenario.elements, scenario.dtype, args.iterations, args.warmup
                )
                aggregate_bytes = message_bytes
            else:  # allgather
                e2e = bench_allgather(
                    scenario.elements, scenario.dtype, args.iterations, args.warmup
                )
                aggregate_bytes = (
                    message_bytes * size
                )  # full gathered output across all ranks

            result = _build_scenario_result(
                scenario,
                rank,
                e2e,
                message_bytes,
                aggregate_bytes,
                args.iterations,
                args.warmup,
                run_ts,
                git_commit,
                git_dirty,
                flex_opensource_commit,
                flex_opensource_dirty,
                env_vars,
                is_proxy=False,
                notes="",
            )
            all_results.append(result)
            if rank == 0:
                history_rows.append(_result_to_csv_row(result))
                s = result["e2e_us"]
                print(
                    f"  {scenario.name:40s}  e2e={s['mean']:8.1f}us (p99={s['p99']:8.1f})  "
                    f"{message_bytes / 1024 / 1024:7.1f}MB  "
                    f"{result['throughput_gbps']:6.2f} GB/s  | {scenario.description}"
                )

    # --- Model-derived MoE all-to-all PROXY ---
    if args.model_bench in ("alltoall_proxy", "all"):
        moe_models = filter_by_names(
            args.models, MOE_MODELS, valid_names=all_model_names
        )
        workload_points = filter_by_names(args.workload_points, WORKLOAD_POINTS)
        scenarios = generate_moe_scenarios(moe_models, workload_points, size)

        if rank == 0:
            print(
                f"# Model-derived MoE all-to-all PROXY scenarios: {len(scenarios)} "
                f"({len(moe_models)} models x {len(workload_points)} workload points)"
            )
            print(
                "  *** PROXY -- lower bound, not a production all-to-all measurement ***"
            )

        proxy_notes = (
            "PROXY: uniform-routing approximation via pairwise sendrecv, not a measured "
            "production all-to-all (no topology-aware routing, no variable-size protocol, "
            "no compute overlap)"
        )
        for scenario in scenarios:
            element_bytes = _element_bytes(scenario.dtype)
            if scenario.elements < 64:
                if rank == 0:
                    print(
                        f"  SKIP {scenario.name}: {scenario.elements} elements < minimum 64 per pair"
                    )
                continue

            e2e = bench_alltoall_proxy(
                scenario.tokens_to_expert,
                scenario.hidden_size,
                scenario.dtype,
                args.iterations,
                args.warmup,
            )
            if e2e is None:
                continue

            message_bytes = scenario.elements * element_bytes
            aggregate_bytes = (
                message_bytes * (size - 1) * 2
            )  # send+recv across all partners

            result = _build_scenario_result(
                scenario,
                rank,
                e2e,
                message_bytes,
                aggregate_bytes,
                args.iterations,
                args.warmup,
                run_ts,
                git_commit,
                git_dirty,
                flex_opensource_commit,
                flex_opensource_dirty,
                env_vars,
                is_proxy=True,
                notes=proxy_notes,
            )
            all_results.append(result)
            if rank == 0:
                history_rows.append(_result_to_csv_row(result))
                s = result["e2e_us"]
                print(
                    f"  {scenario.name:40s}  e2e={s['mean']:8.1f}us (p99={s['p99']:8.1f})  "
                    f"{message_bytes / 1024:7.1f}KB/pair  "
                    f"{result['throughput_gbps']:6.2f} GB/s (aggregate, PROXY)  | {scenario.description}"
                )

    # --- Write results ---
    dist.barrier()
    if args.json and rank == 0:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.json}")

    if history_rows and not args.no_history and rank == 0:
        append_history_rows(history_csv_path, history_rows)
        print(f"Appended {len(history_rows)} row(s) to {history_csv_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
