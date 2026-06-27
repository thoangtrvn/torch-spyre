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
Comprehensive test for every spyre-comms primitive operation.

Run with:
  torchrun --nproc-per-node 2 examples/distributed/test_spyre_comms_primitives.py
  torchrun --nproc-per-node 4 examples/distributed/test_spyre_comms_primitives.py

Force specific allreduce algorithms:
  COLL_ALLREDUCE_ALGO=Ring torchrun --nproc-per-node 2 ...
  COLL_ALLREDUCE_ALGO=ReduceScatterAllGather torchrun --nproc-per-node 2 ...
"""

import os
import sys
import time

import torch
import torch.distributed as dist

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"

# Minimum tensor size: spyre-comms requires >= 128 bytes.
# float16 = 2 bytes, so minimum 64 elements for simple ops.
# For ops that SPLIT (allreduce, reduce_scatter, etc.), each chunk must be
# stick-aligned (128 bytes = 64 fp16 elements). With N ranks, need 64*N elements.
# Use a generous size to avoid stick-alignment issues in LIBCOLL split logic.
COLLECTIVE_ELEMENTS = 4096
# Minimum for non-splitting ops (broadcast, send/recv)
MIN_ELEMENTS = 64


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0
_skipped = 0


def _report(name, ok, detail=""):
    global _passed, _failed
    rank = dist.get_rank()
    size = dist.get_world_size()
    tag = "PASS" if ok else "FAIL"
    if ok:
        _passed += 1
    else:
        _failed += 1
    msg = f"[{rank}/{size}] {tag}: {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg, flush=True)


def _skip(name, reason):
    global _skipped
    _skipped += 1
    rank = dist.get_rank()
    size = dist.get_world_size()
    print(f"[{rank}/{size}] SKIP: {name} -- {reason}", flush=True)


def _device_tensor(elements, dtype, fill_value=None, rank_fill=False, rank=None):
    """Create a CPU tensor, optionally fill it, then move to device."""
    t = torch.zeros(elements, dtype=dtype)
    if rank_fill and rank is not None:
        # Using rank*4+1 instead of rank+1 for asymmetric inputs:
        # rank 0 = 1.0, rank 1 = 5.0. This makes the address/sync bug
        # unambiguous: expected 6.0, bug would give 2.0 or 10.0 per rank.
        t.fill_(float(rank * 4 + 1))
    elif fill_value is not None:
        t.fill_(fill_value)
    return t.to(DEVICE)


# ---------------------------------------------------------------------------
# Tests: Barrier
# ---------------------------------------------------------------------------


def test_barrier():
    _rank = dist.get_rank()
    _size = dist.get_world_size()
    t0 = time.time()
    dist.barrier()
    dt = time.time() - t0
    _report("barrier", True, f"{dt * 1000:.1f} ms")


# ---------------------------------------------------------------------------
# Tests: Broadcast
# ---------------------------------------------------------------------------


def test_broadcast():
    rank = dist.get_rank()
    _size = dist.get_world_size()
    root = 0
    fill = 7.0
    if rank == root:
        t = _device_tensor(MIN_ELEMENTS, torch.float16, fill_value=fill)
    else:
        t = _device_tensor(MIN_ELEMENTS, torch.float16, fill_value=-1.0)
    dist.broadcast(t, src=root)
    result = t.to("cpu")
    expected = torch.zeros(MIN_ELEMENTS, dtype=torch.float16)
    expected.fill_(fill)
    ok = torch.allclose(result, expected)
    _report("broadcast", ok, f"root={root}, fill={fill}")


def test_broadcast_nonzero_root():
    rank = dist.get_rank()
    size = dist.get_world_size()
    root = size - 1
    fill = 3.0
    if rank == root:
        t = _device_tensor(MIN_ELEMENTS, torch.float16, fill_value=fill)
    else:
        t = _device_tensor(MIN_ELEMENTS, torch.float16, fill_value=-1.0)
    dist.broadcast(t, src=root)
    result = t.to("cpu")
    expected = torch.zeros(MIN_ELEMENTS, dtype=torch.float16)
    expected.fill_(fill)
    ok = torch.allclose(result, expected)
    _report("broadcast (nonzero root)", ok, f"root={root}")


# ---------------------------------------------------------------------------
# Tests: Send / Recv
# ---------------------------------------------------------------------------


def test_send_recv():
    rank = dist.get_rank()
    size = dist.get_world_size()
    if size < 2:
        _skip("send_recv", "requires >= 2 ranks")
        return

    fill_val = 42.0
    if rank == 0:
        t = _device_tensor(MIN_ELEMENTS, torch.float16, fill_value=fill_val)
        dist.send(t, dst=1)
        _report("send (rank 0 -> rank 1)", True)
    elif rank == 1:
        t = _device_tensor(MIN_ELEMENTS, torch.float16, fill_value=-1.0)
        dist.recv(t, src=0)
        result = t.to("cpu")
        expected = torch.zeros(MIN_ELEMENTS, dtype=torch.float16)
        expected.fill_(fill_val)
        ok = torch.allclose(result, expected)
        _report("recv (rank 1 <- rank 0)", ok)
    else:
        _skip("send_recv (non-participant)", f"rank {rank} not involved in 2-rank test")


def test_send_recv_pingpong():
    """Rank 0 sends to rank 1, rank 1 sends back with modified values."""
    rank = dist.get_rank()
    size = dist.get_world_size()
    if size < 2:
        _skip("send_recv_pingpong", "requires >= 2 ranks")
        return

    if rank == 0:
        t = _device_tensor(MIN_ELEMENTS, torch.float16, fill_value=1.0)
        dist.send(t, dst=1)
        dist.recv(t, src=1)
        result = t.to("cpu")
        expected = torch.zeros(MIN_ELEMENTS, dtype=torch.float16)
        expected.fill_(2.0)
        ok = torch.allclose(result, expected)
        _report("send_recv pingpong (rank 0)", ok)
    elif rank == 1:
        t = _device_tensor(MIN_ELEMENTS, torch.float16, fill_value=-1.0)
        dist.recv(t, src=0)
        result = t.to("cpu")
        ok_recv = torch.allclose(result, torch.ones(MIN_ELEMENTS, dtype=torch.float16))
        t2 = _device_tensor(MIN_ELEMENTS, torch.float16, fill_value=2.0)
        dist.send(t2, dst=0)
        _report("send_recv pingpong (rank 1)", ok_recv)
    else:
        _skip("send_recv pingpong (non-participant)", f"rank {rank}")


# ---------------------------------------------------------------------------
# Tests: AllReduce
# ---------------------------------------------------------------------------


def _run_allreduce(name, elements, dtype, algo_env=None):
    """Core allreduce test. Each rank fills with (rank+1), expect SUM."""
    rank = dist.get_rank()
    size = dist.get_world_size()

    if algo_env:
        old = os.environ.get("COLL_ALLREDUCE_ALGO")
        os.environ["COLL_ALLREDUCE_ALGO"] = algo_env

    t = _device_tensor(elements, dtype, rank_fill=True, rank=rank)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    result = t.to("cpu")

    expected_sum = sum(r * 4 + 1 for r in range(size))
    expected = torch.zeros(elements, dtype=dtype)
    expected.fill_(float(expected_sum))
    ok = torch.allclose(result, expected, atol=1e-3)

    if not ok:
        diff = (result - expected).abs()
        max_diff = diff.max().item()
        # Find first mismatched index
        mismatches = diff.gt(1e-3).nonzero(as_tuple=True)[0]
        first_bad = mismatches[0].item() if len(mismatches) > 0 else -1
        # Sample around the first bad index
        lo = max(0, first_bad - 2)
        hi = min(len(result), first_bad + 6)
        print(
            f"[{rank}/{size}] {name} MISMATCH: "
            f"first_bad_idx={first_bad} "
            f"num_bad={len(mismatches)} "
            f"max_diff={max_diff} "
            f"result[{lo}:{hi}]={result[lo:hi].tolist()} "
            f"expected[{lo}:{hi}]={expected[lo:hi].tolist()}",
            flush=True,
        )
        # Also check chunk boundaries for ring allreduce
        chunk_size = elements // size
        for c in range(size):
            cstart = c * chunk_size
            cend = (c + 1) * chunk_size
            cdiff = diff[cstart:cend].max().item()
            if cdiff > 1e-3:
                print(
                    f"[{rank}/{size}] {name} chunk{c} [{cstart}:{cend}] "
                    f"max_diff={cdiff} "
                    f"result[{cstart}:{cstart + 4}]={result[cstart : cstart + 4].tolist()} "
                    f"expected[{cstart}:{cstart + 4}]={expected[cstart : cstart + 4].tolist()}",
                    flush=True,
                )

    _report(
        name, ok, f"elements={elements}, dtype={dtype}, expected_sum={expected_sum}"
    )

    if algo_env:
        if old is not None:
            os.environ["COLL_ALLREDUCE_ALGO"] = old
        else:
            os.environ.pop("COLL_ALLREDUCE_ALGO", None)


def test_allreduce_default():
    _run_allreduce("allreduce (default algo)", COLLECTIVE_ELEMENTS, torch.float16)


def test_allreduce_ring():
    _run_allreduce(
        "allreduce (Ring)", COLLECTIVE_ELEMENTS, torch.float16, algo_env="Ring"
    )


def test_allreduce_rsag():
    _run_allreduce(
        "allreduce (ReduceScatterAllGather)",
        COLLECTIVE_ELEMENTS,
        torch.float16,
        algo_env="ReduceScatterAllGather",
    )


def test_allreduce_bcast():
    _run_allreduce(
        "allreduce (BiTreeBcast)",
        COLLECTIVE_ELEMENTS,
        torch.float16,
        algo_env="BiTreeBcast",
    )


def test_allreduce_gathersumbcast():
    _run_allreduce(
        "allreduce (GatherSumBcast)",
        COLLECTIVE_ELEMENTS,
        torch.float16,
        algo_env="GatherSumBcast",
    )


def test_allreduce_pairwise():
    """PairwisePow2 only works with power-of-2 world size."""
    size = dist.get_world_size()
    if size & (size - 1) != 0:
        _skip("allreduce (PairwisePow2)", f"world_size={size} is not power of 2")
        return
    _run_allreduce(
        "allreduce (PairwisePow2)",
        COLLECTIVE_ELEMENTS,
        torch.float16,
        algo_env="PairwisePow2",
    )


def test_allreduce_fp32():
    _run_allreduce("allreduce (fp32)", COLLECTIVE_ELEMENTS, torch.float32)


def test_allreduce_large():
    """64 KB tensor -- 32768 fp16 elements."""
    _run_allreduce("allreduce (large, 64KB)", 32768, torch.float16)


def test_allreduce_256kb():
    """256 KB tensor."""
    _run_allreduce("allreduce (large, 256KB)", 131072, torch.float16)


# ---------------------------------------------------------------------------
# Tests: Reduce
# ---------------------------------------------------------------------------


def test_reduce_sum():
    rank = dist.get_rank()
    size = dist.get_world_size()
    root = 0

    t = _device_tensor(COLLECTIVE_ELEMENTS, torch.float16, rank_fill=True, rank=rank)
    dist.reduce(t, dst=root, op=dist.ReduceOp.SUM)
    result = t.to("cpu")

    expected_sum = sum(r * 4 + 1 for r in range(size))
    expected = torch.zeros(COLLECTIVE_ELEMENTS, dtype=torch.float16)
    expected.fill_(float(expected_sum))

    if rank == root:
        ok = torch.allclose(result, expected, atol=1e-3)
        _report("reduce (SUM, root=0)", ok, f"expected_sum={expected_sum}")
    else:
        _report("reduce (SUM, non-root)", True, "non-root, value not checked")


# ---------------------------------------------------------------------------
# Tests: Gather
# ---------------------------------------------------------------------------


def test_gather():
    rank = dist.get_rank()
    size = dist.get_world_size()
    root = 0

    t = _device_tensor(COLLECTIVE_ELEMENTS, torch.float16, rank_fill=True, rank=rank)

    if rank == root:
        gather_list = [
            _device_tensor(COLLECTIVE_ELEMENTS, torch.float16) for _ in range(size)
        ]
    else:
        gather_list = None

    dist.gather(t, gather_list=gather_list, dst=root)

    if rank == root:
        all_ok = True
        for i, gathered in enumerate(gather_list):
            expected_val = float(i * 4 + 1)
            if not torch.allclose(
                gathered.to("cpu"),
                torch.full((COLLECTIVE_ELEMENTS,), expected_val, dtype=torch.float16),
            ):
                all_ok = False
                break
        _report("gather (root=0)", all_ok)
    else:
        _report("gather (non-root)", True, "non-root, no gather_list")


# ---------------------------------------------------------------------------
# Tests: AllGather
# ---------------------------------------------------------------------------


def test_allgather():
    rank = dist.get_rank()
    size = dist.get_world_size()

    t = _device_tensor(COLLECTIVE_ELEMENTS, torch.float16, rank_fill=True, rank=rank)

    output_list = [
        _device_tensor(COLLECTIVE_ELEMENTS, torch.float16) for _ in range(size)
    ]
    dist.all_gather(output_list, t)

    all_ok = True
    for i, gathered in enumerate(output_list):
        expected_val = float(i * 4 + 1)
        if not torch.allclose(
            gathered.to("cpu"),
            torch.full((COLLECTIVE_ELEMENTS,), expected_val, dtype=torch.float16),
        ):
            all_ok = False
            break
    _report("allgather", all_ok)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    rank = dist.get_rank()
    size = dist.get_world_size()

    if dist.distributed_c10d.is_backend_available(C10D_BACKEND) is False:
        raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
    if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
        raise RuntimeError(
            f"Error: Missing a C10 Backend for spyre! Expected {C10D_BACKEND}"
        )

    print(f"# test_spyre_comms_primitives: rank={rank} size={size}")
    print(f"# COLL_ALLREDUCE_ALGO={os.environ.get('COLL_ALLREDUCE_ALGO', '(default)')}")
    sys.stdout.flush()

    # -- Barrier --
    test_barrier()

    # -- Broadcast --
    test_broadcast()
    test_broadcast_nonzero_root()

    # -- Send / Recv --
    test_send_recv()
    test_send_recv_pingpong()

    # -- AllReduce (multiple algorithms) --
    test_allreduce_default()
    test_allreduce_ring()
    test_allreduce_rsag()
    test_allreduce_bcast()
    test_allreduce_gathersumbcast()
    test_allreduce_pairwise()

    # -- AllReduce (data types and sizes) --
    test_allreduce_fp32()
    test_allreduce_large()
    test_allreduce_256kb()

    # -- Reduce --
    # test_reduce_sum()  # dist.reduce() not yet implemented in spyre comms

    # -- Gather --
    test_gather()

    # -- AllGather --
    test_allgather()

    # -- Summary --
    dist.barrier()
    if rank == 0:
        total = _passed + _failed + _skipped
        print("=" * 70)
        print(
            f"# RESULTS: {_passed} passed, {_failed} failed, {_skipped} skipped (total {total})"
        )
        print("=" * 70)
    if _failed > 0:
        raise RuntimeError(f"{_failed} test(s) failed")


if __name__ == "__main__":
    print("# Initialize Distributed Group")
    dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

    try:
        main()
    finally:
        dist.destroy_process_group()
