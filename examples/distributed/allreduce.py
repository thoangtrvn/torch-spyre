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
import torch
import torch.distributed as dist
import os

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"

# All selectable allreduce algorithms (from collective_algo.cpp)
ALLREDUCE_ALGOS = [
    "ReduceScatterAllGather",
    "PairwisePow2",
    "BiTreeBcast",
    "LinearBcast",
    "GatherSumBcast",
]


def run_test(comm_rank, comm_size, algo=None):
    """Run an allreduce test where all ranks contribute and all receive the sum."""
    global DEVICE

    # Each rank creates a tensor filled with a distinct value.
    # Using rank*4+1 so rank 0 = 1.0, rank 1 = 5.0 (instead of rank+1 = 1.0, 2.0).
    # Asymmetric inputs make the address/sync bug unambiguous:
    #   Expected: 1+5 = 6.0
    #   Bug (reads own data twice): rank 0 = 1+1 = 2.0, rank 1 = 5+5 = 10.0
    input_tensor = torch.zeros(4096, dtype=torch.float16)
    input_tensor.fill_(float(comm_rank * 4 + 1))

    algo_label = algo if algo else os.environ.get("COLL_ALLREDUCE_ALGO", "(default)")
    print("-" * 70)
    print(
        f"[{comm_rank} of {comm_size}] Input Tensor (Before Allreduce): {input_tensor.shape}"
    )
    print(f"[{comm_rank} of {comm_size}] {input_tensor[:10]}")

    # Set algorithm via env var if specified
    old_algo = os.environ.get("COLL_ALLREDUCE_ALGO")
    if algo:
        os.environ["COLL_ALLREDUCE_ALGO"] = algo

    # Send input tensor to Spyre device
    input_device = input_tensor.to(DEVICE)

    # Allreduce with the collective library (SUM operation)
    print(
        f"[{comm_rank} of {comm_size}] Allreduce Tensor (SUM) algo={algo_label}: Spyre"
    )
    dist.all_reduce(input_device, op=dist.ReduceOp.SUM)

    # Restore env var
    if algo:
        if old_algo is not None:
            os.environ["COLL_ALLREDUCE_ALGO"] = old_algo
        else:
            os.environ.pop("COLL_ALLREDUCE_ALGO", None)

    # Check the result at all ranks
    result = input_device.to("cpu")
    print(f"[{comm_rank} of {comm_size}] Reduced Tensor (SUM of all ranks):")
    print(f"[{comm_rank} of {comm_size}] {result[:10]}")

    # Expected result: sum of (rank*4+1) for each rank, matching the fill values above
    expected_sum = sum(r * 4 + 1 for r in range(comm_size))
    expected_tensor = torch.zeros(4096, dtype=torch.float16)
    expected_tensor.fill_(float(expected_sum))

    print(f"  Expected value per element: {expected_sum}")

    if torch.allclose(result, expected_tensor):
        print(f"[{comm_rank} of {comm_size}] PASS: algo={algo_label}")
        return True
    else:
        print(
            f"[{comm_rank} of {comm_size}] FAIL: algo={algo_label} "
            f"expected {expected_tensor[:10]} but got {result[:10]}"
        )
        return False


if __name__ == "__main__":
    # Check that the c10d backend was loaded properly
    if dist.distributed_c10d.is_backend_available(C10D_BACKEND) is False:
        raise RuntimeError(f"Error: Missing the C10D Backend {C10D_BACKEND}")
    if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
        raise RuntimeError(
            f"Error: Missing a C10D Backend for {'spyre'}! Expected {C10D_BACKEND}"
        )

    # Initialize the distributed environment
    # Add 'cpu:gloo' since we want to use the backend as well
    print("# Initialize Distributed Group ")
    dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

    comm_size = dist.get_world_size()
    comm_rank = dist.get_rank()

    # Determine which algorithm(s) to test:
    # - If COLL_ALLREDUCE_ALGO is set, test only that one (backward compatible)
    # - Otherwise, test all selectable algorithms
    env_algo = os.environ.get("COLL_ALLREDUCE_ALGO")
    if env_algo:
        algos_to_test = [env_algo]
    else:
        algos_to_test = ALLREDUCE_ALGOS
        # PairwisePow2 requires power-of-2 world size
        if comm_size & (comm_size - 1) != 0:
            algos_to_test = [a for a in algos_to_test if a != "PairwisePow2"]

    passed = 0
    failed = 0
    for algo in algos_to_test:
        ok = run_test(comm_rank, comm_size, algo=algo)
        if ok:
            passed += 1
        else:
            failed += 1
        dist.barrier()

    # Summary
    if comm_rank == 0:
        print("=" * 70)
        print(f"# ALLREDUCE RESULTS: {passed} passed, {failed} failed")
        print("=" * 70)

    dist.destroy_process_group()

    if failed > 0:
        raise RuntimeError(f"{failed} allreduce algorithm(s) failed")
