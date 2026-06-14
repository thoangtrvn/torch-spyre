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

# All selectable allgather algorithms (from collective_algo.cpp)
ALLGATHER_ALGOS = [
    "PairwisePow2",
    "AllBcast",
    "RingExchange",
    "GatherBcast",
    "Copyless",
]


def run_test(comm_rank, comm_size, algo=None):
    """Run an allgather test where each rank contributes and receives from all."""
    global DEVICE

    # Each rank creates a tensor filled with its rank+1 value
    input_tensor = torch.zeros(128, dtype=torch.float16)
    input_tensor.fill_(float(comm_rank + 1))

    algo_label = algo if algo else os.environ.get("COLL_ALLGATHER_ALGO", "(default)")
    print("-" * 70)
    print(f"[{comm_rank} of {comm_size}] Input Tensor: {input_tensor.shape}")
    print(f"[{comm_rank} of {comm_size}] {input_tensor[:10]}")

    # Set algorithm via env var if specified
    old_algo = os.environ.get("COLL_ALLGATHER_ALGO")
    if algo:
        os.environ["COLL_ALLGATHER_ALGO"] = algo

    # Send input tensor to Spyre device
    input_device = input_tensor.to(DEVICE)

    # Prepare output tensors - all ranks need this for allgather
    output_list = [torch.zeros_like(input_device) for _ in range(comm_size)]

    # Allgather with the collective library
    print(f"[{comm_rank} of {comm_size}] Allgather Tensor algo={algo_label}: Spyre")
    dist.all_gather(output_list, input_device)

    # Restore env var
    if algo:
        if old_algo is not None:
            os.environ["COLL_ALLGATHER_ALGO"] = old_algo
        else:
            os.environ.pop("COLL_ALLGATHER_ALGO", None)

    # Check the result at all ranks
    print(f"[{comm_rank} of {comm_size}] Gathered tensors from all ranks:")
    all_correct = True
    for rank_idx in range(comm_size):
        result = output_list[rank_idx].to("cpu")
        expected_value = float(rank_idx + 1)
        expected_tensor = torch.zeros(128, dtype=torch.float16)
        expected_tensor.fill_(expected_value)

        print(f"  From rank {rank_idx}: {result[:10]}")

        if torch.allclose(result, expected_tensor):
            print(f"  Rank {rank_idx} tensor is correct")
        else:
            print(f"  Rank {rank_idx} tensor is incorrect!")
            all_correct = False

    if all_correct:
        print(f"[{comm_rank} of {comm_size}] PASS: algo={algo_label}")
        return True
    else:
        print(
            f"[{comm_rank} of {comm_size}] FAIL: algo={algo_label} "
            f"some gathered tensors are incorrect"
        )
        return False


if __name__ == "__main__":
    # Check that the c10d backend was loaded properly
    if dist.distributed_c10d.is_backend_available(C10D_BACKEND) is False:
        raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
    if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
        raise RuntimeError(
            f"Error: Missing a C10 Backend for {'spyre'}! Expected {C10D_BACKEND}"
        )

    # Initialize the distributed environment
    # Add 'cpu:gloo' since we want to use the backend as well
    print("# Initialize Distributed Group ")
    dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

    comm_size = dist.get_world_size()
    comm_rank = dist.get_rank()

    # Determine which algorithm(s) to test:
    # - If COLL_ALLGATHER_ALGO is set, test only that one (backward compatible)
    # - Otherwise, test all selectable algorithms
    env_algo = os.environ.get("COLL_ALLGATHER_ALGO")
    if env_algo:
        algos_to_test = [env_algo]
    else:
        algos_to_test = ALLGATHER_ALGOS
        # PairwisePow2 requires even-sized world
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
        print(f"# ALLGATHER RESULTS: {passed} passed, {failed} failed")
        print("=" * 70)

    dist.destroy_process_group()

    if failed > 0:
        raise RuntimeError(f"{failed} allgather algorithm(s) failed")

# Made with Bob
