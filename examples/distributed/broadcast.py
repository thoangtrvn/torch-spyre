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

# All selectable broadcast algorithms (from collective_algo.cpp)
BROADCAST_ALGOS = [
    "Unicast",
    "Linear",
    "Hypertree",
]


def run_test(expected_tensor, comm_rank, comm_size, algo=None):
    """Run a broadcast test with the given expected tensor."""
    global DEVICE
    if 0 != comm_rank:
        x = torch.ones_like(expected_tensor)
    else:
        x = expected_tensor

    algo_label = algo if algo else os.environ.get("COLL_BROADCAST_ALGO", "(default)")

    # Set algorithm via env var if specified
    old_algo = os.environ.get("COLL_BROADCAST_ALGO")
    if algo:
        os.environ["COLL_BROADCAST_ALGO"] = algo

    # Send input tensor to Spyre device
    print("-" * 70)
    print(f"[{comm_rank} of {comm_size}] Tensor Input: {x.shape}")
    print(f"[{comm_rank} of {comm_size}] {x[:10]}")
    x_device = x.to(DEVICE)

    # Broadcast with the collective library
    print(f"[{comm_rank} of {comm_size}] Broadcast Tensor algo={algo_label}: Spyre")
    dist.broadcast(x_device, 0)

    # Restore env var
    if algo:
        if old_algo is not None:
            os.environ["COLL_BROADCAST_ALGO"] = old_algo
        else:
            os.environ.pop("COLL_BROADCAST_ALGO", None)

    result = x_device.to("cpu")
    print(f"[{comm_rank} of {comm_size}] Tensor after collective")
    print(f"[{comm_rank} of {comm_size}] {result[:10]}")

    # Check the result
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
    # - If COLL_BROADCAST_ALGO is set, test only that one (backward compatible)
    # - Otherwise, test all selectable algorithms
    env_algo = os.environ.get("COLL_BROADCAST_ALGO")
    if env_algo:
        algos_to_test = [env_algo]
    else:
        algos_to_test = BROADCAST_ALGOS

    # Test cases: each is an expected tensor that root broadcasts to all ranks
    test_cases = [
        ("small", torch.zeros(128, dtype=torch.float16).fill_(2.0)),
        ("large", torch.zeros(512, 1024, dtype=torch.float16).fill_(4.0)),
    ]

    passed = 0
    failed = 0
    for algo in algos_to_test:
        for label, exp_tensor in test_cases:
            ok = run_test(exp_tensor, comm_rank, comm_size, algo=algo)
            if ok:
                passed += 1
            else:
                failed += 1
            dist.barrier()

    # Summary
    if comm_rank == 0:
        print("=" * 70)
        print(f"# BROADCAST RESULTS: {passed} passed, {failed} failed")
        print("=" * 70)

    dist.destroy_process_group()

    if failed > 0:
        raise RuntimeError(f"{failed} broadcast algorithm(s) failed")
