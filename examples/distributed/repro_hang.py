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

"""Minimal single-allreduce reproducer for the RuntimeStream::synchronize()
lost-completion hang.

Runs exactly ONE allreduce at a chosen size (no warmup, no iteration loop, no
other collectives) and prints a marker before/after each step, per rank, with
flush=True. This isolates whether a given size hangs, and pins down the exact
point (allreduce dispatch vs synchronize) at which it wedges.

Usage:
    # default: 128MB (the size that hangs in the full benchmark)
    torchrun --nproc-per-node 4 --no-python examples/distributed/repro_hang.py

    # pick a size in MB (fp16); 32 passes, 128 hangs in observed runs
    torchrun --nproc-per-node 4 --no-python examples/distributed/repro_hang.py --mb 32

    # pair with flex instrumentation to see which op's completion is lost:
    SPYRE_DEBUG_INFLIGHT=1 DTLOG_LEVEL=error \
      torchrun --nproc-per-node 4 --no-python \
      examples/distributed/repro_hang.py --mb 128
"""

import argparse
import os

import torch
import torch.distributed as dist

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"


def _log(rank, msg):
    """Rank-tagged, flushed print so ordering is visible even mid-hang."""
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Minimal allreduce hang reproducer")
    parser.add_argument(
        "--mb",
        type=float,
        default=128.0,
        help="AllReduce buffer size in MB (fp16). Default 128 (observed to hang).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Number of allreduce iterations (default 1 — minimal repro).",
    )
    args = parser.parse_args()

    dist.init_process_group(backend=C10D_BACKEND)
    rank = dist.get_rank()
    size = dist.get_world_size()

    # fp16 => 2 bytes/element
    elements = int(args.mb * 1024 * 1024 / 2)

    _log(rank, f"init done: world_size={size}, size={args.mb}MB, elements={elements}")

    # Allocate + fill on device. Each rank contributes (rank+1); after allreduce
    # every element should equal sum(1..size) = size*(size+1)/2.
    t = torch.zeros(elements, dtype=torch.float16)
    t.fill_(float(rank + 1))
    t = t.to(DEVICE)
    _log(rank, "device tensor ready")

    expected = float(size * (size + 1) // 2)

    all_ok = True
    for i in range(args.iters):
        # Re-fill each iteration so every all_reduce is an INDEPENDENT reduction
        # of (rank+1) across ranks (expected == sum(1..size)).  Without this,
        # in-place all_reduce would fold the previous result back in and each
        # iteration would multiply by `size` (e.g. 10, 40, 160, ... = 10*size^i),
        # which is a harness artifact, not a collective error.
        t.fill_(float(rank + 1))
        torch.spyre.synchronize()
        _log(rank, f"iter {i}: BEFORE all_reduce")
        dist.all_reduce(t)
        _log(rank, f"iter {i}: all_reduce returned (Work dispatched); BEFORE synchronize")
        torch.spyre.synchronize()
        _log(rank, f"iter {i}: AFTER synchronize (completed)")

        got = t[0].item()
        ok = abs(got - expected) < 1e-2
        all_ok = all_ok and ok
        _log(rank, f"iter {i}: result[0]={got} expected={expected} {'OK' if ok else 'WRONG'}")

    _log(rank, f"ALL ITERS {'OK' if all_ok else 'WRONG'}")

    dist.barrier()
    _log(rank, "barrier passed; done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
