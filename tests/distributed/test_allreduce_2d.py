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

"""Correctness regression for 2-D [tokens, hidden] all_reduce(SUM) on the Spyre
CCL backend -- the tensor-parallel activation reduce.

A >=2-D all_reduce routes through SpyreCCLBackend::allreduce_2d_compose, which
has two HW-verified paths (chunked reduce_scatter+all_gather by default;
whole-tensor all_gather+on-device-sum when SPYRE_ALLREDUCE_2D_WHOLE_TENSOR=1,
and always for tokens<world). This test pins that BOTH paths reduce fully and
correctly across:
  * even token split       [8, 4096]  (tokens % world == 0 at world 2/4)
  * uneven token split     [6, 4096]  (tokens % world != 0 at world 4)
  * decode / tokens<world  [1, 4096]  (forces the whole-tensor path)
  * larger prefill         [32, 4096]
on the WORLD group and, at world_size>=2, on a [0,1] subgroup.

Historically this shape SILENTLY under-reduced at 4 ranks (only numel/ROWS
elements summed) and later CRASHED at compile ("no valid candidate"); both were
the transposed torch.ones layout, fixed by the aten::ones/aten::full kernels.
This test guards against regression of that fix.

Run (needs Spyre hardware + torchrun), from torch-spyre/tests/distributed/:
    torchrun --nproc-per-node 4 -m pytest test_allreduce_2d.py -v -m "not upstream"
    SPYRE_ALLREDUCE_2D_WHOLE_TENSOR=1 torchrun --nproc-per-node 4 -m pytest test_allreduce_2d.py -v
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests, TestCase

if "RANK" not in os.environ:
    pytest.skip(
        "RANK environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

if "WORLD_SIZE" not in os.environ:
    pytest.skip(
        "WORLD_SIZE environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

try:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size < 2:
        pytest.skip(
            f"WORLD_SIZE is {world_size}, need at least 2 for distributed tests",
            allow_module_level=True,
        )
except ValueError:
    pytest.skip(
        "WORLD_SIZE environment variable is not a valid integer, skipping tests",
        allow_module_level=True,
    )

# ------------
# Temporary hack (mirrors the other distributed tests, e.g. test_subgroup.py).
torch.spyre._impl._lazy_init()
# ------------

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"
HIDDEN = 4096

# (tokens, hidden): even split, uneven split (at world 4), decode/tokens<world,
# larger prefill. hidden=4096 = 64 fp16 sticks/row.
SHAPES = [(8, HIDDEN), (6, HIDDEN), (1, HIDDEN), (32, HIDDEN)]


def _world_all_succeeded(local_ok: bool) -> bool:
    """WORLD consensus over a gloo (CPU) flag so a subgroup-only failure fails
    ALL ranks cleanly instead of hanging non-members on the next collective."""
    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int64, device="cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=dist.group.WORLD,
                    async_op=False)
    return bool(flag[0].item() == 1)


class TestAllreduce2D(TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
            raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
        if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
            raise RuntimeError(
                f"Error: Missing a C10 Backend for 'spyre'! Expected {C10D_BACKEND}"
            )
        if not dist.is_initialized():
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")
        cls.comm_size = dist.get_world_size()
        cls.comm_rank = dist.get_rank()
        # A TP-like subgroup [0, 1]. c10d new_group() is collective: every rank
        # must call it (non-members get None). Valid even when world_size == 2.
        cls.tp_members = [0, 1]
        cls.tp = dist.new_group(ranks=cls.tp_members)

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _check_full_reduce(self, group, members):
        """all_reduce SUM an all-ones [tokens, HIDDEN] on `group`; every element
        of the result must equal len(members) (fully reduced, no under-reduce).
        Uses torch.ones (the shape that historically triggered the transposed
        layout) so this test guards the aten::ones/aten::full fix directly."""
        n_members = len(members)
        for tokens, hidden in SHAPES:
            x = torch.ones((tokens, hidden), dtype=torch.float16, device=DEVICE)
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
            got = x.to("cpu")
            want = float(n_members)
            self.assertTrue(
                torch.equal(got, torch.full_like(got, want)),
                f"rank {self.comm_rank}: [{tokens},{hidden}] on {members} "
                f"under/over-reduced: want all=={want}, got "
                f"first4={got.flatten()[:4].tolist()} "
                f"n_wrong={(got != want).sum().item()}/{got.numel()}",
            )

    def test_world_allreduce_2d(self) -> None:
        """2-D all_reduce on the WORLD group reduces fully for every shape
        (even/uneven/decode/prefill)."""
        members = list(range(self.comm_size))
        self._check_full_reduce(dist.group.WORLD, members)

    def test_zz_subgroup_allreduce_2d(self) -> None:
        """2-D all_reduce on a [0,1] TP subgroup reduces fully. Named zz to run
        last (it does not tear the group down, but keeps ordering stable). Only
        subgroup members reduce on it; a partial failure is funnelled through a
        WORLD gloo consensus so non-members never hang."""
        local_ok = True
        err = ""
        if self.comm_rank in self.tp_members:
            try:
                self._check_full_reduce(self.tp, self.tp_members)
            except AssertionError as exc:
                local_ok = False
                err = str(exc).splitlines()[0]
        if not _world_all_succeeded(local_ok):
            self.fail(
                f"subgroup 2-D all_reduce failed on >=1 member; rank "
                f"{self.comm_rank} local_ok={local_ok} err={err!r}"
            )


if __name__ == "__main__":
    run_tests()
