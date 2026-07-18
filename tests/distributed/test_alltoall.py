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

"""Distributed all-to-all tests for the Spyre CCL backend (Phase 1).

Phase 1 supports the BALANCED (equal per-peer) case, serial and correctness-first
(see flex-opensource/docs/alltoall-transport-design.md §13).

alltoall_base decomposes each peer's dim-0 sub-chunk with layout-aware torch ops
(narrow + clone) into a fresh standard-layout device tensor, moves it as a WHOLE
tensor (the layout-safe path), and scatters it back with copy_ into a narrowed
output view. This works for N-D tensors: all tiled-layout handling is delegated to
torch-spyre's own ops, and the dim-0 (row-major) storage_offset handling in
copy_from_d2d has been fixed in torch-spyre core. A 1-D tensor's dim-0 is the stick
dimension, so 1-D splits must be stick-aligned (multiples of 64 fp16 elements);
>=2D dim-0 offsets are unrestricted. Asymmetric (per-peer send != recv) split sizes
fail fast (Phase 2). The list-form all_to_all (whole tensor per peer) is also
supported.

Run (needs Spyre hardware + torchrun), from torch-spyre/tests/distributed/,
e.g. at world_size 4:

    torchrun --nproc-per-node 4 -m pytest test_alltoall.py -v

or via the repo's distributed runner (configs/test_distributed_config.yaml). The
primary correctness signals are test_all_to_all_single_uniform_2d and
test_all_to_all_single_uneven_balanced_splits_2d passing at ws=2 and ws=4.
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
# Temporary hack (mirrors the other distributed tests)
torch.spyre._impl._lazy_init()
# ------------

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"


class TestAllToAll(TestCase):
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

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    # Encode "rank `src` sends to rank `dst`" as a distinct scalar. fp16
    # represents integers exactly well past comm_size*100 + comm_size for any
    # realistic world size, so equality (not allclose) is the right check.
    @staticmethod
    def _tag(src: int, dst: int) -> float:
        return float(src * 100 + dst)

    def _run_single(self, rows_per_peer: int, hidden, dtype) -> None:
        """all_to_all_single with a uniform even split (empty split lists).

        Rank r's input chunk d (rows [d*R, (d+1)*R)) is filled with tag(r, d):
        the data r sends to rank d. After the exchange, rank r's output chunk s
        (received from rank s) must equal tag(s, r).
        """
        n = self.comm_size
        r = self.comm_rank
        shape = (n * rows_per_peer,) if hidden is None else (n * rows_per_peer, hidden)

        cpu_in = torch.empty(shape, dtype=dtype)
        for dst in range(n):
            cpu_in[dst * rows_per_peer : (dst + 1) * rows_per_peer] = self._tag(r, dst)
        x = cpu_in.to(DEVICE)
        out = torch.empty(shape, dtype=dtype, device=DEVICE)

        dist.all_to_all_single(out, x)

        res = out.to("cpu")
        for src in range(n):
            expected = self._tag(src, r)
            block = res[src * rows_per_peer : (src + 1) * rows_per_peer]
            self.assertTrue(
                torch.equal(block, torch.full_like(block, expected)),
                f"Rank {r}: chunk from src {src} wrong; expected {expected}, "
                f"got a block starting {block.flatten()[:4].tolist()}",
            )

    def test_all_to_all_single_uniform_1d(self) -> None:
        # 1-D fp16: row_bytes = 2, so rows_per_peer must be a multiple of 64 for
        # each chunk to be stick (128B) aligned (64 rows => 128 bytes). This is
        # the primary correctness signal for alltoall_base (raw-offset slicing of
        # the caller's long-lived tensor) at ws=2 / ws=4.
        self._run_single(rows_per_peer=64, hidden=None, dtype=torch.float16)

    def test_all_to_all_single_explicit_balanced_splits_1d(self) -> None:
        # Non-empty but balanced (equal) 1-D split lists — exercises the split
        # code path while staying in the supported (1-D, balanced, stick-aligned)
        # regime.
        n = self.comm_size
        r = self.comm_rank
        rows_per_peer = 64  # multiple of 64 => stick-aligned in 1-D fp16
        splits = [rows_per_peer] * n
        shape = (n * rows_per_peer,)

        cpu_in = torch.empty(shape, dtype=torch.float16)
        for dst in range(n):
            cpu_in[dst * rows_per_peer : (dst + 1) * rows_per_peer] = self._tag(r, dst)
        x = cpu_in.to(DEVICE)
        out = torch.empty(shape, dtype=torch.float16, device=DEVICE)

        dist.all_to_all_single(out, x, splits, splits)

        res = out.to("cpu")
        for src in range(n):
            expected = self._tag(src, r)
            block = res[src * rows_per_peer : (src + 1) * rows_per_peer]
            self.assertTrue(torch.equal(block, torch.full_like(block, expected)))

    def test_all_to_all_single_uniform_2d(self) -> None:
        # 2-D [tokens, hidden] — the real MoE/DCP shape. Exercises the
        # layout-aware narrow/clone/copy_ decomposition on a tiled tensor,
        # unblocked by the copy_from_d2d dim-0 storage_offset fix. Primary 2-D
        # correctness signal.
        self._run_single(rows_per_peer=32, hidden=4096, dtype=torch.float16)

    def test_all_to_all_single_uneven_balanced_splits_2d(self) -> None:
        # Non-uniform per-peer split sizes on a 2-D tensor, so narrow() lands at
        # arbitrary (non-64-aligned) dim-0 row offsets and the scatter writes into
        # arbitrary output sub-regions — the exact case the copy_from_d2d dim-0
        # offset fix targets (dim-0 is NOT the stick dimension for a 2-D tensor).
        #
        # The split matrix MUST be consistent across ranks: all_to_all_single
        # requires rank i's send-to-j (in_splits[j]) == rank j's recv-from-i
        # (out_splits[i]). Combined with the Phase-1 balanced requirement
        # (in_splits == out_splits on each rank), the matrix S[i][j] must be
        # SYMMETRIC. S[i][j] = base + i + j is symmetric, non-uniform per peer,
        # and not a multiple of 64. Rank r's split list is row r of S. (Using the
        # SAME list on every rank would be a MALFORMED collective — S[i][j]=f(j)
        # is not symmetric — producing per-leg size mismatches, not a real test.)
        n = self.comm_size
        r = self.comm_rank
        hidden = 4096
        base = 16
        splits = [base + r + j for j in range(n)]  # row r of symmetric S[i][j]=base+i+j
        total = sum(splits)
        off = [sum(splits[:p]) for p in range(n)]

        cpu_in = torch.empty((total, hidden), dtype=torch.float16)
        for dst in range(n):
            cpu_in[off[dst] : off[dst] + splits[dst]] = self._tag(r, dst)
        x = cpu_in.to(DEVICE)
        out = torch.empty((total, hidden), dtype=torch.float16, device=DEVICE)

        dist.all_to_all_single(out, x, splits, splits)

        res = out.to("cpu")
        for src in range(n):
            expected = self._tag(src, r)
            block = res[off[src] : off[src] + splits[src]]
            self.assertTrue(
                torch.equal(block, torch.full_like(block, expected)),
                f"Rank {r}: chunk from src {src} wrong; expected {expected}, "
                f"got a block starting {block.flatten()[:4].tolist()}",
            )

    def test_all_to_all_list_form(self) -> None:
        # List-of-tensors form -> SpyreCCLBackend::alltoall. Each element is a
        # whole tensor (no slicing).
        n = self.comm_size
        r = self.comm_rank
        hidden = 4096
        rows = 32

        inputs = [
            torch.full((rows, hidden), self._tag(r, dst), dtype=torch.float16).to(
                DEVICE
            )
            for dst in range(n)
        ]
        outputs = [
            torch.empty((rows, hidden), dtype=torch.float16, device=DEVICE)
            for _ in range(n)
        ]

        dist.all_to_all(outputs, inputs)

        for src in range(n):
            expected = self._tag(src, r)
            block = outputs[src].to("cpu")
            self.assertTrue(torch.equal(block, torch.full_like(block, expected)))

    def test_asymmetric_splits_fail_fast(self) -> None:
        # Phase 1 does not support asymmetric (per-peer send != recv) splits; the
        # up-front balanced guard must raise rather than silently transfer wrong
        # data.
        #
        # Use a globally-consistent asymmetric matrix S[i][j] = base + i (rank i
        # sends base+i rows to every peer). Then on rank r: in_splits[p] = base+r
        # (sends) and out_splits[s] = base+s (recvs), which is consistent across
        # ranks (rank r's send to p == rank p's recv from r) yet per-leg
        # asymmetric for every non-self leg (base+r != base+p when r != p). The
        # guard validates all peers before any transfer, so every rank raises
        # locally with no risk of a hang.
        n = self.comm_size
        if n < 2:
            self.skipTest("needs world_size >= 2")
        base = 16
        r = self.comm_rank
        in_splits = [base + r for _ in range(n)]
        out_splits = [base + s for s in range(n)]
        x = torch.ones((sum(in_splits),), dtype=torch.float16, device=DEVICE)
        out = torch.empty((sum(out_splits),), dtype=torch.float16, device=DEVICE)
        with self.assertRaises(Exception):
            dist.all_to_all_single(out, x, out_splits, in_splits)


if __name__ == "__main__":
    run_tests()
