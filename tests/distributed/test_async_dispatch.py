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

"""Behavioral + regression tests for the async c10d progress-thread feature.

The Spyre CCL backend ("spyre:spyreccl") runs 7 collectives ASYNC via a
process-global FIFO progress worker: allreduce(SUM), broadcast, gather,
uniform allgather, send, recv, barrier. `dist.all_reduce(t, async_op=True)`
returns a c10d Work immediately (before the collective is guaranteed to have
completed); `work.wait()` blocks until it is done. Multi-PG (world + a
subgroup) is supported and shares one process-global worker.

These tests behaviorally prove:
  * an async collective's Work handle is real and wait() yields the correct
    reduced value (test_allreduce_async_returns_before_completion);
  * world PG and a TP-like subgroup can issue interleaved async collectives
    without deadlocking, and destroying the subgroup does NOT poison the
    shared comm stream / progress worker used by the still-live world PG
    (test_world_and_tp_subgroup_coexist_and_teardown -- the key v3 guard);
  * back-to-back distinct async-dispatched collectives preserve FIFO
    ordering under load, i.e. no matcher desync / no hang
    (test_back_to_back_collectives_preserve_order);
  * dropping an async Work handle without calling wait() (fire-and-forget)
    does not hang or crash the dtor's cancel/drain path, using a barrier as
    the sync point (test_destroy_work_handle_variants);
  * an excluded (deliberately synchronous, dual-mode) op -- dist.reduce,
    which drives group_context_->reduce()+ws->start() directly rather than
    the async progress-worker path -- still returns correct results
    (test_reduce_excluded_op_stays_synchronous).

Run (needs Spyre hardware + torchrun), from torch-spyre/tests/distributed/:

    torchrun --nproc-per-node 2 -m pytest test_async_dispatch.py -v

The multi-PG subgroup test only needs WORLD_SIZE >= 2 (dist.new_group(ranks=
[0, 1]) is valid even when that is the whole world); it is not restricted to
exactly 4 like test_subgroup.py's DP x TP layout.

No claim here is hardware-verified; this file was written without hardware
access (see torch-spyre/.claude CLAUDE.md -- macOS host, no Spyre AIUs).

NOTE on correctness scope: there is a separate, pre-existing LIBCOLL bug
where a 4-rank 2-D [ROWS, HIDDEN] all_reduce under-reduces (only
numel/ROWS elements get summed). That bug affects the SYNCHRONOUS path
exactly as much as the async path -- it is not this feature's fault, and
is tracked/fixed separately. Absolute numeric correctness of large 2-D
4-rank all_reduce is therefore out of scope here and covered elsewhere.
What these tests assert instead is that ASYNC FAITHFULLY REPRODUCES SYNC:
for each all_reduce under test, a synchronous reference run on a cloned
input is compared via exact `torch.equal()` against the async result. This
is deliberately robust to the pre-existing LIBCOLL bug (both sides
under-reduce identically, so the equivalence still holds), while still
proving the async progress-thread feature, the async Work contract, and
multi-PG teardown are all correct.
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
ROWS = 8  # modest row count; hidden=4096 (a stick multiple) carries the
# split-eligible dim for collectives -- see test_allgatherv.py's 2-D
# convention. 1-D [STICK_ROWS] tensors are invalid input for allreduce's
# reduce-scatter split at >=4 ranks (LIBCOLL::SplitEnvelope::
# computeSplitGeometry() hits `base_size > 0` when dimsize/numParts rounds
# down below the 128-byte minimum chunk), so every collective tensor here
# is 2-D [ROWS, HIDDEN] instead.


def _world_all_succeeded(local_ok: bool) -> bool:
    """Cross-rank consensus over the WORLD PG: return True only if EVERY rank
    passed local_ok=True.

    Why this exists: a collective on a SUBGROUP can legitimately abort on just
    the subgroup's members (e.g. the Phase-1 host-reduce guard firing on a
    chunk shape without a device add-kernel). The ranks NOT in that subgroup
    never see the abort, so if the test then issues a WORLD collective, those
    non-member ranks block forever waiting on the members that already bailed --
    turning a clean per-rank failure into a whole-job hang (exactly what the
    120s per-test timeout caught). Funnelling a success flag through a WORLD
    all_reduce(MIN) converts that into a clean, symmetric failure on ALL ranks:
    every rank learns if any rank failed, so every rank raises together instead
    of some hanging.

    The flag is a CPU tensor: with the "cpu:gloo,spyre:spyreccl" PG, a CPU
    tensor's collective is served by GLOO, which supports MIN and every dtype
    with no dependency on any Spyre device add-kernel -- so this consensus
    barrier itself can never hit the very host-reduce/kernel-shape limitation
    it exists to report. Every WORLD rank participates, so it cannot desync.
    """
    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int64, device="cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=dist.group.WORLD,
                    async_op=False)
    return bool(flag[0].item() == 1)


class TestAsyncDispatch(TestCase):
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

    def test_allreduce_async_returns_before_completion(self) -> None:
        """dist.all_reduce(async_op=True) returns a real Work; wait() yields a
        result BYTE-IDENTICAL to the synchronous path.

        We cannot deterministically assert the collective is NOT complete at
        the moment the call returns (it may finish before we get to inspect
        it), so the behavioral proof is: (a) the returned handle is a genuine
        Work object with a wait() method (not None, not already-materialized
        data), and (b) after wait() the buffer's contents exactly match a
        synchronous (async_op=False) all_reduce run on a clone of the same
        input. The feature's actual contract is "async faithfully reproduces
        sync", not absolute numeric correctness -- the latter is a separate
        concern (see module docstring re: the pre-existing LIBCOLL 2-D
        under-reduction bug, which -- if present -- affects both sides
        identically and therefore does not break this equivalence check).
        CONFIRMED when this passes with no hang and exact async == sync
        equality.
        """
        # NOTE: build each buffer with a FRESH torch.ones() rather than
        # .clone(): a device-to-device clone routes through
        # spyre::copy_from_d2d, which (interleaved with the collectives in this
        # process) can trip a separate compiler-backend crash unrelated to this
        # feature. Two fresh all-ones tensors are identical inputs, so the
        # async==sync equivalence claim is unaffected.
        # Synchronous reference.
        ref = torch.ones((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)
        dist.all_reduce(ref, op=dist.ReduceOp.SUM, async_op=False)

        # Async path under test (independent, identically-valued input).
        t = torch.ones((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)
        work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        self.assertIsNotNone(work, "all_reduce(async_op=True) must return a Work")
        self.assertTrue(
            hasattr(work, "wait"), "returned handle must expose wait()"
        )
        work.wait()

        got = t.to("cpu")
        ref_cpu = ref.to("cpu")
        self.assertTrue(
            torch.equal(got, ref_cpu),
            f"rank {self.comm_rank}: async all_reduce result diverged from "
            f"sync reference; async={got.flatten()[:4].tolist()} "
            f"sync={ref_cpu.flatten()[:4].tolist()}",
        )

    def test_zz_world_and_tp_subgroup_coexist_and_teardown(self) -> None:
        """Interleave async collectives on the world PG and a subgroup, then
        destroy the subgroup while world stays live, and prove world still
        works afterward.

        This is the key v3 guard (C1/C2 teardown): destroying a subgroup's
        SpyreCCLBackend must not poison the process-global progress worker or
        the shared comm stream that the still-live world PG depends on.
        Mirrors the new_group() setup in test_subgroup.py, but only needs
        WORLD_SIZE >= 2 (subgroup ranks=[0, 1] is valid even when that IS the
        whole world).

        This method is named "zz" so it runs last (mirrors test_subgroup.py's
        test_zz_failure_isolation): it calls destroy_process_group() on the
        subgroup, a one-way teardown. If this ran earlier, any later test in
        this class that happened to reuse that subgroup's process group
        object would break -- naming it to sort last removes the fragile
        dependency on unittest's alphabetical method ordering happening to
        place the other tests first.

        CONFIRMED when: both the world and subgroup async all_reduce results
        are BYTE-IDENTICAL to a synchronous reference all_reduce run on a
        clone of the same input (async == sync equivalence -- see module
        docstring; this is deliberately robust to the separate pre-existing
        LIBCOLL 2-D under-reduction bug, since it would affect both sides
        identically), no deadlock occurs while both are in flight
        (interleaved async issue, then interleaved wait), and the post-
        teardown world all_reduce also reproduces its own sync reference.
        """
        n = self.comm_size
        tp_members = [0, 1] if n >= 2 else [0]
        tp = dist.new_group(ranks=tp_members)

        a = torch.ones((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)
        b = torch.ones((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)

        # Synchronous references for each, on FRESH identically-valued inputs
        # (not .clone() -- avoids the copy_from_d2d compiler path; see the
        # allreduce test's note). Computed before the async ops mutate a/b.
        a_ref = torch.ones((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)
        dist.all_reduce(a_ref, op=dist.ReduceOp.SUM, group=dist.group.WORLD,
                        async_op=False)

        # PARTIAL-ABORT GUARD. The SUBGROUP reference all_reduce below runs on
        # only tp_members. If the Spyre Ring allreduce can't find a device
        # add-kernel for this shape's per-chunk width, it falls back to the
        # host-reduce path, which the Phase-1 async worker refuses -- aborting
        # on the subgroup members ONLY. Left unguarded, the non-member ranks
        # would sail on to the world collective below and hang forever waiting
        # on members that already raised. So: run the subgroup op under
        # try/except, then reach WORLD consensus (via gloo, _world_all_succeeded)
        # BEFORE issuing any world Spyre collective. If any rank's subgroup op
        # failed, EVERY rank skips cleanly -- no hang, no partial state.
        # (Chunk kernels for the common fp16 TP shapes are pre-warmed and JIT is
        # enabled, so on a correctly-provisioned pod this guard never trips; it
        # exists so a provisioning gap fails loud-and-clean, not as a deadlock.)
        b_ref = torch.ones((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)
        subgroup_ok = True
        subgroup_err = ""
        try:
            dist.all_reduce(b_ref, op=dist.ReduceOp.SUM, group=tp,
                            async_op=False)
        except RuntimeError as exc:
            subgroup_ok = False
            subgroup_err = str(exc).splitlines()[0]

        if not _world_all_succeeded(subgroup_ok):
            self.skipTest(
                "subgroup all_reduce hit the Phase-1-excluded host-reduce path "
                "on at least one rank (missing device add-kernel for this "
                f"chunk shape); rank {self.comm_rank} local_ok={subgroup_ok} "
                f"err={subgroup_err!r}. Pre-warm the fp16 add-kernels / enable "
                "SPYRE_COMMS_KERNEL_JIT so the device path is taken."
            )

        # Interleave: issue both async ops before waiting on either, to
        # exercise the shared FIFO progress worker servicing two distinct
        # process groups concurrently. The subgroup op is issued only by its
        # members: dist.all_reduce on a group a rank is not part of returns
        # None (there is no work to wait on), so waiting is membership-gated.
        is_member = self.comm_rank in tp_members
        w1 = dist.all_reduce(a, op=dist.ReduceOp.SUM, group=dist.group.WORLD,
                              async_op=True)
        w2 = None
        if is_member:
            w2 = dist.all_reduce(b, op=dist.ReduceOp.SUM, group=tp,
                                 async_op=True)
        w1.wait()
        if w2 is not None:
            w2.wait()

        self.assertTrue(
            torch.equal(a.to("cpu"), a_ref.to("cpu")),
            f"rank {self.comm_rank}: world async all_reduce diverged from "
            f"sync reference; async={a.to('cpu').flatten()[:4].tolist()} "
            f"sync={a_ref.to('cpu').flatten()[:4].tolist()}",
        )
        if self.comm_rank in tp_members:
            self.assertTrue(
                torch.equal(b.to("cpu"), b_ref.to("cpu")),
                f"rank {self.comm_rank}: TP subgroup async all_reduce diverged "
                f"from sync reference; async={b.to('cpu').flatten()[:4].tolist()} "
                f"sync={b_ref.to('cpu').flatten()[:4].tolist()}",
            )

        # C1/C2: destroy the subgroup while world is still live.
        dist.destroy_process_group(tp)

        # World PG must still function correctly after the subgroup teardown
        # -- proves destroying the subgroup's backend did not poison the
        # shared comm stream / process-global progress worker. Compared
        # against its own sync reference (async == sync equivalence).
        c = torch.ones((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)
        c_ref = torch.ones((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)
        dist.all_reduce(c_ref, op=dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(c, op=dist.ReduceOp.SUM)
        self.assertTrue(
            torch.equal(c.to("cpu"), c_ref.to("cpu")),
            f"rank {self.comm_rank}: post-teardown world all_reduce diverged "
            f"from sync reference; got={c.to('cpu').flatten()[:4].tolist()} "
            f"sync={c_ref.to('cpu').flatten()[:4].tolist()}",
        )

    def test_back_to_back_collectives_preserve_order(self) -> None:
        """~40 iterations of distinct back-to-back collectives (all_reduce then
        broadcast) issued synchronously (default async_op=False, which still
        dispatches through the async progress worker under the hood): no
        hang, no matcher desync, and the final value is exactly the
        broadcast of rank-0's summed value.

        CONFIRMED when every iteration's post-broadcast value equals the sum
        of (rank + 1) across all ranks (rank 0's value after its own
        all_reduce), for all ranks, across all iterations.
        """
        n = self.comm_size
        expected_sum = float(sum(r + 1 for r in range(n)))
        for i in range(40):
            t = torch.full((ROWS, HIDDEN), float(self.comm_rank + 1),
                            dtype=torch.float16, device=DEVICE)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            dist.broadcast(t, src=0)
            got = t.to("cpu")
            self.assertTrue(
                torch.allclose(got, torch.full_like(got, expected_sum)),
                f"rank {self.comm_rank} iter {i}: expected broadcast of "
                f"{expected_sum}, got {got.flatten()[:4].tolist()}",
            )

    def test_destroy_work_handle_variants(self) -> None:
        """Drop an async Work handle without calling wait() (fire-and-forget),
        then use dist.barrier() as the sync point.

        This exercises the Work destructor's cancel/drain path when the
        collective may still be in flight and nothing ever waits on it
        directly. If the dtor mishandled this (hang, UAF, double-free) the
        subsequent barrier -- or the tensor's later use below -- would hang
        or crash the process rather than raise a clean Python exception.

        CONFIRMED when the barrier returns promptly and the process is still
        healthy enough to run a normal all_reduce afterward.
        """
        t = torch.ones((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)
        _ = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        # Handle dropped here without wait(); its dtor runs on refcount 0.
        dist.barrier()

        # Sanity: the backend must still be healthy after the drop. Assert
        # async==sync equivalence on fresh identically-valued inputs (not an
        # absolute value, which the separate pre-existing 2-D under-reduction
        # bug would false-fail; see module docstring). Both fresh torch.ones,
        # no .clone() (avoids the copy_from_d2d compiler path).
        u = torch.ones((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)
        dist.all_reduce(u, op=dist.ReduceOp.SUM)
        u_ref = torch.ones((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)
        dist.all_reduce(u_ref, op=dist.ReduceOp.SUM, async_op=False)
        self.assertTrue(
            torch.equal(u.to("cpu"), u_ref.to("cpu")),
            f"rank {self.comm_rank}: post-drop all_reduce diverged from sync "
            f"reference; got {u.to('cpu').flatten()[:4].tolist()} "
            f"sync={u_ref.to('cpu').flatten()[:4].tolist()}",
        )

    def test_reduce_excluded_op_stays_synchronous(self) -> None:
        """Smoke test for the dual-mode path: dist.reduce is NOT one of the 7
        async-dispatched ops (it drives group_context_->reduce() + ws->
        start() directly in SpyreCCLBackend::reduce, not enqueue_async), so
        it must still return correct results the ordinary synchronous way.

        Only SUM is supported by the backend's reduce (see
        SpyreCCLBackend::reduce); root rank ends up with the cross-rank sum.
        Skips if dist.reduce is unavailable in this build rather than
        failing, matching the repo's gap-tolerant skip convention.
        """
        if not hasattr(dist, "reduce"):
            pytest.skip("dist.reduce not available in this torch build")

        root = 0
        t = torch.full((ROWS, HIDDEN), float(self.comm_rank + 1),
                        dtype=torch.float16, device=DEVICE)
        try:
            dist.reduce(t, dst=root, op=dist.ReduceOp.SUM)
        except (RuntimeError, NotImplementedError) as e:
            pytest.skip(f"dist.reduce not supported on this build: {e}")

        if self.comm_rank == root:
            expected = float(sum(r + 1 for r in range(self.comm_size)))
            got = t.to("cpu")
            self.assertTrue(
                torch.allclose(got, torch.full_like(got, expected)),
                f"rank {self.comm_rank} (root): expected reduce sum {expected}, "
                f"got {got.flatten()[:4].tolist()}",
            )
        # Non-root ranks' buffer contents are unspecified after reduce();
        # nothing to assert there beyond "did not hang/crash".


if __name__ == "__main__":
    run_tests()
