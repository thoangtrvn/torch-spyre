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
"""EXECUTE the attention executors off-box (gauntlet R1 M1 — the
vacuous-green class fix).

The pre-fix off-box suite was green only because NOTHING executed either
executor: the first in-envelope container call crashed on
``torch.from_numpy(list)``. These tests run ``_run_container`` and
``_run_host_serial`` END-TO-END on CPU with a fake device launch that
SIMULATES each phase's kernel semantics (q_scale, IBR gather decoded
from the actual table values, matmuls, softmax), so every host-side
defect class — crashes, wrong table order (M2), wrong scatter, scale
dropped (M3), wrong partials — fails the MATH against the fp64 flash
reference, not just a type check.

The fake launch dispatches on the plan's own role vocabulary
(``id(binary) -> roles``), so the simulator conforms to the SAME
declared contract the real device does: tensor i binds segment i.
"""

from __future__ import annotations

import math

import pytest
import torch

ROWS = 8
HEAD_DIM = 64
EPS = 64


def _plan_fn():
    try:
        from sentient_codegen.encoder.attention_aten_dispatch import (
            attention_aten_plan,
        )
    except ImportError:
        pytest.skip("sentient_codegen not importable (source env.sh)")
    return attention_aten_plan


def _data(rows, kv, seed=20261123):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(1, 4, (rows, HEAD_DIM), generator=g, dtype=torch.float16)
    k = torch.randint(-1, 2, (kv, HEAD_DIM), generator=g, dtype=torch.float16)
    v = torch.randint(1, 16, (kv, HEAD_DIM), generator=g, dtype=torch.float16)
    return q, k, v


def _flash(q, k, v, scale):
    qf = q.float().numpy()
    kf = k.float().numpy()
    vf = v.float().numpy()
    s = qf @ kf.T * scale
    s -= s.max(axis=-1, keepdims=True)
    e = numpy_exp(s)
    p = e / e.sum(axis=-1, keepdims=True)
    return p @ vf


def numpy_exp(x):
    import numpy as np

    return np.exp(x)


class _FakeDevice:
    """Simulates every launch the executors make, keyed by the plan's
    own role vocabulary. Records the slots each ingress gather consumed
    (the M2 round-trip check)."""

    def __init__(self, plan, scale, rows, head_dim):
        self.roles_by_binary = {}
        if plan is not None:
            self.add_plan(plan)
        self.scale = scale
        self.rows = rows
        self.head_dim = head_dim
        self.consumed_slots = []  # (roles, [slot per entry])

    def add_plan(self, plan):
        for step in plan.steps:
            self.roles_by_binary.setdefault(id(step.binary), step.roles)

    def __call__(self, binary, tensors):
        roles = self.roles_by_binary[id(binary)]
        if roles == ("q", "q_scaled"):
            q = tensors[0].reshape(self.rows, self.head_dim)
            tensors[1].copy_((q.float() * self.scale).half().reshape(-1, EPS))
        elif roles == ("kv_src", "table", "gather_out"):
            entries = tensors[1].reshape(-1).tolist()
            spt = self.head_dim // EPS
            slots = [int(e) // spt for e in entries]
            self.consumed_slots.append((roles, slots))
            for t, slot in enumerate(slots):
                tensors[2][t * spt : (t + 1) * spt] = tensors[0][
                    slot * spt : (slot + 1) * spt
                ]
        elif roles == ("q_scaled", "b_operand", "scores"):
            a = tensors[0].reshape(self.rows, -1).float()
            tensors[2].copy_((a @ tensors[1].float()).half())
        elif roles == ("softmax_scratch", "scores", "probs"):
            tensors[2].copy_(torch.softmax(tensors[1].float(), -1).half())
        elif roles == ("probs", "b_operand", "pv_out"):
            tensors[2].copy_((tensors[0].float() @ tensors[1].float()).half())
        elif roles[0] == "kv_src":  # the container's 7-segment ABI
            self._container(tensors)
        else:
            raise AssertionError(f"unmodeled roles {roles}")

    def _container(self, tensors):
        kv_src, _p, q_or_out, tables, _ks, _v, _s = tensors
        spt = self.head_dim // EPS
        tab = tables.tolist()
        # kv_len derives from the TABLE (K gathers then V gathers, 32
        # rows each) — NEVER from kv_src's row count, which in the paged
        # form includes gaps
        kv_len = (len(tab) // 2) * 32
        k_rows = torch.stack(
            [
                kv_src[tab[r // 32][r % 32] : tab[r // 32][r % 32] + spt].reshape(-1)
                for r in range(kv_len)
            ]
        )
        v_rows = torch.stack(
            [
                kv_src[
                    tab[kv_len // 32 + r // 32][r % 32] : tab[kv_len // 32 + r // 32][
                        r % 32
                    ]
                    + spt
                ].reshape(-1)
                for r in range(kv_len)
            ]
        )
        q = q_or_out.reshape(self.rows, self.head_dim).float()
        s = q @ k_rows.float().T * self.scale
        p = torch.softmax(s, -1)
        out = (p @ v_rows.float()).half()
        q_or_out.copy_(out.reshape(-1, EPS))


def _run(plan, q, k, v, block_table, scale):
    import torch_spyre.ops.attention as att

    # CLONE the inputs: on the real device `.to(dev)` copies, but on
    # the CPU harness it is a no-op — the executor's output write would
    # alias the caller's q (a simulator-only artifact; the executors are
    # unchanged). Cloning keeps the caller's tensors pristine for the
    # reference computation.
    q, k, v = q.clone(), k.clone(), v.clone()
    fake = _FakeDevice(plan, scale, plan.rows, plan.head_dim)
    real_launch = att._launch
    att._launch = fake
    try:
        if plan.strategy.value == "container":
            out = att._run_container(plan, q, k, v, block_table, "cpu")
        else:
            out = att._run_host_serial(plan, q, k, v, block_table, "cpu")
    finally:
        att._launch = real_launch
    return out, fake


def _check(out, ref, band=0.25):
    import numpy as np

    got = out.float().numpy().astype("float64")
    md = float(np.abs(got - ref).max())
    assert md <= band, f"max_abs={md} > {band}"


# ---------------------------------------------------------------------------
# The CONTAINER executor, executed
# ---------------------------------------------------------------------------


def test_container_executes_contiguous():
    plan_fn = _plan_fn()
    plan = plan_fn(rows=ROWS, head_dim=HEAD_DIM, kv_len=64)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    q, k, v = _data(ROWS, 64)
    out, _fake = _run(plan, q, k, v, None, scale)
    _check(out, _flash(q, k, v, scale))


def test_container_executes_paged():
    """The paged container: gapped slots, K and V in DISJOINT regions
    (the a5p form), and the SENTINEL fill leaves unaddressed rows at
    7.0 — a wrong slot read shows up in the math."""
    plan_fn = _plan_fn()
    plan = plan_fn(rows=ROWS, head_dim=HEAD_DIM, kv_len=64)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    q, k, v = _data(ROWS, 64)

    def phys(r):
        page, off = divmod(r, 32)
        return page * 40 + off

    v_base = 40 * 2
    table = torch.tensor(
        [[phys(r) for r in range(g * 32, (g + 1) * 32)] for g in range(2)]
        + [[v_base + phys(r) for r in range(g * 32, (g + 1) * 32)] for g in range(2)],
        dtype=torch.int64,
    )
    out, _fake = _run(plan, q, k, v, table, scale)
    _check(out, _flash(q, k, v, scale))


# ---------------------------------------------------------------------------
# The HOST_SERIAL executor, executed — the M2 table-order round trip
# ---------------------------------------------------------------------------


def test_host_serial_executes_contiguous():
    plan_fn = _plan_fn()
    plan = plan_fn(rows=ROWS, head_dim=HEAD_DIM, kv_len=128)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    q, k, v = _data(ROWS, 128, seed=20261122)
    out, fake = _run(plan, q, k, v, None, scale)
    _check(out, _flash(q, k, v, scale))
    # the contiguous slot tables are the identity walk per block
    kinds = [roles[0] for roles, _ in fake.consumed_slots]
    assert kinds == ["kv_src"] * 8  # 2 blocks x (2 K + 2 V) gathers
    for i, (_roles, slots) in enumerate(fake.consumed_slots):
        b, within = divmod(i, 4)
        g = within % 2
        lo = b * 64 + g * 32
        assert slots == [lo + t for t in range(32)]


def test_host_serial_executes_paged_round_trip():
    """THE M2 ROUND TRIP: the scatter and the consumer must agree on the
    ONE declared order (the plan's ingress step order). A kind-major
    scatter under a step-order consumer feeds K slots to V gathers —
    the simulator decodes the ACTUAL table the executor built, gathers
    from the ACTUAL staged source, and the math fails if the orders
    disagree."""
    plan_fn = _plan_fn()
    plan = plan_fn(rows=ROWS, head_dim=HEAD_DIM, kv_len=128)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    q, k, v = _data(ROWS, 128, seed=20261122)

    def phys(r):
        page, off = divmod(r, 32)
        return page * 40 + off

    # step order: per block, K gathers then V gathers (disjoint regions)
    tbl = []
    for b in range(2):
        for kind in ("k", "v"):
            for g in range(2):
                base = 0 if kind == "k" else 40 * 4
                lo = b * 64 + g * 32
                tbl.append([base + phys(r) for r in range(lo, lo + 32)])
    table = torch.tensor(tbl, dtype=torch.int64)
    out, fake = _run(plan, q, k, v, table, scale)
    _check(out, _flash(q, k, v, scale))
    # the consumed slots ARE the caller's table rows, in step order
    for i, (_roles, slots) in enumerate(fake.consumed_slots):
        assert slots == tbl[i]


# ---------------------------------------------------------------------------
# Scale threads through EXECUTION (M3)
# ---------------------------------------------------------------------------


def test_scale_threads_through_execution():
    plan_fn = _plan_fn()
    scale = 0.5
    plan = plan_fn(rows=ROWS, head_dim=HEAD_DIM, kv_len=64, scale=scale)
    q, k, v = _data(ROWS, 64)
    out, _fake = _run(plan, q, k, v, None, scale)
    _check(out, _flash(q, k, v, scale))
    # and the plan itself carries the scale (not the default)
    assert plan.scale == 0.5


# ---------------------------------------------------------------------------
# The registered decomposition executes its refusal path (M6, off-box
# half; the silicon e1/e2 runs are the executed-admitted-half receipts)
# ---------------------------------------------------------------------------


def test_registered_decomposition_refuses_clause_named():
    from torch_spyre._inductor.decompositions import spyre__sdpa_overrideable

    q, k, v = _data(4, 64)
    q4 = q.reshape(1, 1, 4, HEAD_DIM)
    k4 = k.reshape(1, 1, 64, HEAD_DIM)
    v4 = v.reshape(1, 1, 64, HEAD_DIM)
    with pytest.raises(NotImplementedError) as ei:
        spyre__sdpa_overrideable(q4.float(), k4.float(), v4.float())
    assert "[dtype]" in str(ei.value)
    with pytest.raises(NotImplementedError) as ei:
        spyre__sdpa_overrideable(q4, k4, v4, is_causal=True)
    assert "[mask-mode]" in str(ei.value)  # rows=4 > 1 causal gate (M4)


# ---------------------------------------------------------------------------
# The OP-LEVEL scale path (R2 M3): the scale must survive the op body's
# own plan builds — the first fix round threaded it only into the
# direct attention_aten_plan calls, so F.sdpa(scale=s) silently computed
# at the default.
# ---------------------------------------------------------------------------


def test_op_level_scale_through_the_registered_route():
    """The full ATen route with an explicit non-default scale: the
    mapping admits it, and the OP BODY's own plan build carries it (the
    executed math reflects the scale — asserted through the batched
    entry with the phase simulator, the off-box stand-in for the device
    run)."""
    import torch_spyre.ops.attention as att
    from torch_spyre.ops.attention import sdpa_attention_mapping

    q, k, v = _data(ROWS, 64)
    q4 = q.reshape(1, 1, ROWS, HEAD_DIM)
    k4 = k.reshape(1, 1, 64, HEAD_DIM)
    v4 = v.reshape(1, 1, 64, HEAD_DIM)
    scale = 0.5

    # (a) the mapping's plan carries the explicit scale (the admitted
    # route — F.sdpa -> overrideable -> mapping)
    calls = sdpa_attention_mapping(q4, k4, v4, None, 0.0, False, False, scale)
    assert calls[0]["scale"] == 0.5

    # (b) the OP BODY's own plan build carries it and the EXECUTED math
    # reflects it: run _attention_batched with the fake launch and the
    # fp64 reference AT the explicit scale. A default-scale plan (the
    # R2 defect) computes softmax(q@k.T/8) and fails this band.
    plans = []

    real_build = att._build_plan_offline

    def spy_build(*args):
        plan = real_build(*args)
        plans.append(plan)
        fake.add_plan(plan)  # the op may build a distinct (cached) plan
        return plan

    # the plan the op will build is the lru-cached one the mapping
    # already built — same key, same object; the fake dispatches on its
    # step binaries
    fake = _FakeDevice(calls[0]["plan"], scale, ROWS, HEAD_DIM)
    real_launch = att._launch
    att._build_plan_offline = spy_build
    att._launch = fake
    try:
        # clones: the CPU harness's .to("cpu") no-op would alias the
        # executor's output write onto the caller's tensors (the same
        # simulator-only artifact _run guards against)
        out = att._attention_batched(
            q4.clone(), k4.clone(), v4.clone(), None, "cpu", scale
        )
    finally:
        att._build_plan_offline = real_build
        att._launch = real_launch
    assert plans and plans[0].scale == 0.5, (
        f"the op body's plan build dropped the explicit scale "
        f"(plan.scale={plans[0].scale if plans else 'no plan'})"
    )
    # the fake needs the plan's binaries for role dispatch
    _check(out, _flash(q, k, v, scale))
