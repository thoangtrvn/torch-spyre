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
"""EXECUTE the compiled SDPA entry off-box (the M6 cell).

torch.compile through ``spyre__sdpa_overrideable`` — the entry the eager
silicon e-cells never exercised. Two load-bearing pins:

1. The top-level ``aten::scaled_dot_product_attention`` must route to the
   overrideable seam under dynamo (the ``spyre_scaled_dot_product_attention``
   decomposition). Without it, aot traces the C++ composite's MATH form
   (fp32 upcast, bmm, softmax) and the Sentient codegen refuses typed
   (f32 pointee, no 32-bit-grain route) — the red this test was written
   against.
2. The compiled invocation must actually go through dynamo — correct
   values alone would pass vacuously on a silent eager fallback. The
   dynamo frame counter must show a compiled frame.

The device launch is the phase simulator (the
``test_attention_executor_execution`` pattern): it dispatches on the
plan's own role vocabulary and computes each phase's semantics on host
copies moved through the codegen-only shadow channel. The three
codegen-only stand-ins (launch simulator, ``require_runtime`` no-op,
``Tensor.to`` through ``copy_``) are the same ones the off-box ladder
receipt documents; on the box none of them exist.
"""

from __future__ import annotations

import math
import os
import sys

# Dev-env hygiene: a pip editable install of ANOTHER torch-spyre checkout
# installs a meta_path finder that shadows this tree's torch_spyre at
# `import torch` (the torch.backends autoload) — its compile_fx wrapper
# would then register a DIFFERENT tree's decomposition table and this
# test would run against the wrong code. Remove any such finder that
# maps torch_spyre outside this file's repository root, BEFORE the first
# torch import. No-op in a normal install.
_THIS_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
for _hook in list(sys.meta_path):
    _finder_mod = sys.modules.get(getattr(_hook, "__module__", ""))
    _mapping = getattr(_finder_mod, "MAPPING", None)
    if _mapping and not str(_mapping.get("torch_spyre", _THIS_REPO)).startswith(
        _THIS_REPO
    ):
        sys.meta_path.remove(_hook)

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROWS = 8
HEAD_DIM = 64
KV_LEN = 128
OUT_MAX_ABS = 0.25  # the e-cell band (e2 silicon observed 0.0294)
DEVICE = torch.device("spyre")


def _plan_fn():
    try:
        from sentient_codegen.encoder.attention_aten_dispatch import (
            attention_aten_plan,
        )
    except ImportError:
        pytest.skip("sentient_codegen not importable (source env.sh)")
    return attention_aten_plan


def _data(rows=ROWS, kv=KV_LEN, seed=20261122):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(1, 4, (rows, HEAD_DIM), generator=g, dtype=torch.float16)
    k = torch.randint(-1, 2, (kv, HEAD_DIM), generator=g, dtype=torch.float16)
    v = torch.randint(1, 16, (kv, HEAD_DIM), generator=g, dtype=torch.float16)
    return q, k, v


def _flash(q, k, v):
    import numpy as np

    qf = q.float().numpy().astype("float64")
    kf = k.float().numpy().astype("float64")
    vf = v.float().numpy().astype("float64")
    s = qf @ kf.T / math.sqrt(HEAD_DIM)
    s -= s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    p = e / e.sum(axis=-1, keepdims=True)
    return p @ vf


class _PhaseSimulator:
    """Simulates every launch the executors make, keyed by the plan's own
    role vocabulary. Channel-less spyre tensors are the executors'
    write-only device buffers (zeros / -777 scratch) — stand in a zero
    host buffer and write results back through copy_."""

    def __init__(self):
        self.roles_by_binary = {}
        self.launches = 0

    def add_plan(self, plan):
        for step in getattr(plan, "steps", []):
            self.roles_by_binary.setdefault(id(step.binary), step.roles)

    def __call__(self, binary, tensors):
        roles = self.roles_by_binary[id(binary)]
        self.launches += 1
        host = [self._host(t) for t in tensors]
        self._run(roles, host)
        for t, h in zip(tensors, host):
            if isinstance(t, torch.Tensor) and t.device.type == "spyre":
                t.copy_(h)

    @staticmethod
    def _host(t):
        if not isinstance(t, torch.Tensor) or t.device.type != "spyre":
            return t
        try:
            return t.cpu()
        except RuntimeError:
            return torch.zeros(tuple(t.shape), dtype=t.dtype)

    def _run(self, r, ts):
        eps = 64
        rows, hd = ROWS, HEAD_DIM
        scale = 1.0 / math.sqrt(hd)
        if r == ("q", "q_scaled"):
            q = ts[0].reshape(rows, hd)
            ts[1].copy_((q.float() * scale).half().reshape(-1, eps))
        elif r == ("kv_src", "table", "gather_out"):
            entries = ts[1].reshape(-1).tolist()
            spt = hd // eps
            slots = [int(e) // spt for e in entries]
            for t, slot in enumerate(slots):
                ts[2][t * spt : (t + 1) * spt] = ts[0][slot * spt : (slot + 1) * spt]
        elif r == ("q_scaled", "b_operand", "scores"):
            a = ts[0].reshape(rows, -1).float()
            ts[2].copy_((a @ ts[1].float()).half())
        elif r == ("softmax_scratch", "scores", "probs"):
            ts[2].copy_(torch.softmax(ts[1].float(), -1).half())
        elif r == ("probs", "b_operand", "pv_out"):
            ts[2].copy_((ts[0].float() @ ts[1].float()).half())
        else:
            raise AssertionError(f"unmodeled roles {r}")


class _CompiledHarness:
    """The three codegen-only stand-ins (see the module docstring)."""

    def __enter__(self):
        from torch_spyre._inductor.decompositions import (
            _register_spyre_dispatchkey_kernels_permanently,
        )

        # the PrivateUse1 kernel registration the real runtime's
        # _lazy_init performs (codegen-only mode skips _lazy_init)
        _register_spyre_dispatchkey_kernels_permanently()
        import torch_spyre.ops.attention as att

        self._att = att
        self.simulator = _PhaseSimulator()
        self._real_launch = att._launch
        self._real_require = att.require_runtime
        self._real_build = att._build_plan_offline
        att._launch = self.simulator
        att.require_runtime = lambda name: None

        def build(*args):
            plan = self._real_build(*args)
            self.simulator.add_plan(plan)
            return plan

        att._build_plan_offline = build

        # cpu -> spyre through copy_ so the shadow channel carries values
        # (the fake backend's _to_copy is channel-less; the real backend's
        # .to is a real H2D)
        self._orig_to = torch.Tensor.to
        orig_to = self._orig_to  # the tensor arg below shadows `self`

        def to(self, *args, **kwargs):
            device = kwargs.get("device")
            if device is None and args and isinstance(args[0], (torch.device, str)):
                device = args[0]
            if isinstance(device, str):
                device = torch.device(device)
            if (
                isinstance(device, torch.device)
                and device.type == "spyre"
                and self.device.type != "spyre"
                and "dtype" not in kwargs
                and not (args and isinstance(args[0], torch.dtype))
            ):
                out = torch.empty_like(self, device=device)
                out.copy_(self)
                return out
            return orig_to(self, *args, **kwargs)

        torch.Tensor.to = to
        return self

    def __exit__(self, *exc):
        att = self._att
        att._launch = self._real_launch
        att.require_runtime = self._real_require
        att._build_plan_offline = self._real_build
        torch.Tensor.to = self._orig_to
        return False


def _run_compiled(q, k, v):
    """torch.compile the user-level F.sdpa function and EXECUTE it."""
    _plan_fn()  # skip if codegen is unreachable
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()
    with _CompiledHarness() as h:
        q4 = q.reshape(1, 1, ROWS, HEAD_DIM).to(DEVICE)
        k4 = k.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)
        v4 = v.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)
        compiled = torch.compile(
            lambda q_, k_, v_: F.scaled_dot_product_attention(q_, k_, v_),
            dynamic=False,
        )
        out = compiled(q4, k4, v4).cpu().reshape(ROWS, HEAD_DIM)
        frames = dict(counters["frames"])
        breaks = dict(counters["graph_breaks"])
        launches = h.simulator.launches
    return out, frames, breaks, launches


def test_compiled_sdpa_executes_through_dynamo():
    """The M6 cell: values correct AND the dynamo path actually taken.

    The frame counter is the anti-fallback pin: a silent eager fallback
    produces correct values with zero compiled frames and must FAIL."""
    q, k, v = _data()
    ref = _flash(q, k, v)
    out, frames, breaks, launches = _run_compiled(q, k, v)
    import numpy as np

    md = float(np.abs(out.float().numpy().astype("float64") - ref).max())
    assert md <= OUT_MAX_ABS, f"max_abs={md} > {OUT_MAX_ABS}"
    assert frames.get("ok", 0) > 0, (
        f"the compiled invocation never went through dynamo (frames={frames}) "
        f"— correct values alone do not distinguish a compiled run from a "
        f"silent eager fallback"
    )
    assert launches > 0, "no device launch executed"


def test_compiled_graph_routes_through_the_attention_program():
    """The top-level SDPA op must reach spyre::attention, not the math
    form: the HOST_SERIAL phase sequence (15 launches: q_scale + 2 blocks
    x (k/v ingress + scores + softmax + pv)) executed with the plan's own
    role vocabulary. The math form cannot get here (it never launches)."""
    q, k, v = _data()
    _out, _frames, breaks, launches = _run_compiled(q, k, v)
    assert launches == 15, (
        f"expected the kv=128 HOST_SERIAL sequence (15 launches), got {launches}"
    )
    assert not breaks, f"unexpected graph breaks: {breaks}"


def test_compiled_matches_the_eager_seam():
    """Compiled output == the eager overrideable-seam output (the form the
    eager silicon e-cell reaches through C++ backend selection)."""
    q, k, v = _data()
    out_c, _f, _b, _l = _run_compiled(q, k, v)
    with _CompiledHarness():
        q4 = q.reshape(1, 1, ROWS, HEAD_DIM).to(DEVICE)
        k4 = k.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)
        v4 = v.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)
        out_e, *_ = (
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default(  # noqa: E501
                q4, k4, v4
            )
        )
        out_e = out_e.cpu().reshape(ROWS, HEAD_DIM)
    delta = float((out_c.float() - out_e.float()).abs().max())
    assert delta == 0.0, f"compiled != eager seam (max delta {delta})"


def test_out_of_envelope_refuses_typed_under_compile():
    """rows=33 (> the v1 row ceiling): the predicate's own [row-budget]
    clause surfaces at compile time — typed refusal, never a CPU
    fallback."""
    q, k, v = _data(rows=33, kv=64, seed=20261123)
    torch._dynamo.reset()
    with _CompiledHarness():
        q4 = q.reshape(1, 1, 33, HEAD_DIM).to(DEVICE)
        k4 = k.reshape(1, 1, 64, HEAD_DIM).to(DEVICE)
        v4 = v.reshape(1, 1, 64, HEAD_DIM).to(DEVICE)
        compiled = torch.compile(
            lambda q_, k_, v_: F.scaled_dot_product_attention(q_, k_, v_),
            dynamic=False,
        )
        with pytest.raises(Exception) as ei:
            compiled(q4, k4, v4)
    assert "[row-budget]" in str(ei.value), (
        f"the refusal must carry the predicate's own clause name, got: "
        f"{str(ei.value)[:200]}"
    )
