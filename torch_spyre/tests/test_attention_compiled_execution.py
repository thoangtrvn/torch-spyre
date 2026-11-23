# Copyright 2025 The Torch-Spyre Authors.
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
2. The compiled invocation must actually go through dynamo AT THE USER
   LEVEL — correct values alone would pass vacuously on a silent eager
   fallback. Two independent pins (mutation-proven, see the receipts):
   ``torch.compiler.is_compiling()`` recorded True at the USER function's
   entry (an eager fallback records False — the seam's own OPWrapper
   inner compile compiles the DECOMPOSITION, never the user function),
   and the captured aot graph contains ``spyre.attention`` (the math
   form cannot produce it). The dynamo frame counter alone is NOT a pin:
   any inner torch.compile increments it identically.

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

import torch_spyre  # noqa: E402

# Post-condition on the meta_path cleanup above: the torch_spyre this
# test executes must be THIS tree's (an editable install of another
# checkout would otherwise silently run a different decomposition table).
assert torch_spyre.__file__.startswith(_THIS_REPO), (
    f"torch_spyre resolves outside this tree: {torch_spyre.__file__} "
    f"(repo root {_THIS_REPO}) — the meta_path cleanup above failed"
)

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
        self.executed_roles = []

    def add_plan(self, plan):
        for step in getattr(plan, "steps", []):
            self.roles_by_binary.setdefault(id(step.binary), step.roles)

    def __call__(self, binary, tensors):
        roles = self.roles_by_binary[id(binary)]
        self.executed_roles.append(roles)
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
    """The three codegen-only stand-ins (see the module docstring).

    ``register_dispatch`` installs the PrivateUse1 kernels the real
    runtime's ``_lazy_init`` performs (codegen-only mode skips
    ``_lazy_init``). It is load-bearing for BOTH paths: dynamo inlines
    the Python dispatch kernel at trace time (the compiled route), and
    eager dispatch reaches the same kernels directly."""

    def __init__(self, register_dispatch=True):
        self._register_dispatch = register_dispatch
        self.plans = []

    def __enter__(self):
        if self._register_dispatch:
            from torch_spyre._inductor.decompositions import (
                _register_spyre_dispatchkey_kernels_permanently,
            )

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
            self.plans.append(plan)
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


def _recording_backend(graphs):
    """A torch.compile backend that records the post-decomposition (aot
    output) graph, then delegates to the PRODUCTION compile path —
    torch_spyre wraps ``torch._inductor.compile_fx.compile_fx`` (see
    ``torch_spyre/_inductor/__init__.py::enable_spyre_compile_fx_wrapper``),
    so this backend runs the exact same lowering the default
    ``backend='inductor'`` call would. In-process and deterministic (the
    TORCH_LOGS=aot_graphs route was probed and REJECTED: the artifact
    records reach stderr from a compile subprocess and a logging handler
    on the artifact logger captures nothing — see the receipts)."""

    def backend(gm, example_inputs):
        graphs.append(gm)
        import torch._inductor.compile_fx as cfx

        return cfx.compile_fx(gm, example_inputs)

    return backend


def _run_compiled(q, k, v):
    """torch.compile the user-level F.sdpa function and EXECUTE it.

    Returns (out, user_is_compiling, graph_text, breaks, launches,
    expected_launches, executed_roles). ``user_is_compiling`` is the
    record of ``torch.compiler.is_compiling()`` at the USER function's
    entry — the anti-fallback pin (False on a silent eager fallback,
    including one that routes through the seam's own OPWrapper inner
    compile, which compiles the decomposition, never this function)."""
    _plan_fn()  # skip if codegen is unreachable
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()
    user_is_compiling = []
    graphs = []

    def user_fn(q_, k_, v_):
        user_is_compiling.append(torch.compiler.is_compiling())
        return F.scaled_dot_product_attention(q_, k_, v_)

    with _CompiledHarness() as h:
        q4 = q.reshape(1, 1, ROWS, HEAD_DIM).to(DEVICE)
        k4 = k.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)
        v4 = v.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)
        compiled = torch.compile(
            user_fn, backend=_recording_backend(graphs), dynamic=False
        )
        out = compiled(q4, k4, v4).cpu().reshape(ROWS, HEAD_DIM)
        breaks = dict(counters["graph_breaks"])
        launches = h.simulator.launches
        # derived from the plan itself, not a hand-counted constant
        expected_launches = sum(len(getattr(p, "steps", [])) for p in h.plans)
    graph_text = "\n".join(str(gm.graph) for gm in graphs)
    executed_roles = [tuple(r) for r in h.simulator.executed_roles]
    return (
        out,
        user_is_compiling,
        graph_text,
        breaks,
        launches,
        expected_launches,
        executed_roles,
    )


def test_compiled_sdpa_executes_through_dynamo():
    """The M6 cell: values correct AND the dynamo path actually taken.

    The is_compiling record is the anti-fallback pin: a silent eager
    fallback produces correct values with is_compiling False at the user
    entry and must FAIL (mutation-proven — see the receipts)."""
    q, k, v = _data()
    ref = _flash(q, k, v)
    out, user_is_compiling, _g, _b, launches, _e, _r = _run_compiled(q, k, v)
    import numpy as np

    md = float(np.abs(out.float().numpy().astype("float64") - ref).max())
    assert md <= OUT_MAX_ABS, f"max_abs={md} > {OUT_MAX_ABS}"
    assert user_is_compiling and all(user_is_compiling), (
        f"the user function never ran inside torch.compile "
        f"(is_compiling record: {user_is_compiling}) — correct values alone "
        f"do not distinguish a compiled run from a silent eager fallback"
    )
    assert launches > 0, "no device launch executed"


def test_compiled_graph_routes_through_the_attention_program():
    """The top-level SDPA op must reach spyre.attention, not the math
    form. Two independent facts, both in-process:

    - the DYNAMO graph (captured by a recording backend that delegates
      to the production compile_fx) contains the top-level
      ``scaled_dot_product_attention`` call — the user function was
      traced, not skipped;
    - the EXECUTED launch roles are the attention program's own phase
      vocabulary (q_scale / gather / scores / softmax / pv, from the
      codegen plan the attention route builds) and the launch count
      equals the plan's own step count (derived, not hand-counted).
      The math form cannot produce these — it never builds an attention
      plan and is refused typed (F32PointerElementGrainError) before any
      launch.

    Plus zero graph breaks."""
    q, k, v = _data()
    _out, _u, graph_text, breaks, launches, expected, roles = _run_compiled(q, k, v)
    assert "scaled_dot_product_attention" in graph_text, (
        "the traced graph does not contain the top-level SDPA op — the "
        "user function was not traced"
    )
    assert ("q", "q_scaled") in roles, (
        f"the attention plan's own role vocabulary never executed (roles: "
        f"{sorted(set(roles))})"
    )
    assert ("softmax_scratch", "scores", "probs") in roles, (
        f"the softmax phase never executed (roles: {sorted(set(roles))})"
    )
    assert launches == expected, (
        f"launches {launches} != the plan's own step count {expected}"
    )
    assert launches > 1, (
        f"expected the kv=128 HOST_SERIAL multi-launch sequence, got {launches}"
    )
    assert not breaks, f"unexpected graph breaks: {breaks}"


def test_compiled_matches_the_eager_seam():
    """Compiled output matches the eager overrideable-seam output (the
    form the eager silicon e-cell reaches through C++ backend selection)
    within the DL16 store band — the same computation executed through
    two dispatch paths, not a bit-identity claim."""
    q, k, v = _data()
    out_c, *_ = _run_compiled(q, k, v)
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
    assert delta <= 1e-3, f"compiled != eager seam (max delta {delta})"


def test_out_of_envelope_refuses_typed_under_compile():
    """rows=33 (> the v1 row ceiling): the predicate's own [row-budget]
    clause surfaces at compile time — typed refusal (the codegen
    predicate's UnsupportedAttentionError, wrapped by the backend
    compiler), never a CPU fallback and never an untyped crash."""
    from sentient_codegen.scheduler.attention_predicate import (
        UnsupportedAttentionError,
    )

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
    typed = [
        e for e in _exc_chain(ei.value) if isinstance(e, UnsupportedAttentionError)
    ]
    assert typed, (
        f"the refusal must be the predicate's own typed "
        f"UnsupportedAttentionError (chain: "
        f"{[type(e).__name__ for e in _exc_chain(ei.value)]})"
    )
    assert "[row-budget]" in str(typed[0]), (
        f"the refusal must carry the predicate's own clause name, got: "
        f"{str(typed[0])[:200]}"
    )


def test_eager_sdpa_reroutes_through_the_seam():
    """The EAGER reroute, disclosed and pinned: registering the top-level
    decomposition's PrivateUse1 kernel (load-bearing for dynamo inlining)
    also intercepts eager dispatch — eager ``F.scaled_dot_product_attention``
    must run the seam (overrideable -> spyre.attention), NOT the C++
    composite. Proof: the top-level OPWrapper records the dispatch, and
    the C++ composite path CANNOT succeed in this env anyway (its
    fp16->fp32 device-side ``_to_copy`` prelude is refused by the
    codegen-only ``copy_``) — correct values through the simulator are
    therefore seam values, not composite values.

    Silicon note (the disclosure): this route BYPASSES the C++ composite's
    fp16->fp32 prelude, so it is not the pre-change e-cell path — the box
    cell must re-witness eager F.sdpa before the eager claim is reused."""
    from torch_spyre._inductor.decompositions import (
        spyre_decompositions_via_dispatchkey,
    )

    top = "aten::scaled_dot_product_attention"
    # the kernel exists and the wrapper is installed (spy-proven reach:
    # swap the wrapper's class for an instrumented subclass and record)
    assert torch._C._dispatch_has_kernel_for_dispatch_key(top, "PrivateUse1"), (
        f"{top} has no PrivateUse1 kernel — the compiled route is dead too"
    )
    wrapper = [
        w for op, w in spyre_decompositions_via_dispatchkey.items() if op._name == top
    ][0]
    calls = []

    class _Instr(type(wrapper)):
        def __call__(self, *args, **kwargs):
            calls.append(torch.compiler.is_compiling())
            return super().__call__(*args, **kwargs)

    q, k, v = _data()
    ref = _flash(q, k, v)
    import numpy as np

    orig_cls = wrapper.__class__
    with _CompiledHarness() as h:
        q4 = q.reshape(1, 1, ROWS, HEAD_DIM).to(DEVICE)
        k4 = k.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)
        v4 = v.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)
        wrapper.__class__ = _Instr
        try:
            out = F.scaled_dot_product_attention(q4, k4, v4)
        finally:
            wrapper.__class__ = orig_cls
        out = out.cpu().reshape(ROWS, HEAD_DIM)
        launches = h.simulator.launches
    assert calls and not any(calls), (
        f"the top-level PrivateUse1 wrapper was not reached in EAGER "
        f"(is_compiling record: {calls}) — eager dispatch did not reroute "
        f"through the decomposition"
    )
    md = float(np.abs(out.float().numpy().astype("float64") - ref).max())
    assert md <= OUT_MAX_ABS, f"eager reroute max_abs={md} > {OUT_MAX_ABS}"
    assert launches > 0, "no device launch executed in the eager reroute"


def _exc_chain(exc):
    while exc is not None:
        yield exc
        exc = exc.__cause__ or exc.__context__


def test_negative_dropout_p_refuses_typed():
    """ATen's own contract: dropout_p outside [0, 1) is an error. The C++
    composite validated it before dispatch; the compiled path bypasses
    the composite, so the mapping owns the range now (typed
    NotImplementedError, not an untyped crash)."""
    q, k, v = _data()
    torch._dynamo.reset()
    with _CompiledHarness():
        q4 = q.reshape(1, 1, ROWS, HEAD_DIM).to(DEVICE)
        k4 = k.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)
        v4 = v.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)
        compiled = torch.compile(
            lambda q_, k_, v_: F.scaled_dot_product_attention(
                q_, k_, v_, dropout_p=-0.5
            ),
            dynamic=False,
        )
        with pytest.raises(Exception) as ei:
            compiled(q4, k4, v4)
    typed = [e for e in _exc_chain(ei.value) if isinstance(e, NotImplementedError)]
    assert typed and "dropout_p" in str(typed[0]), (
        f"expected the typed dropout-range NotImplementedError in the chain, "
        f"got {[type(e).__name__ for e in _exc_chain(ei.value)]}"
    )
