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

"""PrivateUse1 fold-surface tests (codegen-only).

Derived from the actual consumer: Inductor's
``torch/_inductor/fx_passes/joint_graph.py::constant_fold_uniform_value``
eagerly evaluates device-side scalars (transformers' ``eager_mask`` pattern:
``torch.where(mask, scalar_tensor(0.0), scalar_tensor(min_dtype))``) via
``aten.scalar_tensor``/``aten.full`` (-> ``empty`` + ``fill_.Scalar``) and
``tensor.flatten()[0].item()`` (-> ``view``/``as_strided``/
``_local_scalar_dense``).  The codegen-only backend implements exactly that
surface with a host-side shadow-value channel, and refuses loudly outside it.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_WORKTREE_ROOT = Path(__file__).resolve().parents[1]

# Import-resolution honesty: the tests must exercise the worktree build.
_HEADER = (
    "import torch\n"
    "import torch_spyre\n"
    f"assert torch_spyre.__file__.startswith({str(_WORKTREE_ROOT)!r}), "
    "torch_spyre.__file__\n"
    "assert torch_spyre.codegen_only_available() is True\n"
)


def _run_fold_script(body: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    env["PYTHONPATH"] = (
        str(_WORKTREE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", _HEADER + textwrap.dedent(body)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_eager_eval_chain_flatten_item_roundtrip():
    """The add_node_replacement chain: flatten()[0].item() on a spyre scalar.

    The scalar is built the way the fold finds it: empty (registered) + copy_
    from a CPU value (the path that used to be a silent no-op).  The exact
    round-trip must hold for 0.0 and for float32's finfo.min (the transformers
    min_dtype value).
    """
    result = _run_fold_script(
        """
        minv = torch.finfo(torch.float32).min
        for v in (0.0, minv, 1.5):
            t = torch.empty([1], device="spyre:0")
            t.copy_(torch.tensor([v]))
            got = t.flatten()[0].item()
            assert got == v, (v, got)
            # select path: [0] -> as_strided
            t2 = torch.empty([2, 2], device="spyre:0")
            t2.copy_(torch.full((2, 2), v))
            got2 = t2.flatten()[0].item()
            assert got2 == v, (v, got2)
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr


def test_constant_fold_uniform_value_seam():
    """The true seam: constant_fold_uniform_value on the where/scalar_tensor
    pattern (transformers eager_mask) must complete and fold the scalars."""
    result = _run_fold_script(
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._inductor.fx_passes.joint_graph import constant_fold_uniform_value

        minv = torch.finfo(torch.float32).min
        spyre = torch.device("spyre:0")
        g = torch.fx.Graph()
        with FakeTensorMode():
            mask = g.placeholder("mask")
            mask.meta["val"] = torch.ones(2, 2, dtype=torch.bool, device=spyre)
            z = g.call_function(
                torch.ops.aten.scalar_tensor.default, (0.0,),
                {"dtype": torch.float32, "device": spyre})
            z.meta["val"] = torch.ops.aten.scalar_tensor(
                0.0, dtype=torch.float32, device=spyre)
            m = g.call_function(
                torch.ops.aten.scalar_tensor.default, (minv,),
                {"dtype": torch.float32, "device": spyre})
            m.meta["val"] = torch.ops.aten.scalar_tensor(
                minv, dtype=torch.float32, device=spyre)
            w = g.call_function(torch.ops.aten.where.default, (mask, z, m))
            w.meta["val"] = torch.where(
                mask.meta["val"], z.meta["val"], m.meta["val"])
        g.output(w)
        gm = torch.fx.GraphModule({}, g)
        gm.recompile()

        # RED at base: NotImplementedError aten::fill_.Scalar at PrivateUse1.
        constant_fold_uniform_value(gm)
        gm.recompile()

        # The foldable scalar producers must be replaced by full constructors.
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert torch.ops.aten.scalar_tensor.default not in targets, targets
        assert torch.ops.aten.where.default in targets, targets
        assert any(
            t == torch.ops.aten.full.default for t in targets
        ), targets
        # The folded values must round-trip exactly.
        full_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.full.default
        ]
        assert len(full_nodes) == 2, [n.args for n in full_nodes]
        values = sorted(n.args[1] for n in full_nodes)
        assert values == [minv, 0.0], values
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr


def test_copy_extraction_and_clone_roundtrip():
    """cpu.copy_(spyre) must extract real values (or fail loudly), never
    silently no-op; clone must copy real values."""
    result = _run_fold_script(
        """
        t = torch.empty([2, 2], device="spyre:0")
        t.copy_(torch.arange(4, dtype=torch.float32).reshape(2, 2))

        out = torch.empty(2, 2)
        out.copy_(t)
        assert torch.equal(out, torch.arange(4, dtype=torch.float32).reshape(2, 2)), out

        c = t.clone()
        assert c.flatten()[0].item() == 0.0
        assert c[1, 1].item() == 3.0, c[1, 1].item()
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr


def test_refusal_outside_derived_set():
    """Loud refusals must survive the surface extension:
    - unregistered pointwise compute (neg) still NotImplementedError;
    - extracting from a spyre tensor with no shadow channel fails loudly.
    """
    result = _run_fold_script(
        """
        t = torch.empty([2, 2], device="spyre:0")

        try:
            torch.neg(t)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("neg must stay unregistered (loud refusal)")

        # No fill_/copy_ ever touched t: no host-side data channel exists, so
        # extraction must refuse loudly rather than silently return garbage.
        out = torch.empty(2, 2)
        try:
            out.copy_(t)
        except RuntimeError as e:
            assert "spyre" in str(e).lower(), e
        else:
            raise AssertionError("copy_ from a channel-less spyre tensor must fail loudly")

        try:
            t.flatten()[0].item()
        except RuntimeError as e:
            assert "spyre" in str(e).lower(), e
        else:
            raise AssertionError("item on a channel-less spyre tensor must fail loudly")
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
