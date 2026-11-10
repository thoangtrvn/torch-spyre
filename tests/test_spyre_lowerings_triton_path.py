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

"""Clamp-adapter gating tests for enable_spyre_lowerings(triton_path).

The non-triton branch installs aten.clamp adapters whose inner_fn calls
lower_clamp -> V.ops.clamp — a handler only the (deleted) SDSC ops class
defined. TritonKernelOverrides has no ``clamp``, so installing the
adapters on the Triton path made any clamp-bearing graph die at Triton
codegen with AttributeError. These tests pin:

  - the triton path (explicit AND the default) installs no clamp adapter;
  - clamp_min/clamp_max keep their in-tree maximum/minimum lowerings;
  - the CM restores the lowerings table on exit.

Scope: lowerings-table state only — no end-to-end compile, no clamp-bearing
graph execution, no silicon. The adapter-absence assertion is name-based
(adapter functions are identified by their _impl_lower_aten_clamp* names /
"clamp" in __name__); an adapter installed under an unrelated name or via a
nameless wrapper would evade it (the restore test, by contrast, checks
identity with ``is``).
"""

import pytest
import torch

torch_clamp = torch.ops.aten.clamp.default
torch_clamp_min = torch.ops.aten.clamp_min.default
torch_clamp_max = torch.ops.aten.clamp_max.default


def _lowerings():
    from torch._inductor import lowering

    return lowering.lowerings


@pytest.mark.parametrize("kwargs", [{}, {"triton_path": True}],
                         ids=["default", "explicit-true"])
def test_triton_path_installs_no_clamp_adapter(kwargs):
    from torch_spyre._inductor.lowering import enable_spyre_lowerings

    with enable_spyre_lowerings(**kwargs):
        table = _lowerings()
        clamp_fn = table.get(torch_clamp)
        assert clamp_fn is None or "clamp" not in getattr(
            clamp_fn, "__name__", ""
        ), clamp_fn
        for ov in (torch_clamp_min, torch_clamp_max):
            fn = table[ov]
            # In-tree lowerings only: none of the adapter closures
            # (_impl_lower_aten_clamp*) may be installed.
            assert not getattr(fn, "__name__", "").startswith(
                "_impl_lower_aten_clamp"
            ), (ov, fn)


def test_lowerings_table_restored_on_exit():
    from torch_spyre._inductor.lowering import enable_spyre_lowerings

    before = dict(_lowerings())
    with enable_spyre_lowerings(triton_path=True):
        pass
    after = _lowerings()
    for ov in (torch_clamp, torch_clamp_min, torch_clamp_max):
        assert after.get(ov) is before.get(ov), ov


def test_exit_does_not_pop_excluded_intree_lowerings():
    """Leak regression: mean.dim is in spyre_lowerings AND in
    _TRITON_INCOMPATIBLE_LOWERINGS — on the triton path it is never
    installed, so the exit loop must NOT pop it (the old exit popped every
    spyre_lowerings op absent from the saved dict, deleting the in-tree
    lowering)."""
    from torch._inductor import lowering

    from torch_spyre._inductor.lowering import enable_spyre_lowerings

    mean_ov = torch.ops.aten.mean.dim
    before = lowering.lowerings.get(mean_ov)
    assert before is not None, "in-tree mean.dim lowering must pre-exist"
    with enable_spyre_lowerings(triton_path=True):
        pass
    assert lowering.lowerings.get(mean_ov) is before, (
        "exit popped the in-tree mean.dim lowering it never installed"
    )
