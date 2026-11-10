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

"""YBLOCK/ZBLOCK fixed_config key-set tests (2D/3D-grid pointwise).

Pins the invariant whose pointwise-side gap broke 2D-grid pointwise
compiles (MiniLM pooled-embedding repro): every range-tree root the
scheduler emits declares a ``<PREFIX>BLOCK`` constexpr parameter, so a
pointwise tiling dict carrying a ``"y"`` (or ``"z"``) range REQUIRES
YBLOCK (ZBLOCK) in the fixed_config. Triton's ``max_block`` does a raw
``fixed_config[prefix + "BLOCK"]`` lookup, so omitting the key killed
``_has_constant_mask -> max_block("y")`` with ``KeyError: 'YBLOCK'`` —
the kernel never emitted. The fix adds YBLOCK=elements_per_stick /
ZBLOCK=1 keyed on the tiling dict (the scheduled range trees), not a
shape guess.

Behavioral pin, in-process (no compile stack): same harness shape as
test_bmm_zblock_fixed_config — stub features + a V.graph sizevars proxy.
"""

import torch

from torch_spyre._inductor.choices import SpyreHeuristics


class _StubBuffer:
    @staticmethod
    def get_dtype():
        return torch.float16

    @staticmethod
    def get_size():
        return [96, 64]


class _StubNode:
    node = _StubBuffer()


class _StubPointwiseFeatures:
    """Minimum feature surface the pointwise branch of
    ``triton_kernel_kwargs`` reads: not a reduction, no dot."""

    def is_reduction(self):
        return False

    def contains_op(self, name):
        return None

    def scheduler_nodes(self):
        return [_StubNode()]

    def numel_hints(self):
        return [96 * 64]

    def is_template(self):
        return False


def _fixed_config(tiling):
    """Call triton_kernel_kwargs in-process; return the fixed_config dict.

    Stubbing ``V.graph`` with int-passing sizevars resolvers — the same
    read-only surface a real compile provides. Class swap restored in
    ``finally`` so the patch never leaks.
    """
    from types import SimpleNamespace

    from torch._inductor import virtualized

    class _VProxy(virtualized._V):
        @property
        def graph(self):
            return SimpleNamespace(sizevars=SimpleNamespace(
                statically_known_int=lambda v: int(v),
                statically_known_leq=lambda a, b: int(a) <= int(b)))

    original_cls = virtualized.V.__class__
    virtualized.V.__class__ = _VProxy
    try:
        kernel_kwargs = {}
        SpyreHeuristics().triton_kernel_kwargs(
            None, _StubPointwiseFeatures(), [tiling], kernel_kwargs)
    finally:
        virtualized.V.__class__ = original_cls
    fixed_config = kernel_kwargs["fixed_config"]
    cfg = getattr(fixed_config, "config", fixed_config)
    return dict(cfg) if hasattr(cfg, "keys") else dict(vars(cfg))


def test_y_carrying_pointwise_tiling_gets_yblock():
    config = _fixed_config({"x": 4096, "y": 512})
    # problem_m/problem_n/num_cores are always-present kernel-arg keys in
    # the same config dict; the block-key claim is YBLOCK membership.
    assert "YBLOCK" in config, config
    # fp16: one stick is 64 elements; YBLOCK is stick-aligned like XBLOCK.
    assert config["YBLOCK"] == 64
    assert config["XBLOCK"] == 64
    assert "ZBLOCK" not in config, config


def test_y_and_z_carrying_pointwise_tiling_gets_yblock_and_zblock():
    config = _fixed_config({"x": 4096, "y": 512, "z": 8})
    assert config["YBLOCK"] == 64
    # Inductor's own bmm policy: the batch dimension stays unblocked.
    assert config["ZBLOCK"] == 1


def test_x_only_pointwise_tiling_stays_xblock_only():
    """Control leg: no y/z range trees means no YBLOCK/ZBLOCK keys —
    injecting them would make ASTSource raise 'YBLOCK' is not in list
    against a 1D-grid signature that does not declare them."""
    config = _fixed_config({"x": 4096})
    assert "YBLOCK" not in config, config
    assert "ZBLOCK" not in config, config
    assert config["XBLOCK"] == 64
