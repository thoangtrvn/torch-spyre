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

"""Tests for _embedding_host.py — torch-free embedding launch planner.

Loaded via importlib by path so that this test never triggers torch_spyre/__init__.py
(which imports torch). Run with:

    PYTHONPATH=/Users/tmhoangt/Codes.IBM/AIU/spyre-knowledgebase/codegen \
        /Users/tmhoangt/Codes.IBM/AIU/spyre-knowledgebase/mcp-server/.venv/bin/python \
        -m pytest tests/test_embedding_host.py -v
"""
import importlib.util, os
HELPER = os.path.join(os.path.dirname(__file__), "..", "torch_spyre", "ops", "_embedding_host.py")

def _load_helper():
    spec = importlib.util.spec_from_file_location("_embedding_host", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_build_launches_single_launch_small():
    h = _load_helper()
    # d_model=64 -> 1 stick/token; 4 tokens -> 1 launch, slice (0,4)
    launches, tokens_per_launch = h.build_embedding_launches(vocab=1000, d_model=64, element_bits=16, flat_idx=[3, 1, 2, 0])
    assert len(launches) == 1
    (start, end), binary = launches[0]
    assert (start, end) == (0, 4)
    assert isinstance(binary, bytes) and len(binary) % 128 == 0
    # tokens_per_launch == num_cores * K: N=4 <= ebr_count(8) -> K=4, num_cores=1 -> 4
    assert tokens_per_launch == 4

def test_build_launches_multipass_covers_all_tokens():
    h = _load_helper()
    # d_model=64 (single-stick, hardware-verified); N=300 -> 2 launches covering [0,300)
    # (300 > 256 = 32 cores * 8 tokens/core, so multi-launch path is exercised)
    launches, tokens_per_launch = h.build_embedding_launches(vocab=128000, d_model=64, element_bits=16, flat_idx=list(range(300)))
    slices = [s for s, _ in launches]
    assert slices[0][0] == 0 and slices[-1][1] == 300
    for i in range(len(slices) - 1):
        assert slices[i][1] == slices[i + 1][0]  # no gaps/overlaps
    # tokens_per_launch is the binary's output size: 32 cores * 8 tokens/core = 256
    # (this is the key invariant: the final launch [256,300) writes into 256-row buf,
    # not a 44-row slice, preventing the hardware overrun bug)
    assert tokens_per_launch == 256


def test_build_launches_rejects_multistick_d_model():
    import pytest
    h = _load_helper()
    # d_model=2048 at 16-bit requires 32 sticks/token — multi-stick is NOT hardware-verified
    with pytest.raises(NotImplementedError, match="single-stick|multi-stick"):
        h.build_embedding_launches(vocab=1000, d_model=2048, element_bits=16, flat_idx=[0, 1])

def test_build_launches_rejects_out_of_vocab():
    import pytest
    h = _load_helper()
    with pytest.raises(ValueError):
        h.build_embedding_launches(vocab=10, d_model=64, element_bits=16, flat_idx=[12])
