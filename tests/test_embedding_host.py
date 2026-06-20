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
    launches = h.build_embedding_launches(vocab=1000, d_model=64, element_bits=16, flat_idx=[3, 1, 2, 0])
    assert len(launches) == 1
    (start, end), binary = launches[0]
    assert (start, end) == (0, 4)
    assert isinstance(binary, bytes) and len(binary) % 128 == 0

def test_build_launches_multipass_covers_all_tokens():
    h = _load_helper()
    # 300 tokens, ceiling 256 (32 cores x 8) -> 2 launches covering [0,300)
    launches = h.build_embedding_launches(vocab=128000, d_model=2048, element_bits=16, flat_idx=list(range(300)))
    slices = [s for s, _ in launches]
    assert slices[0][0] == 0 and slices[-1][1] == 300
    for i in range(len(slices) - 1):
        assert slices[i][1] == slices[i + 1][0]  # no gaps/overlaps

def test_build_launches_rejects_out_of_vocab():
    import pytest
    h = _load_helper()
    with pytest.raises(ValueError):
        h.build_embedding_launches(vocab=10, d_model=64, element_bits=16, flat_idx=[12])
