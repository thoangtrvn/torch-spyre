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

"""Offline binary decode gate for embedding launch binaries.

Decodes PatchInit row headers from Task 1's build_embedding_launches output
and asserts structural invariants: multi-core patch rows must use separate
L3SU-output (0x40) and L3LU-source (0x80) slices — never a combined 0xC0
collision row. Passes immediately when the helper's geometry is correct.

Run with:

    PYTHONPATH=/Users/tmhoangt/Codes.IBM/AIU/spyre-knowledgebase/codegen \\
        /Users/tmhoangt/Codes.IBM/AIU/spyre-knowledgebase/mcp-server/.venv/bin/python \\
        -m pytest tests/test_embedding_decode.py -v
"""
import importlib.util, os
HELPER = os.path.join(os.path.dirname(__file__), "..", "torch_spyre", "ops", "_embedding_host.py")

def _load_helper():
    spec = importlib.util.spec_from_file_location("_embedding_host", HELPER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_multicore_binary_has_split_patch_rows_no_collision():
    from sentient_codegen.gen.rcudd1a.flit_formats import PATCH_INIT_HEADER_FLIT
    h = _load_helper()
    # 4 tokens, d_model=64 -> 4 tokens fit one launch; force 2 cores via 8 tokens
    launches = h.build_embedding_launches(vocab=1000, d_model=64, element_bits=16,
                                          flat_idx=[7, 6, 5, 4, 3, 2, 1, 0])
    (start, end), binary = launches[0]
    slices_seen = set()
    for r in range(len(binary) // 128):
        hdr = PATCH_INIT_HEADER_FLIT.decode(int.from_bytes(binary[r*128:r*128+16], "little"))
        if hdr.get("patch_flag"):
            slices_seen.add(hdr["target_slices"])
    # multi-core embedding: L3SU output (0x40) + L3LU source (0x80), never combined 0xC0
    assert 0xC0 not in slices_seen, "forbidden L3SU+L3LU collision row"
    assert 0x80 in slices_seen, "missing L3LU source rows"
