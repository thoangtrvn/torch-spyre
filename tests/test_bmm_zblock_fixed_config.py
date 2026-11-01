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

"""ZBLOCK fixed_config key-set tests (batched 3-range matmul).

Pins the invariant whose matmul-side gap broke every batched (3-range)
native-matmul compile:

``SpyreHeuristics.triton_kernel_kwargs`` must install a FixedTritonConfig
containing EXACTLY the ``*_BLOCK`` keys the generated kernel declares as
constexpr parameters — no more, no fewer. A 3-range (batched bmm) kernel
declares one ``<PREFIX>BLOCK`` constexpr per range-tree root, so a tiling
dict carrying a ``"z"`` range means ZBLOCK is a REQUIRED key. The
prediction used to emit only {YBLOCK, XBLOCK} (+ conditional R0_BLOCK),
leaving the ZBLOCK constexpr unbound: Triton's ast_to_ttir resolved
``ZBLOCK`` to None and crashed on ``tl.program_id(0) * ZBLOCK``
(AttributeError "'NoneType' object has no attribute 'type'", surfacing
from Inductor as a ZBLOCK KeyError). The fix adds ZBLOCK=1 when the
tiling dict carries a "z" key — 1 is Inductor's own bmm policy
(torch/_inductor/codegen/triton.py: "we force ZBLOCK to be always 1
during autotune"), so the key is a signature-shape obligation, not a
tiling decision.

Both directions of the invariant are pinned end-to-end. The concrete
kernel does not exist at the hook, so the ORACLE is what actually reaches
``triton.compile``: the ASTSource constants (the fixed_config keys,
index-resolved against the kernel's declared constexpr parameters) and
the compile getting PAST ast_to_ttir (the capture stub only runs if it
did). Constexpr names do not survive into TTIR text, so the constants
are recorded by wrapping triton.compile in the subprocess.

Capture pattern follows test_ttir_parse_backend_channel /
test_descriptor_pointwise_patch: monkeypatch
``SpyreBackend._make_spyre_ir`` with a stub that records the TTIR and
aborts the compile (the pre-T-SB-13b capture discipline), so the pin is
independent of the CompiledKernel metadata contract owned by the
triton-spyre fork.
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest


def _run_script(script: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    env.pop("SENCORES", None)
    env.pop("SPYRE_INJECT_BINARY", None)
    env.pop("SPYRE_INJECT_BINARY_ACK", None)
    # Isolate every compile cache per run (same discipline as
    # test_ttir_parse_backend_channel: no ambient cache state).
    for var, prefix in (
        ("TORCHINDUCTOR_CACHE_DIR", "/tmp/zblock_ind_"),
        ("TRITON_CACHE_DIR", "/tmp/zblock_tri_"),
        ("SENTIENT_CACHE_DIR", "/tmp/zblock_sen_"),
    ):
        cache_dir = tempfile.mkdtemp(prefix=prefix)
        env[var] = cache_dir
    # Run from THIS checkout so the subprocess imports the worktree's
    # torch_spyre (cwd is sys.path[0] under python -c), not the shared
    # clone an editable install might point at.
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


_CAPTURE_PREAMBLE = """
    import torch
    import triton
    from triton.backends.spyre.compiler import SpyreBackend

    captured = []

    def capture_spyre_ir(mod, metadata, options):
        captured.append(
            mod.str_nodebug() if hasattr(mod, "str_nodebug") else str(mod))
        raise RuntimeError('captured')

    SpyreBackend._make_spyre_ir = staticmethod(capture_spyre_ir)

    # Record what reaches triton.compile: the ASTSource constants ARE the
    # fixed_config keys (index-resolved against the kernel's declared
    # constexpr parameters) — the key-set oracle for this test.
    compile_records = []
    _orig_compile = triton.compile

    def recording_compile(src, *args, **kwargs):
        try:
            arg_names = list(src.fn.arg_names)
            constants = {
                arg_names[index[0]]: value
                for index, value in src.constants.items()
                if isinstance(index, tuple) and index and index[0] < len(arg_names)
            }
            compile_records.append((src.name, constants))
        except Exception as exc:  # pragma: no cover - probe failure
            raise AssertionError(f'compile record failed: {exc}')
        return _orig_compile(src, *args, **kwargs)

    triton.compile = recording_compile
"""


def _compile_bmm(batch: int) -> str:
    """Batched a@b through torch.compile; captures TTIR + compile records.

    batch=1 uses the 2-range schedule; batch>=2 with a batched b uses the
    3-range (ZBLOCK-declaring) schedule.
    """
    return textwrap.dedent(_CAPTURE_PREAMBLE) + textwrap.dedent(f"""
    @torch.compile
    def kernel(a, b):
        return a @ b

    a = torch.empty(({batch}, 96, 64), dtype=torch.float16).to('spyre')
    b = torch.empty(({batch}, 64, 64), dtype=torch.float16).to('spyre')
    try:
        kernel(a, b)
    except Exception:
        if not captured:
            raise

    assert captured, 'spyre_ir stage never ran — compile died before it'
    ttir = captured[0]
""")


def test_bmm_3range_compiles_and_binds_zblock():
    """The regression: a batched 3-range bmm compiles past ast_to_ttir.

    At the defect base the compile crashed INSIDE ast_to_ttir (ZBLOCK
    unbound -> None -> AttributeError), so the capture stub never ran and
    this test's subprocess exited nonzero. At the fix the kernel compiles
    AND the ZBLOCK key is present in the constants that reached
    triton.compile, with Inductor's bmm policy value 1.
    """
    result = _run_script(
        _compile_bmm(batch=2)
        + """
bmm_records = [c for name, c in compile_records if 'bmm' in name]
assert bmm_records, compile_records
constants = bmm_records[0]
assert 'ZBLOCK' in constants, f'ZBLOCK missing from constants: {constants}'
assert constants['ZBLOCK'] == 1, constants
assert 'YBLOCK' in constants and 'XBLOCK' in constants, constants
print('TTIR_OK')
"""
    )
    assert result.returncode == 0, result.stderr
    assert "TTIR_OK" in result.stdout


def test_matmul_2range_declares_no_zblock():
    """The no-more direction: a 2-range matmul must not get a ZBLOCK key.

    An injected key the kernel does not declare makes Triton's ASTSource
    raise ``ValueError: 'ZBLOCK' is not in list`` — the same class as the
    #193 R0_BLOCK break — so the fix must gate on the tiling dict's z
    range, not inject unconditionally. (A 2-range kernel's signature has
    no ZBLOCK parameter, so a correctly-gated config leaves it absent
    from the constants.)
    """
    result = _run_script(
        _compile_bmm(batch=1)
        + """
mm_records = [c for name, c in compile_records if 'mm' in name or 'matmul' in name]
if not mm_records:
    mm_records = [c for _, c in compile_records]
assert mm_records, compile_records
for constants in mm_records:
    assert 'ZBLOCK' not in constants, f'2-range kernel got ZBLOCK: {constants}'
print('TTIR_OK')
"""
    )
    assert result.returncode == 0, result.stderr
    assert "TTIR_OK" in result.stdout


def test_zblock_gate_lives_in_the_matmul_branch():
    """Source-level guard so a future edit cannot silently drop the gate.

    The end-to-end tests above are the behavioral pin; this keeps the
    z-gate spelled at the matmul fixed_config site (not moved, not
    unconditional).
    """
    import inspect

    from torch_spyre._inductor import choices

    src = inspect.getsource(choices)
    matmul_block = src.split("if is_matmul:", 1)[1].split("elif is_reduction:", 1)[0]
    assert '"z" in tiling' in matmul_block
    assert '"ZBLOCK"] = 1' in matmul_block
