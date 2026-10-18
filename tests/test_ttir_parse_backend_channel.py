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

"""T-SR-40: torch-level ttir_parse_backend channel tests.

Pins the flip-soak knob end to end:

1. Opt-in ``config.ttir_parse_backend = "structured"`` lands BOTH the
   parse backend and the auto-paired ``require_stage0_admission=True``
   on the fork's ``SpyreOptions`` at ``_make_spyre_ir`` — the licensed
   combination only; the T-SR-38 trapped combination (structured without
   admission) is inexpressible from this channel.
2. Default ``None`` is behavior-neutral: nothing is injected, the fork's
   own defaults ("legacy", admission off) stand, so the options cache
   key is unchanged for every existing compile.
3. A misspelling refuses loud at the injection point (choices.py), the
   same refusal class as the fork's parse_options misspelling guard —
   never a silent fall-back to legacy.

Capture pattern follows test_descriptor_pointwise_patch /
test_sencores_bridge: monkeypatch ``SpyreBackend._make_spyre_ir`` so the
pin is on the channel (options landing), not on the parse itself (the
fork's test_parse_backend_wiring.py owns the routing pin).
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
    # Isolate every compile cache per run (T-SR-40 R2): the shared
    # inductor cache was demonstrably contaminated cross-run/cross-agent
    # (a concurrent mutation probe left structured-without-admission
    # wrappers that masqueraded as a worker-side admission drop).
    # Correctness is already protected by the options-hash and FX-key
    # splits, but these tests must not depend on ambient cache state.
    for var, prefix in (
        ("TORCHINDUCTOR_CACHE_DIR", "/tmp/tsr40_ind_"),
        ("TRITON_CACHE_DIR", "/tmp/tsr40_tri_"),
        ("SENTIENT_CACHE_DIR", "/tmp/tsr40_sen_"),
    ):
        cache_dir = tempfile.mkdtemp(prefix=prefix)
        env[var] = cache_dir
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


_CAPTURE_PREAMBLE = """
    import torch
    from triton.backends.spyre.compiler import SpyreBackend

    captured = []

    def capture_spyre_ir(mod, metadata, options):
        captured.append(options)
        raise RuntimeError('captured')

    SpyreBackend._make_spyre_ir = staticmethod(capture_spyre_ir)

    from torch_spyre._inductor import config as spyre_config
"""


def _compile_and_capture(opt_in: str) -> str:
    # Each piece is dedented separately before concatenation — the
    # preamble and the body carry the same 4-space source indent, the
    # per-test tails are flush-left, so a single dedent over the
    # concatenation cannot strip both.
    return textwrap.dedent(_CAPTURE_PREAMBLE) + textwrap.dedent(f"""
    spyre_config.ttir_parse_backend = {opt_in}

    @torch.compile
    def add(x):
        return x + 1

    x = torch.empty((512, 128), dtype=torch.float16).to('spyre')
    try:
        add(x)
    except Exception:
        if not captured:
            raise

    options = captured[0]
""")


def test_structured_opt_in_lands_licensed_combination():
    """structured → ttir_parse_backend="structured" AND
    require_stage0_admission=True on the fork SpyreOptions: the channel
    auto-pairs the T-SR-38 admission fence, so the convicted
    structured-without-admission trap is inexpressible from torch."""
    result = _run_script(
        _compile_and_capture('"structured"')
        + """
assert options.ttir_parse_backend == 'structured', options.ttir_parse_backend
assert options.require_stage0_admission is True, options.require_stage0_admission
"""
    )
    assert result.returncode == 0, result.stderr


def test_default_none_is_behavior_neutral():
    """Unset knob keeps the fork's own defaults — nothing injected, so
    the options cache key is unchanged for every existing compile."""
    result = _run_script(
        _compile_and_capture('None')
        + """
assert options.ttir_parse_backend == 'legacy', options.ttir_parse_backend
assert options.require_stage0_admission is False, options.require_stage0_admission
"""
    )
    assert result.returncode == 0, result.stderr


def test_explicit_legacy_lands_legacy():
    """Explicit "legacy" rides the channel unpaired (admission stays
    off) — legacy parsing needs no admission gate."""
    result = _run_script(
        _compile_and_capture('"legacy"')
        + """
assert options.ttir_parse_backend == 'legacy', options.ttir_parse_backend
assert options.require_stage0_admission is False, options.require_stage0_admission
"""
    )
    assert result.returncode == 0, result.stderr


def test_misspelling_refuses_loud():
    """A misspelled knob refuses at the injection point (choices.py)
    with the knob named — the same refusal class as the fork's
    parse_options misspelling guard; never a silent legacy fall-back."""
    result = _run_script(
        _compile_and_capture('"structure"')
        + """
assert captured, 'compile silently succeeded with a misspelled knob'
"""
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "ttir_parse_backend" in combined
    assert "'structure'" in combined
