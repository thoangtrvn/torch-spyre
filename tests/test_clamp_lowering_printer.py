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

"""Clamp lowering must be printable by the Inductor loop-body handler chain.

torch_spyre's lower_clamp reroutes aten.clamp/clamp_min/clamp_max to a
spyre::clamp pointwise. Historically its inner_fn emitted ``ops.clamp`` into
the loop body — but no clamp printer exists anywhere in the Inductor handler
chain (stock torch decomposes clamp to min/max and never prints it), so every
full-model compile containing a clamp died at codegen with

    AttributeError: 'TritonKernelOverrides' object has no attribute 'clamp'

(vLLM compile-gap wall #3, stage0-local census 2026-11-09). The fix composes
``ops.maximum(min, ops.minimum(max, x))`` — the exact form of torch's own
in-tree clamp lowering — so the loop body only ever uses stock min/max
printers.

Every compiled spyre-path test here runs offline in codegen-only mode with
FakeTensor inputs (the compile_fx wrapper detects the spyre device from fake
inputs and skips runtime init), so no Spyre hardware or full runtime build is
needed: the success signal is that compile gets PAST loop-body codegen. The
acceptable terminal outcomes after the printer gap is closed are the same
taxon as test_t_sr59_dynamo_capture: the codegen-only launch refusal
(SpyreRuntimeUnavailableError) or the structured-production typed refusal
(T-SR-29, the known op-coverage blocker for the fused clamp kernel — the
min/max call helpers are not yet in the structured production subset).
"""

import os
import subprocess
import sys
import textwrap

import pytest


def _spyre_codegen_available() -> bool:
    """The codegen-only triton driver imports the KB sentient_codegen package."""
    try:
        import sentient_codegen  # noqa: F401

        return True
    except ImportError:
        return False


_codegen_needed = pytest.mark.skipif(
    not _spyre_codegen_available(),
    reason="sentient_codegen (KB codegen/) not on PYTHONPATH",
)


def _run_script(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


_SINGLE_ARG_CLAMP_VARIANTS = {
    "clamp": "torch.clamp(x, 1.0, 65504.0)",
    "clamp_min": "torch.clamp_min(x, 1.0)",
    "clamp_max": "torch.clamp_max(x, 65504.0)",
}


def _compile_clamp_script(expr: str) -> str:
    return f"""
        import torch
        import torch_spyre

        torch_spyre._autoload()

        def fn(x):
            return {expr}

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.empty(2, 64, dtype=torch.float32, device="spyre")
            cfn = torch.compile(fn, dynamic=False)
            try:
                cfn(x)
            except Exception as exc:
                chain = [f"{{type(exc).__name__}}: {{exc}}"]
                seen = {{id(exc)}}
                cur = exc
                while True:
                    nxt = cur.__cause__ or cur.__context__
                    if nxt is None or id(nxt) in seen:
                        break
                    seen.add(id(nxt))
                    chain.append(f"{{type(nxt).__name__}}: {{nxt}}")
                    cur = nxt
                text = " <- ".join(chain)
                # The printer gap this test guards: the loop body asked the
                # handler chain for a clamp printer that does not exist.
                if "TritonKernelOverrides' object has no attribute 'clamp'" in text:
                    raise SystemExit(4)
                # Compile returned; the terminal stop is a known coverage /
                # runtime boundary, not a codegen defect.
                if "SpyreRuntimeUnavailableError" in text:
                    raise SystemExit(0)
                if "StructuredProductionSubsetRefusedError" in text:
                    raise SystemExit(3)
                raise SystemExit(1)
            raise SystemExit(0)
    """


@_codegen_needed
@pytest.mark.parametrize("variant", sorted(_SINGLE_ARG_CLAMP_VARIANTS))
def test_clamp_lowering_prints_through_loop_body(variant: str):
    """aten.clamp variants must clear loop-body codegen (no clamp printer).

    Outcome taxonomy: exit 0 = compiled and ran (or launch-refused with the
    codegen-only sentinel); exit 3 = compile-stage T-SR-29 typed refusal (the
    known structured-production coverage blocker for the fused clamp kernel —
    acceptable, the printer contract already held); exit 4 = the printer gap
    regressed (ops.clamp emitted with no handler); anything else = unexpected
    failure class.
    """
    result = _run_script(
        _compile_clamp_script(_SINGLE_ARG_CLAMP_VARIANTS[variant])
    )
    if result.returncode == 3:
        pytest.xfail(
            "spyre structured-production envelope refused the fused clamp "
            "kernel at compile (T-SR-29 typed refusal) — the loop-body "
            "printer contract already held; lift this when the envelope "
            "admits the min/max call helpers"
        )
    assert result.returncode != 4, (
        "clamp lowering emitted ops.clamp into the Inductor loop body, but "
        "no clamp printer exists in the handler chain "
        f"({variant}):\n{result.stderr}"
    )
    assert result.returncode == 0, result.stderr
