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

"""R0_BLOCK key-set / kernel loop-vs-persistent decision parity tests.

Pins the law that the injected fixed-config R0_BLOCK key is derived from the
CONSTRUCTED TritonKernel's own loop/persistent decision, not from a parallel
Inductor prediction that can drift from it.

The drift mechanism (derived from installed torch + an offline compile-only
reproduction, not transcribed): ``TritonScheduling.create_kernel_choices``
calls ``TritonKernel.apply_feature_required_overrides`` BEFORE the
``triton_kernel_kwargs`` hook, and that call FORCES
``kernel_kwargs["override_persistent_reduction"] = False`` whenever
``TritonKernel.has_persistent_RBLOCK(reduction_numel)`` fails — which it does
for dynamic reduction extents whose upper bound is not statically known at
that point (executed witness: dynamic masked-sum pooling, reduction_numel
``s20`` with bounds ``[2, int_oo]`` at override time). The kernel is then
constructed LOOPED no matter what the heuristics predict. The hook's own
dim-resolution subsequently specializes the symbol (bounds ``[16,16]``), so
``should_use_persistent_reduction(features, cooperative_reduction=False)``
evaluated inside the hook predicts PERSISTENT and omits R0_BLOCK — leaving
the looped kernel's R0_BLOCK signature parameter unbound (codegen-time
``KeyError: 'R0_BLOCK'`` in ``max_block`` on this branch; the ast_to_ttir
constexpr wall once a query-side fallback answers the codegen query).

The fix keys the injection on the kernel's own decision inputs, in the same
order ``SIMDKernel.__init__`` resolves them (torch/_inductor/codegen/simd.py):
``override_persistent_reduction`` from kernel_kwargs wins; otherwise
``inside_reduction and should_use_persistent_reduction(features,
cooperative_reduction)`` with the kernel's own cooperative_reduction
(``override_cooperative_reduction`` if set, else
``should_use_cooperative_reduction(device, numel, reduction_numel)``).

The parity invariant under test is BOTH directions (the ASTSource
exactly-declared-keys constraint):
- looped kernel  => R0_BLOCK IS injected (this lane's defect);
- persistent kernel => R0_BLOCK is NOT injected (#193 regression:
  ``ValueError: 'R0_BLOCK' is not in list`` when bound into a persistent
  signature).

Mutation targets: reverting the injection gate to the bare
``should_use_persistent_reduction(features, cooperative_reduction=False)``
prediction reddens ``test_dynamic_pooling_r0_block_parity`` (the witness
kernel is looped with R0_BLOCK missing). Injecting R0_BLOCK unconditionally
reddens ``test_static_persistent_reduction_has_no_r0_block`` (the #193
direction) and the parity assertions in the looped cases.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest


def _run_compile(script: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    env.pop("SENCORES", None)
    env.pop("SPYRE_INJECT_BINARY", None)
    env.pop("SPYRE_INJECT_BINARY_ACK", None)
    # Isolate every compile cache per run — no ambient cache state.
    for var, prefix in (
        ("TORCHINDUCTOR_CACHE_DIR", "/tmp/r0pd_ind_"),
        ("TRITON_CACHE_DIR", "/tmp/r0pd_tri_"),
        ("SENTIENT_CACHE_DIR", "/tmp/r0pd_sen_"),
    ):
        env[var] = tempfile.mkdtemp(prefix=prefix)
    # Run from THIS checkout so the subprocess imports the worktree's
    # torch_spyre (cwd is sys.path[0] under python -c), not the shared clone
    # an editable install might point at.
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


# Records every constructed TritonKernel's own loop/persistent decision and
# the fixed_config the hook injected for it, then runs one compile. The
# per-case setup/call text is spliced in at the two markers (plain .replace,
# NOT f-strings — the script body is full of braces an f-string would eat).
_RECORDER_SCRIPT = """
    import json

    import torch
    import torch._dynamo as dynamo
    import torch_spyre

    assert "site-packages" not in torch_spyre.__file__, torch_spyre.__file__

    from torch._inductor.codegen.triton import TritonKernel

    kernels = []

    _orig_init = TritonKernel.__init__

    def recording_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        cfg = dict(self.fixed_config.config) if self.fixed_config else None
        kernels.append({
            "persistent_reduction": self.persistent_reduction,
            "cooperative_reduction": self.cooperative_reduction,
            "inside_reduction": self.inside_reduction,
            "tiling": {k: str(v) for k, v in self.tiling.items()},
            "fixed_config": cfg,
        })

    TritonKernel.__init__ = recording_init

    dev = "spyre"
    # CASE_SETUP
    final_exc = None
    try:
        CALL
    except Exception as e:
        final_exc = e

    chain = []
    node = final_exc
    seen = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        chain.append(f"{type(node).__name__}: {node}")
        node = node.__cause__ or node.__context__
    print("R0PD_RESULT " + json.dumps({
        "kernels": kernels,
        "final_exc": f"{type(final_exc).__name__}" if final_exc else None,
        "chain": chain[:6],
    }))
"""


def _recorder_body(case_setup: str, call: str) -> str:
    script = _RECORDER_SCRIPT.replace("        CALL", "        " + call)
    setup = textwrap.dedent(case_setup)
    return script.replace("    # CASE_SETUP", textwrap.indent(setup, "    "))


def _reduction_kernels(summary: dict) -> list[dict]:
    """Kernels whose tiling carries a reduction range (r-prefix) or that
    Inductor classifies as reductions — the R0_BLOCK gate's domain."""
    return [
        k
        for k in summary["kernels"]
        if any(prefix.startswith("r") for prefix in k["tiling"])
        or k["inside_reduction"]
    ]


def test_dynamic_pooling_r0_block_parity():
    """Red-first: the dynamic masked-sum pooling witness (looped kernel).

    At the defect base the hook's prediction says PERSISTENT for the dynamic
    reduction extent while the constructed kernel is FORCED looped by
    ``override_persistent_reduction=False`` (set before the hook by
    ``apply_feature_required_overrides``, because the dynamic extent has no
    static upper bound at that point) — so R0_BLOCK is missing from the
    injected key set and the compile dies at the R0_BLOCK wall. After the fix
    the key set matches the kernel's own decision and the compile proceeds
    past that wall (to whatever the next codegen-only wall is — the assert is
    wall-independent: the R0_BLOCK wall must be gone, and the looped kernel
    must carry R0_BLOCK).
    """
    script = _recorder_body(
        case_setup=(
            "x = torch.empty(4, 16, 384, dtype=torch.float32).to(dev)\n"
            "m = torch.ones(4, 16, dtype=torch.float32).to(dev)\n"
            "dynamo.mark_dynamic(x, 0)\n"
            "dynamo.mark_dynamic(x, 1)\n"
            "dynamo.mark_dynamic(m, 0)\n"
            "dynamo.mark_dynamic(m, 1)"
        ),
        call=(
            "torch.compile(lambda a, b: (a * b.unsqueeze(-1)).sum(dim=1))(x, m)"
        ),
    )
    result = _run_compile(script)
    assert result.returncode == 0, result.stderr
    lines = [l for l in result.stdout.splitlines() if l.startswith("R0PD_RESULT ")]
    assert lines, f"no result line; stderr={result.stderr[-2000:]}"
    summary = json.loads(lines[0][len("R0PD_RESULT "):])

    reds = _reduction_kernels(summary)
    assert reds, "no reduction kernel was constructed — stale pin"
    for k in reds:
        # R0_BLOCK is injected iff the kernel is NOT persistent (looped):
        # a persistent kernel bakes it in-body and must NOT carry the key.
        assert ("R0_BLOCK" in (k["fixed_config"] or {})) == (
            not k["persistent_reduction"]
        ), (
            f"R0_BLOCK key set does not match the kernel's own decision: "
            f"persistent_reduction={k['persistent_reduction']}, "
            f"fixed_config={k['fixed_config']}"
        )
    # The witness kernel is the drift itself: looped AND carrying R0_BLOCK.
    assert any(
        (not k["persistent_reduction"]) and "R0_BLOCK" in (k["fixed_config"] or {})
        for k in reds
    ), f"no looped reduction kernel with R0_BLOCK injected: {reds}"
    # The R0_BLOCK wall is gone: no KeyError('R0_BLOCK') and no
    # ASTSource 'R0_BLOCK' is not in list anywhere in the failure chain.
    chain_text = "\n".join(summary["chain"])
    assert "KeyError: 'R0_BLOCK'" not in chain_text, summary["chain"]
    assert "'R0_BLOCK' is not in list" not in chain_text, summary["chain"]


def test_static_persistent_reduction_has_no_r0_block():
    """#193 regression guard (red-first on the reverse mutation): a STATIC
    small reduction extent is persistent — the kernel bakes R0_BLOCK as an
    in-body constexpr and does NOT declare it as a signature parameter, so
    the injected key set must NOT carry R0_BLOCK (unconditional injection is
    the original #193 ``ValueError: 'R0_BLOCK' is not in list`` break).
    """
    script = _recorder_body(
        case_setup=(
            "x = torch.empty(4, 16, 384, dtype=torch.float32).to(dev)\n"
            "m = torch.ones(4, 16, dtype=torch.float32).to(dev)"
        ),
        call=(
            "torch.compile(lambda a, b: (a * b.unsqueeze(-1)).sum(dim=1))(x, m)"
        ),
    )
    result = _run_compile(script)
    assert result.returncode == 0, result.stderr
    lines = [l for l in result.stdout.splitlines() if l.startswith("R0PD_RESULT ")]
    assert lines, f"no result line; stderr={result.stderr[-2000:]}"
    summary = json.loads(lines[0][len("R0PD_RESULT "):])

    reds = _reduction_kernels(summary)
    assert reds, "no reduction kernel was constructed — stale pin"
    persistents = [k for k in reds if k["persistent_reduction"]]
    assert persistents, (
        f"expected at least one persistent reduction kernel for the static "
        f"shape (rnumel=16): {reds}"
    )
    for k in persistents:
        assert "R0_BLOCK" not in (k["fixed_config"] or {}), (
            f"R0_BLOCK injected into a persistent-reduction signature "
            f"(#193 regression): {k}"
        )
    chain_text = "\n".join(summary["chain"])
    assert "'R0_BLOCK' is not in list" not in chain_text, summary["chain"]


def test_static_looped_reduction_carries_r0_block():
    """Large static reduction extent: the kernel is looped for real (no
    override, prediction looped) — R0_BLOCK must be present (both the base
    behavior and the fix agree here; pins that the fix did not move the
    no-override path).
    """
    script = _recorder_body(
        case_setup="x = torch.empty(4, 4096, 384, dtype=torch.float32).to(dev)",
        call="torch.compile(lambda a: a.sum(dim=1))(x)",
    )
    result = _run_compile(script)
    assert result.returncode == 0, result.stderr
    lines = [l for l in result.stdout.splitlines() if l.startswith("R0PD_RESULT ")]
    assert lines, f"no result line; stderr={result.stderr[-2000:]}"
    summary = json.loads(lines[0][len("R0PD_RESULT "):])

    reds = _reduction_kernels(summary)
    assert reds, "no reduction kernel was constructed — stale pin"
    looped = [k for k in reds if not k["persistent_reduction"]]
    assert looped, (
        f"expected at least one looped reduction kernel for rnumel=4096: {reds}"
    )
    for k in looped:
        assert "R0_BLOCK" in (k["fixed_config"] or {}), (
            f"looped reduction kernel missing R0_BLOCK: {k}"
        )
