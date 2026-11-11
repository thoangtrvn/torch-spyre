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

"""FixedTritonConfig block-query fallback tests.

Pins the law that the injected config object owns the TOTAL function over
block-size queries: torch's ``TritonKernel.max_block`` does a raw
``self.fixed_config[f"{prefix.upper()}BLOCK"]`` for ANY range-tree prefix
(torch/_inductor/codegen/triton.py, via ``_has_constant_mask`` ->
``iteration_ranges_codegen_header``), but the injected key set carries
EXACTLY the signature-declared ``*_BLOCK`` keys (the ASTSource binding
invariant documented on ``SpyreHeuristics.triton_kernel_kwargs`` — an extra
key the kernel does not declare raises ``ValueError: '<KEY>' is not in list``,
so the key set cannot be widened to cover the queries). Any kernel whose
range-tree prefixes exceed the predicted key set therefore died with
``KeyError: 'YBLOCK'`` / ``KeyError: 'R0_BLOCK'`` at codegen time. On CUDA
``fixed_config is None`` (autotune) and ``max_block`` falls back to
``TRITON_MAX_BLOCK``; ``SpyreFixedTritonConfig`` restores that same fallback
on the query side only.

The witnessed casualty class (MiniLM masked-sum pooling through the vLLM
compile-gap probe, wall #2) is reproduced minimally here as a dynamic-shape
masked-sum pooling reduction: the hook's ``should_use_persistent_reduction``
prediction and the kernel's own looped/persistent decision disagree for
dynamic reduction extents, so the kernel queries ``max_block("r0_")`` for an
uninjected key. Before the fix the compile died inside ``max_block`` with
``KeyError: 'R0_BLOCK'``; after the fix every block query is answered and
the compile proceeds past codegen to its NEXT wall. That next wall (the
R0_BLOCK runtime constexpr binding — the same persistent-prediction drift,
now surfacing at ast_to_ttir) is a key-set/prediction defect this lane
deliberately does NOT own: fixing it would require widening the predicted
key set, which the binding invariant forbids doing blindly. The end-to-end
test therefore pins "every max_block query is answered" (wall-independent),
not "the whole compile succeeds".

Mutation targets: replacing ``SpyreFixedTritonConfig`` with plain
``FixedTritonConfig`` at the construction sites must redden
``test_construction_sites_use_the_wrapper`` (the source pin) and
``test_pooling_max_block_queries_all_answered`` (the recorded query list
shows the unresolved KeyError). Deleting or breaking the fallback in the
class itself additionally reddens
``test_undeclared_block_keys_fall_back_to_triton_max_block`` (that unit
test imports the class directly, so it is blind to the construction-site
mutation).
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Unit pins: the wrapper's lookup/contains/eq/hash surface, in-process.
# ---------------------------------------------------------------------------


def test_undeclared_block_keys_fall_back_to_triton_max_block():
    """Undeclared <PREFIX>BLOCK queries resolve to torch's TRITON_MAX_BLOCK.

    TRITON_MAX_BLOCK is imported from torch AT TEST TIME, so if the wrapper
    ever hardcodes or drifts from torch's current table, this reddens.
    """
    from torch._inductor.codegen.triton import (
        TRITON_MAX_BLOCK,
    )

    from torch_spyre._inductor.choices import SpyreFixedTritonConfig

    cfg = SpyreFixedTritonConfig({"XBLOCK": 32})
    assert cfg["XBLOCK"] == 32
    for key, prefix in (
        ("YBLOCK", "Y"),
        ("ZBLOCK", "Z"),
        ("R0_BLOCK", "R0_"),
        ("R1_BLOCK", "R1_"),
    ):
        assert key not in cfg, f"{key} must not be contained (exact-key __contains__)"
        assert cfg[key] == TRITON_MAX_BLOCK[prefix], (
            f"{key} must fall back to TRITON_MAX_BLOCK[{prefix!r}]"
        )
    # Declared keys win over the fallback.
    assert SpyreFixedTritonConfig({"YBLOCK": 32})["YBLOCK"] == 32


def test_non_block_keys_still_keyerror():
    """Keys that do not name a TRITON_MAX_BLOCK prefix keep raising KeyError.

    'RSPLIT' (read by TritonKernel.max_rsplit on cooperative reductions) has
    no TRITON_MAX_BLOCK analogue — silently answering it would change
    cooperative-reduction semantics this lane does not own.
    """
    from torch_spyre._inductor.choices import SpyreFixedTritonConfig

    cfg = SpyreFixedTritonConfig({"XBLOCK": 32})
    with pytest.raises(KeyError):
        cfg["RSPLIT"]
    with pytest.raises(KeyError):
        cfg["problem_m"]


def test_cache_key_surface_is_unchanged():
    """eq/hash/config identity vs the plain FixedTritonConfig.

    FixedTritonConfig is part of the autotune cache-key channel: codegen
    serializes the runtime config from ``.config`` (a plain dict literal via
    ``{self.fixed_config.config!r}``), which is where the cache key is
    derived from. The wrapper adds no fields and does not touch ``.config``.
    The plain dataclass is UNHASHABLE (eq without frozen), so hashing cannot
    be part of the channel; the equality surface must simply match the
    plain-on-plain semantics for every pair — derived here, not assumed.
    """
    from torch._inductor.codegen.triton import FixedTritonConfig

    from torch_spyre._inductor.choices import SpyreFixedTritonConfig

    d = {"XBLOCK": 32, "R0_BLOCK": 32, "problem_m": 4, "num_cores": 32}
    wrapped = SpyreFixedTritonConfig(dict(d))
    plain = FixedTritonConfig(dict(d))
    assert wrapped.config == d
    # eq against the plain class, both directions (a naive dataclass
    # subclass would compare False here — a silent cache-hit change).
    assert wrapped == plain
    assert plain == wrapped
    assert wrapped == SpyreFixedTritonConfig(dict(d))
    assert wrapped != FixedTritonConfig({"XBLOCK": 64})
    assert wrapped != object()
    # Both classes are unhashable — the object itself cannot be a hash key.
    for cfg in (wrapped, plain):
        with pytest.raises(TypeError):
            hash(cfg)


def test_construction_sites_use_the_wrapper():
    """Source-level guard: every injected fixed_config is the wrapper.

    The hook builds fixed_config at four sites (matmul, reduction, pointwise,
    and the final re-wrap that carries the problem-dims channel). If any site
    regresses to the plain FixedTritonConfig, kernels routed through that
    branch lose the query fallback silently.
    """
    import inspect
    import re

    from torch_spyre._inductor import choices

    src = inspect.getsource(choices)
    # Whitespace-normalized so a reindent/rewrap of the pointwise branch
    # cannot false-red the pin (the mutation target is the CLASS at the
    # construction site, not the formatting).
    flat = re.sub(r"\s+", " ", src)
    assert flat.count("SpyreFixedTritonConfig(block_config)") == 2
    # Pointwise site (merged form): the YBLOCK/ZBLOCK tiling-dict-gated
    # pointwise_block_config from the codegen-only arc, wrapped in the
    # lane's subclass. Pin BOTH the wrapper class and the dict name so a
    # regression to either the plain class or the XBLOCK-only dict reddens.
    assert "SpyreFixedTritonConfig( pointwise_block_config )" in flat
    assert 'pointwise_block_config = {"XBLOCK": elements_per_stick}' in flat
    assert 'pointwise_block_config["YBLOCK"] = elements_per_stick' in flat
    assert "SpyreFixedTritonConfig(config_dict)" in flat
    # The empty-default read at the problem-dims channel stays plain (it is
    # only .config-copied, never queried by prefix).
    assert (
        'kernel_kwargs.get( "fixed_config", FixedTritonConfig({}) ).config.copy()'
        in flat
    )


# ---------------------------------------------------------------------------
# End-to-end red-first pin: every max_block query is answered.
# ---------------------------------------------------------------------------


def _run_script(script: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    env.pop("SENCORES", None)
    env.pop("SPYRE_INJECT_BINARY", None)
    env.pop("SPYRE_INJECT_BINARY_ACK", None)
    # Isolate every compile cache per run (same discipline as
    # test_bmm_zblock_fixed_config: no ambient cache state).
    for var, prefix in (
        ("TORCHINDUCTOR_CACHE_DIR", "/tmp/fcbf_ind_"),
        ("TRITON_CACHE_DIR", "/tmp/fcbf_tri_"),
        ("SENTIENT_CACHE_DIR", "/tmp/fcbf_sen_"),
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


_POOLING_SCRIPT = """
    import json
    import os

    import torch
    import torch._dynamo as dynamo
    import torch_spyre

    # Fail loud if the subprocess resolved a different torch_spyre (e.g. an
    # editable install pointing at a shared clone) — the pin would silently
    # test the wrong tree.
    assert "site-packages" not in torch_spyre.__file__, torch_spyre.__file__

    from torch._inductor.codegen.triton import TritonKernel

    queries = []

    _orig_max_block = TritonKernel.max_block

    def recording_max_block(self, prefix):
        key = f"{prefix.upper()}BLOCK"
        try:
            value = _orig_max_block(self, prefix)
        except KeyError:
            queries.append({"prefix": prefix, "resolved": False})
            raise
        declared = key in (self.fixed_config.config if self.fixed_config else {})
        queries.append({"prefix": prefix, "resolved": True,
                        "declared": declared, "value": value})
        return value

    TritonKernel.max_block = recording_max_block

    dev = "spyre"
    x = torch.empty(4, 16, 384, dtype=torch.float32).to(dev)
    m = torch.ones(4, 16, dtype=torch.float32).to(dev)
    dynamo.mark_dynamic(x, 0)
    dynamo.mark_dynamic(x, 1)
    dynamo.mark_dynamic(m, 0)
    dynamo.mark_dynamic(m, 1)

    final_exc = None
    try:
        torch.compile(lambda a, b: (a * b.unsqueeze(-1)).sum(dim=1))(x, m)
    except Exception as e:
        final_exc = e

    # Walk the full cause/context chain: the pinned defect is a KeyError
    # raised by the fixed-config lookup inside max_block, whatever wrapper
    # Inductor puts around it.
    chain = []
    seen = set()
    node = final_exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        chain.append(f"{type(node).__name__}: {node}")
        node = node.__cause__ or node.__context__
    chain_text = "\\n".join(chain)

    unresolved = [q for q in queries if not q["resolved"]]
    fallback = [q for q in queries if q["resolved"] and not q.get("declared")]
    print("FCBF_RESULT " + json.dumps({
        "queries": queries,
        "unresolved": len(unresolved),
        "fallback_answered": len(fallback),
        "final_exc": f"{type(final_exc).__name__}" if final_exc else None,
        "keyerror_in_chain": "KeyError" in chain_text,
        "chain": chain[:6],
    }))
"""


def test_pooling_max_block_queries_all_answered():
    """Red-first: the masked-sum pooling class no longer dies in max_block.

    At the defect base the compile died inside ``TritonKernel.max_block``
    with ``KeyError: 'R0_BLOCK'`` (the hook's persistent-reduction prediction
    omits R0_BLOCK; the looped kernel still queries it). With the query-side
    fallback every max_block query is answered — declared keys from the
    injected value, undeclared keys from torch's TRITON_MAX_BLOCK — and the
    compile proceeds past codegen to its next wall.

    The assertions are deliberately wall-independent: the compile is EXPECTED
    to still stop somewhere (codegen-only environment), but that stop must
    not be a fixed-config KeyError from the block-size query surface. Since
    the r0-drift merge (e68375aa) the R0_BLOCK key-set gate keys on the
    kernel's own looped decision, so this shape's queries are all declared
    and the fallback must stay silent — a fallback answer here would mean
    the gate regressed to the bare persistent-reduction prediction.
    """
    result = _run_script(_POOLING_SCRIPT)
    assert result.returncode == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("FCBF_RESULT ")]
    assert lines, f"no result line; stderr={result.stderr[-2000:]}"
    summary = json.loads(lines[0][len("FCBF_RESULT ") :])

    assert summary["queries"], "no max_block queries were made — stale pin"
    assert summary["unresolved"] == 0, (
        f"unresolved max_block queries: {summary['queries']}"
    )
    # Re-pinned at the r0-drift merge (e68375aa): _kernel_persistent_reduction
    # now keys R0_BLOCK injection on the kernel's own looped decision, so the
    # dynamic pooling shape's r0_ queries are all DECLARED and the fallback
    # correctly stays silent (fallback_answered == 0). The earlier
    # `fallback_answered > 0` tripwire keyed on the prediction-drift defect
    # this merge fixes; the fallback mechanism itself stays pinned at unit
    # level by test_undeclared_block_keys_fall_back_to_triton_max_block.
    assert summary["fallback_answered"] == 0, (
        "the fallback fired on the dynamic pooling shape — R0_BLOCK should "
        "be declared by _kernel_persistent_reduction since the r0-drift "
        "merge; a fallback answer here means the key-set gate regressed to "
        "the bare prediction"
    )
    assert not summary["keyerror_in_chain"], (
        f"KeyError still in the failure chain: {summary['chain']}"
    )
