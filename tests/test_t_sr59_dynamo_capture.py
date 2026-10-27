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

"""T-SR-59: Dynamo capture of stock HF decoder-cache usage on the spyre stack.

The torch_spyre monkey patches (Tensor.to, torch.empty, torch.empty_strided)
and the spyre::overwrite lowering must be Dynamo-transparent: a stock
transformers decoder block with a StaticCache must capture without graph
breaks, and the position bookkeeping must respect the 1p0 host-boundary law
(i64 is metadata-only on the device).

Every compiled spyre-path test here runs offline in codegen-only mode
(TORCH_SPYRE_CODEGEN_ONLY=1, explicit torch_spyre._autoload()): the compiled
artifact cannot launch (SpyreRuntimeUnavailableError), so reaching that
refusal is the success signal — an InductorError before it is a capture or
codegen defect.
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


def _transformers_importable() -> bool:
    """transformers must import cleanly in a fresh interpreter — the same
    environment the subprocess tests run in (a broken torchvision binary in
    the environment crashes its import only when torchvision's C extension
    has not already loaded; that is environment debt, not a seam defect —
    the workaround shim stays probe-only per T-SR-59 tasking)."""
    try:
        env = os.environ.copy()
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        env.pop("TORCH_SPYRE_CODEGEN_ONLY", None)
        # modeling_llama (not bare transformers) pulls torchvision via
        # processing_utils -> image_utils; that import is what breaks.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_spyre; "
                "from transformers.models.llama import modeling_llama",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


_codegen_needed = pytest.mark.skipif(
    not _spyre_codegen_available(),
    reason="sentient_codegen (KB codegen/) not on PYTHONPATH",
)
_transformers_needed = pytest.mark.skipif(
    not _transformers_importable(),
    reason="transformers not importable in this environment (torchvision)",
)


def _run_script(script: str, codegen_only: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Control the stack explicitly: never let an inherited AUTOLOAD=1 flip
    # import-only semantics into a registered stack.
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    if codegen_only:
        env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    else:
        env.pop("TORCH_SPYRE_CODEGEN_ONLY", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@_codegen_needed
def test_spyre_to_dtype_arg_captures_without_graph_break():
    """x.to(torch.float32) under tracing must not reach torch.device().

    Outcome taxonomy (the c2 structured-production envelope postdates the
    original test): exit 0 = compile returned and unique_graphs == 1;
    exit 3 = compile-stage T-SR-29 typed refusal (known op-coverage
    blocker, xfail); exit 1 / nonzero otherwise = capture regression or
    unexpected failure class. The capture contract (no graph breaks) is
    asserted FIRST in every branch.
    """
    result = _run_script(
        """
        import torch
        import torch_spyre

        torch_spyre._autoload()

        def fn(x):
            return x.to(torch.float32) + 1.0

        def _chain_text(exc):
            parts = [f"{type(exc).__name__}: {exc}"]
            seen = {id(exc)}
            while True:
                nxt = exc.__cause__ or exc.__context__
                if nxt is None or id(nxt) in seen:
                    break
                seen.add(id(nxt))
                parts.append(f"{type(nxt).__name__}: {nxt}")
                exc = nxt
            return " <- ".join(parts)


        x = torch.randn(4, 4).to("spyre")
        cfn = torch.compile(fn, dynamic=False)
        import sys

        outcome = "returned"
        try:
            cfn(x)
        except Exception as exc:
            chain = _chain_text(exc)
            print(chain, file=sys.stderr)
            if "SpyreRuntimeUnavailableError" in chain:
                # The codegen-only LAUNCH refusal sentinel (runtime
                # extension absent) — proves compile returned. Matched on
                # the class name, not the "codegen-only" substring: two
                # unrelated codegen-only errors inside torch_spyre/_C.py
                # also carry that substring and would mislabel.
                outcome = "launch_refused"
            elif "StructuredProductionSubsetRefusedError" in chain:
                # Compile-stage typed refusal from the c2 structured
                # production envelope (T-SR-29) — a KNOWN op-coverage
                # blocker class, not a capture regression. The capture
                # asserts below still run and still gate this outcome.
                outcome = "typed_refusal"
            else:
                raise SystemExit(1)

        import torch._dynamo.utils as dynamo_utils

        graph_breaks = dict(dynamo_utils.counters["graph_break"])
        assert not graph_breaks, graph_breaks
        if outcome == "typed_refusal":
            # unique_graphs only increments after the backend returns;
            # it is unobservable while compile refuses. Capture is the
            # contract under test here and is already asserted above.
            raise SystemExit(3)
        stats = dict(dynamo_utils.counters["stats"])
        assert stats.get("unique_graphs", 0) == 1, stats
        """
    )
    if result.returncode == 3:
        pytest.xfail(
            "spyre structured-production envelope refused this kernel at "
            "compile (T-SR-29 typed refusal) — the capture contract (no "
            "graph breaks) already passed; lift this when the envelope "
            "admits the fused add"
        )
    assert result.returncode == 0, result.stderr


@_codegen_needed
def test_torch_empty_on_spyre_device_captures_without_graph_break():
    """A torch.empty inside the graph must not hit the pybind layout ctor.

    Uses torch.empty DIRECTLY: torch.zeros reaches the C++ fill_ path and
    never calls the patched Python-level torch.empty, so a zeros-bodied
    test is green even on the unfixed parent — empty is the red-on-parent
    body that actually exercises spyre_empty under tracing. Acceptance is
    by counters, not exception text: no graph breaks, and unique_graphs
    == 1 whenever the backend returned. Outcome taxonomy: exit 0 =
    compile returned (unique_graphs asserted); exit 3 = compile-stage
    T-SR-29 typed refusal (known op-coverage blocker, xfail); exit 1 =
    capture regression or unexpected failure class.
    """
    result = _run_script(
        """
        import torch
        import torch_spyre

        torch_spyre._autoload()

        def fn(x):
            return x + torch.empty(1, x.shape[1], 32, device=x.device)

        def _chain_text(exc):
            parts = [f"{type(exc).__name__}: {exc}"]
            seen = {id(exc)}
            while True:
                nxt = exc.__cause__ or exc.__context__
                if nxt is None or id(nxt) in seen:
                    break
                seen.add(id(nxt))
                parts.append(f"{type(nxt).__name__}: {nxt}")
                exc = nxt
            return " <- ".join(parts)


        x = torch.randn(1, 8, 32).to("spyre")
        cfn = torch.compile(fn, dynamic=False)
        import sys

        outcome = "returned"
        try:
            cfn(x)
        except Exception as exc:
            chain = _chain_text(exc)
            print(chain, file=sys.stderr)
            if "SpyreRuntimeUnavailableError" in chain:
                # The codegen-only LAUNCH refusal sentinel (runtime
                # extension absent) — proves compile returned. Matched on
                # the class name, not the "codegen-only" substring: two
                # unrelated codegen-only errors inside torch_spyre/_C.py
                # also carry that substring and would mislabel.
                outcome = "launch_refused"
            elif "StructuredProductionSubsetRefusedError" in chain:
                # Compile-stage typed refusal from the c2 structured
                # production envelope (T-SR-29) — a KNOWN op-coverage
                # blocker class, not a capture regression. The capture
                # asserts below still run and still gate this outcome.
                outcome = "typed_refusal"
            else:
                # Any other execution-time failure (e.g. the triton-spyre
                # driver launcher segment-mismatch ValueError, another
                # lane) is AFTER compile: unique_graphs == 1 below still
                # proves the compile succeeded.
                outcome = "post_compile"

        import torch._dynamo.utils as dynamo_utils

        graph_breaks = dict(dynamo_utils.counters["graph_break"])
        assert not graph_breaks, graph_breaks
        if outcome == "typed_refusal":
            # unique_graphs only increments after the backend returns;
            # it is unobservable while compile refuses. Capture is the
            # contract under test here and is already asserted above.
            raise SystemExit(3)
        stats = dict(dynamo_utils.counters["stats"])
        assert stats.get("unique_graphs", 0) == 1, stats
        """
    )
    if result.returncode == 3:
        pytest.xfail(
            "spyre structured-production envelope refused this kernel at "
            "compile (T-SR-29 typed refusal) — the capture contract (no "
            "graph breaks) already passed; lift this when the envelope "
            "admits the fused add"
        )
    assert result.returncode == 0, result.stderr


@_codegen_needed
def test_torch_cat_on_spyre_compiles():
    """spyre::overwrite lowering must not require the deleted SDSC ops marker."""
    result = _run_script(
        """
        import torch
        import torch_spyre

        torch_spyre._autoload()

        def fn(a, b):
            return torch.cat([a, b], dim=1)

        a = torch.randn(2, 64).to("spyre")
        b = torch.randn(2, 64).to("spyre")
        cfn = torch.compile(fn, dynamic=False)
        try:
            cfn(a, b)
        except Exception as exc:
            # Compile must succeed; only the codegen-only launch refusal is
            # acceptable. An InductorError here means the overwrite lowering
            # is broken (the pre-fix failure was:
            #   AttributeError: 'TritonKernelOverrides' object has no
            #   attribute 'overwrite'
            # on every cat / constant_pad_nd consumer). Keyed on the
            # SpyreRuntimeUnavailableError sentinel class (raised only in
            # _runtime.require_runtime), not the bare "codegen-only"
            # substring — two unrelated codegen-only errors in _C.py carry
            # that substring too.
            chain = f"{type(exc).__name__}: {exc}"
            seen = {id(exc)}
            while True:
                nxt = exc.__cause__ or exc.__context__
                if nxt is None or id(nxt) in seen:
                    break
                seen.add(id(nxt))
                chain += f" <- {type(nxt).__name__}: {nxt}"
                exc = nxt
            assert "SpyreRuntimeUnavailableError" in chain, chain
        """
    )
    assert result.returncode == 0, result.stderr


@_transformers_needed
def test_static_cache_capture_and_execution_counts_on_cpu():
    """Acceptance leg that can execute offline: stock decoder + StaticCache.

    On CPU (stack imported but NOT registered, AUTOLOAD=0) with a recording
    backend around stock Inductor: 0 graph breaks, 2 unique graphs (prefill
    + decode), 2 backend compilations, and every one of the 13 calls served
    by a compiled artifact — a silent eager fallback or a recompile-per-token
    regression cannot pass this.
    """
    pytest.importorskip("transformers")
    result = _run_script(
        """
        import torch
        import torch_spyre  # import side effects only; stack NOT registered

        assert torch._dynamo.config.cache_size_limit == 8  # stock limit

        import torch._dynamo.utils as dynamo_utils
        from torch._inductor.compile_fx import compile_fx

        compiles = 0
        executions = 0

        def _recording(gm, example_inputs):
            global compiles, executions
            compiles += 1
            inner = compile_fx(gm, example_inputs)

            def run(*a, **kw):
                global executions
                executions += 1
                return inner(*a, **kw)

            return run

        from transformers.cache_utils import StaticCache
        from transformers.models.llama.modeling_llama import (
            LlamaConfig,
            LlamaDecoderLayer,
        )

        cfg = LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            vocab_size=128,
            _attn_implementation="sdpa",
        )
        layer = LlamaDecoderLayer(cfg, 0).eval()
        cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=128)

        def block_forward(hidden_states, position_ids, attn_mask, cache):
            seq_len = hidden_states.shape[1]
            rope = (
                torch.zeros(1, seq_len, 32),
                torch.ones(1, seq_len, 32),
            )
            out = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                position_embeddings=rope,
            )
            return out[0]

        fn = torch.compile(block_forward, dynamic=False, backend=_recording)

        def prefill(mask_len):
            hidden = torch.randn(1, 64, 128)
            positions = torch.arange(64).unsqueeze(0)
            causal = torch.triu(
                torch.full((64, 64), float("-inf")), diagonal=1
            ).fill_diagonal_(0.0)
            if mask_len == 64:
                mask = causal.unsqueeze(0).unsqueeze(0)
            else:
                mask = torch.full((1, 1, 64, mask_len), float("-inf"))
                mask[..., :64] = causal
            return hidden, positions, mask

        def decode(step, mask_len):
            hidden = torch.randn(1, 1, 128)
            positions = torch.tensor([[64 + step]])
            written = 64 + step + 1
            mask = torch.zeros(1, 1, 1, mask_len)
            mask[..., written:] = float("-inf")
            return hidden, positions, mask

        calls = 0
        for args in [prefill(128)] + [decode(s, 128) for s in range(12)]:
            calls += 1
            fn(*args, cache)

        graph_breaks = dict(dynamo_utils.counters["graph_break"])
        assert not graph_breaks, graph_breaks
        stats = dict(dynamo_utils.counters["stats"])
        assert stats.get("unique_graphs", 0) == 2, stats
        assert compiles == 2, compiles
        assert executions == 13, executions
        assert calls == 13
        """,
        codegen_only=False,
    )
    assert result.returncode == 0, result.stderr


@_codegen_needed
@_transformers_needed
def test_stock_decoder_static_cache_spyre_capture():
    """Full acceptance on the registered spyre stack (codegen-only, offline).

    Capture asserts run FIRST and are inspectable while the backend
    raises: 0 graph breaks, no failure outside the known op-coverage
    blocker classes, no silent eager fallback. A capture regression
    (e.g. a graph break reintroduced into the monkey patches) FAILS this
    test even while the op-coverage blockers still fire — it cannot hide
    behind the blocker xfail.

    The subprocess classifies outcomes and exits:
      0  full acceptance (all calls launch-refused; unique_graphs is then
         observable and asserted == 2 with exactly 2 backend compiles)
      3  capture clean, but spyre op-coverage blockers still live
         (prefill UnsupportedPointwiseOpError / decode KeyError 'YBLOCK')
      1  capture regression or unexpected failure class
    Exit 3 maps to pytest.xfail; exit 0 FAILS the test with instructions
    to promote it (blockers lifted); anything else FAILS with stderr.
    """
    pytest.importorskip("transformers")
    result = _run_script(
        """
        import torch
        import torch_spyre

        torch_spyre._autoload()
        assert torch._dynamo.config.cache_size_limit == 1024

        import torch._dynamo.utils as dynamo_utils
        import torch._inductor.compile_fx as cfx

        reentries = 0
        _orig = cfx.compile_fx

        def _counting(gm, example_inputs, *args, **kwargs):
            global reentries
            reentries += 1
            return _orig(gm, example_inputs, *args, **kwargs)

        cfx.compile_fx = _counting

        from transformers.cache_utils import StaticCache, StaticLayer
        from transformers.models.llama.modeling_llama import (
            LlamaConfig,
            LlamaDecoderLayer,
        )

        # Op-coverage blockers (spyre codegen, NOT the cache seam):
        BLOCKERS = (
            "UnsupportedPointwiseOpError",  # prefill: SFP pointwise
            "KeyError: 'YBLOCK'",           # decode: triton heuristics
            # c2 structured-production envelope (T-SR-29): both graphs
            # can also be refused at compile with typed refusal codes
            # (fused DAG / constant non-metadata user / unsupported op)
            "StructuredProductionSubsetRefusedError",
            # RMSNorm sum-of-squares fused reduction kernel routed to
            # the default GPU triton path ("Could not find an active
            # GPU backend") — triton-fork/Inductor routing seam
            "Could not find an active GPU backend",
        )

        dynamo_utils.counters.clear()

        cfg = LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            vocab_size=128,
            _attn_implementation="sdpa",
        )
        device = torch.device("spyre")
        layer = LlamaDecoderLayer(cfg, 0).eval().to(device)
        cache = StaticCache(
            config=cfg, max_batch_size=1, max_cache_len=128, device=device
        )

        # 1p0 host-boundary law: i64 is metadata-only; position bookkeeping
        # stays int32 (the production mechanism is a thin spyre Cache
        # subclass; the probe subclasses per-layer to prove the seam).

        class Int32StaticLayer(StaticLayer):
            def update(self, key_states, value_states, *args, **kwargs):
                if not self.is_initialized:
                    self.lazy_initialization(key_states, value_states)
                kv_length = key_states.shape[-2]
                cache_position = (
                    torch.arange(
                        kv_length, device=self.device, dtype=torch.int32
                    )
                    + self.cumulative_length
                )
                self.cumulative_length.add_(kv_length)
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
                return self.keys, self.values

        for layer_cache in cache.layers:
            layer_cache.__class__ = Int32StaticLayer
            layer_cache.cumulative_length = layer_cache.cumulative_length.to(
                torch.int32
            )

        def block_forward(hidden_states, position_ids, attn_mask, cache):
            seq_len = hidden_states.shape[1]
            rope = (
                torch.zeros(1, seq_len, 32, device=hidden_states.device),
                torch.ones(1, seq_len, 32, device=hidden_states.device),
            )
            out = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                position_embeddings=rope,
            )
            return out[0]

        fn = torch.compile(block_forward, dynamic=False)

        def prefill(mask_len):
            hidden = torch.randn(1, 64, 128)
            positions = torch.arange(64).unsqueeze(0)
            causal = torch.triu(
                torch.full((64, 64), float("-inf")), diagonal=1
            ).fill_diagonal_(0.0)
            if mask_len == 64:
                mask = causal.unsqueeze(0).unsqueeze(0)
            else:
                mask = torch.full((1, 1, 64, mask_len), float("-inf"))
                mask[..., :64] = causal
            return (
                hidden.to(device),
                positions.to(device),
                mask.to(device),
            )

        def decode(step, mask_len):
            hidden = torch.randn(1, 1, 128)
            positions = torch.tensor([[64 + step]])
            written = 64 + step + 1
            mask = torch.zeros(1, 1, 1, mask_len)
            mask[..., written:] = float("-inf")
            return (
                hidden.to(device),
                positions.to(device),
                mask.to(device),
            )

        outcomes = []
        for args in [prefill(128)] + [decode(s, 128) for s in range(12)]:
            try:
                fn(*args, cache)
                outcomes.append("ok")
            except Exception as exc:
                text = f"{type(exc).__name__}: {exc}"
                if "codegen-only" in text:
                    outcomes.append("refused")
                elif any(sig in text for sig in BLOCKERS):
                    outcomes.append("blocker")
                else:
                    outcomes.append(
                        "FAILED " + text.splitlines()[0][:200]
                    )

        # ── Capture asserts FIRST (inspectable while compile fails) ──
        graph_breaks = dict(dynamo_utils.counters["graph_break"])
        assert not graph_breaks, graph_breaks
        failed = [o for o in outcomes if o.startswith("FAILED")]
        assert not failed, failed
        # No silent eager fallback: every call compiled (re-entry) or
        # reached the codegen-only launch refusal.
        assert reentries + outcomes.count("refused") >= len(outcomes), (
            reentries,
            outcomes,
        )

        if outcomes.count("refused") == len(outcomes):
            # Full acceptance: backend returned for every call, so
            # unique_graphs is observable — exactly prefill + decode,
            # compiled once each.
            stats = dict(dynamo_utils.counters["stats"])
            assert stats.get("unique_graphs", 0) == 2, stats
            assert reentries == 2, reentries
            raise SystemExit(0)
        raise SystemExit(3)
        """
    )
    if result.returncode == 0:
        pytest.fail(
            "full spyre-path acceptance passed — the op-coverage "
            "blockers are lifted; promote the unique_graphs/reentry "
            "assertions to unconditional and drop the exit-3 branch"
        )
    if result.returncode == 3:
        pytest.xfail(
            "spyre op-coverage blockers (prefill: 'pointwise_unsupported' "
            "has no SFP binary; decode: KeyError 'YBLOCK') — the capture "
            "asserts (no graph breaks, no out-of-class failures, no eager "
            "fallback) already passed before this xfail"
        )
    pytest.fail(
        f"capture regression or unexpected failure (exit "
        f"{result.returncode}): {result.stderr}"
    )
