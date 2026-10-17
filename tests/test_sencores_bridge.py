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

"""T-SR-39: SENCORES core-budget bridge tests.

Pins the bridge end to end:

1. ``core_budget.validated_sencores`` refuses loud on malformed and
   out-of-range values (the silent-fallback-to-32 defect class).
2. The e2e channel: a torch.compile pointwise compile under a
   non-default ``SENCORES`` lands ``config.sencores`` on
   ``SpyreOptions.num_cores`` at the fork's ``_make_spyre_ir``
   boundary — through the same serialized FixedTritonConfig channel as
   problem_m/n/k (the value is a function argument in the compile
   pipeline, so async_compile worker processes observe the scheduling
   process's budget; an env-read default_factory would not, and is the
   forbidden pattern this bridge replaces).
3. The embedding host's ``max_cores`` defaults to the validated budget
   (no hardcoded 32).
"""

import os
import subprocess
import sys
import textwrap

import pytest

from torch_spyre.core_budget import validated_sencores


# ---------------------------------------------------------------------------
# 1. Loud validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "-1", "33", "64", "abc", "", "3.5", None])
def test_validated_sencores_refuses_loud(bad):
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        validated_sencores(bad)


@pytest.mark.parametrize("good", ["1", "8", "32", 4, 32])
def test_validated_sencores_accepts_in_range(good):
    assert validated_sencores(good) == int(good)


def test_garbage_env_refuses_at_config_import():
    """A malformed SENCORES env value refuses at the config read — no
    silent 32 anywhere (the device-interface defect this repairs).
    ``import torch`` first: the torch_spyre._inductor package cannot be
    the first torch_spyre import (partially-initialized circular import
    through patches -> torch -> backend autoload)."""
    result = _run_script(
        """
        import torch
        from torch_spyre._inductor.config import sencores
        print(sencores)
        """,
        sencores="garbage",
    )
    assert result.returncode != 0
    assert "[1, 32]" in (result.stderr + result.stdout)


def test_device_properties_report_the_requested_budget():
    result = _run_script(
        """
        import torch
        from torch_spyre.device.interface import _detect_device_properties
        props = _detect_device_properties()
        assert props.multi_processor_count == 8, props
        """,
        sencores="8",
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# 2. The e2e channel: SENCORES -> config.sencores -> SpyreOptions.num_cores
# ---------------------------------------------------------------------------


def test_compile_lands_sencores_on_spyre_options_num_cores():
    """The bridge's terminal pin: under SENCORES=8, a real torch.compile
    pointwise compile carries num_cores=8 on the fork SpyreOptions at
    the _make_spyre_ir boundary — same capture pattern as the
    descriptor-route dimension pins in test_descriptor_pointwise_patch."""
    result = _run_script(
        """
        import torch
        from triton.backends.spyre.compiler import SpyreBackend

        captured = []

        class CapturedIR(RuntimeError):
            pass

        def capture_spyre_ir(mod, metadata, options):
            captured.append(options)
            raise CapturedIR('captured')

        SpyreBackend._make_spyre_ir = staticmethod(capture_spyre_ir)

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
        assert options.num_cores == 8, options.num_cores
        # problem dims ride the same channel (1D-grid pointwise
        # decomposes 512x128 -> M=ceil(65536/64)=1024, N=64) — pinned
        # only as channel-liveness, the axis under test is num_cores.
        assert options.problem_m == 1024, options.problem_m
        """,
        sencores="8",
    )
    assert result.returncode == 0, result.stderr


def test_default_sencores_leaves_num_cores_at_32():
    """Unset SENCORES keeps the field default — the bridge is
    behavior-neutral for every existing default-budget compile."""
    result = _run_script(
        """
        import torch
        from triton.backends.spyre.compiler import SpyreBackend

        captured = []

        def capture_spyre_ir(mod, metadata, options):
            captured.append(options)
            raise RuntimeError('captured')

        SpyreBackend._make_spyre_ir = staticmethod(capture_spyre_ir)

        @torch.compile
        def add(x):
            return x + 1

        x = torch.empty((512, 128), dtype=torch.float16).to('spyre')
        try:
            add(x)
        except Exception:
            if not captured:
                raise

        assert captured[0].num_cores == 32, captured[0].num_cores
        """,
        sencores=None,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# 3. Embedding host inherits the budget
# ---------------------------------------------------------------------------


def test_embedding_host_max_cores_defaults_to_sencores(monkeypatch):
    """build_embedding_launches(max_cores=None) resolves to the
    validated config budget, not a hardcoded 32. The planning math is
    exercised with a stubbed schedule to keep this a unit pin — the
    launch geometry itself is covered by the embedding op tests."""
    import sentient_codegen.scheduler.engine as engine
    import torch_spyre.ops._embedding_host as host
    from torch_spyre._inductor import config as spyre_config

    calls = {}

    def fake_schedule(op, tiling, target):
        calls["num_cores"] = tiling.num_cores
        raise RuntimeError("stop-after-tiling")

    monkeypatch.setattr(engine, "schedule", fake_schedule)
    monkeypatch.setattr(engine, "TilingConfig", lambda **kw: type("T", (), kw)())
    # Budget 1: this geometry (8 tokens, spt=2 -> K=4, ceil(N/K)=2)
    # requests 2 cores, so the budget BOUNDS the tiling — a regressed
    # hardcoded max_cores=32 would emit num_cores=2 and fail this pin;
    # a budget of 4 would be vacuous here (min(4,2) == min(32,2) == 2).
    monkeypatch.setattr(spyre_config, "sencores", 1)

    with pytest.raises(RuntimeError, match="stop-after-tiling"):
        host.build_embedding_launches(
            vocab=1024,
            d_model=128,
            element_bits=16,
            flat_idx=[0, 1, 2, 3, 4, 5, 6, 7],
            max_cores=None,
        )
    # The budget that bounded the tiling is the config read (1), not a
    # hardcoded 32 — the geometry requests 2, so only the budget can
    # produce 1.
    assert calls["num_cores"] == 1, calls["num_cores"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_script(script: str, *, sencores: str | None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    if sencores is None:
        env.pop("SENCORES", None)
    else:
        env["SENCORES"] = sencores
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
