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

"""T-SR-41: cache re-key measurement pins (warm-compile -> flip).

Pins the measured behavior of each flip axis across the disk cache
layers, per the T-SB-23 one-window law (the measurement is the
deliverable — these tests ARE the timestamped measurement, kept green):

  - parse_flip (ttir_parse_backend None -> "structured"): RECOMPILES.
    Mechanism (measured 2026-10-18, /tmp/tsr41_measure_root): the
    triton cache key's backend_options.hash() splits — T-SR-40 rides
    the knob on SpyreOptions. The FX-graph layer is not what splits it
    (a semantically identical None->"legacy" flip STALE-SERVES: same
    options value "legacy", same hash, one triton cache entry).
  - sencores_flip (SENCORES 8 -> 4): RECOMPILES. Same mechanism —
    SpyreOptions.num_cores (T-SR-39) is in backend_options.hash().
  - effort_flip (SENTIENT_AUTOTUNE_EFFORT unset -> "full"): RECOMPILES
    (re-keyed by T-SR-43, 2026-10-18). Formerly the measured T-SR-41
    defect: the env var is read inside codegen (autotuner/effort.py),
    rides no SpyreOptions field, and the triton cache env_vars component
    hashed only the C++ CACHE_INVALIDATING_ENV_VARS set — the flip
    stale-served the warm kernel. The fork now merges SENTIENT_* env
    vars (minus an observation-only neutral set) into get_cache_key.

Each test runs the warm phase WITHOUT capture (no-op launch seam so the
compile succeeds and the caches store real entries), then the flip
phase WITH a SpyreBackend._make_spyre_ir capture: reached-and-carries-
the-flip = recompiled; not-reached = stale serve.
"""

import importlib.util
import os
import tempfile

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _probe():
    """Import tests/scripts/probe_cache_rekey.py as a module (single
    source of truth for the inner compile script and env assembly)."""
    path = os.path.join(_TESTS_DIR, "scripts", "probe_cache_rekey.py")
    spec = importlib.util.spec_from_file_location("probe_cache_rekey", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def cache_root():
    root = tempfile.mkdtemp(prefix="tsr41_test_")
    yield root
    # Leave the root for post-mortem on failure; /tmp cleanup handles it.


def test_parse_backend_flip_recompiles_not_stale(cache_root):
    """The flip wave's own axis: a warm legacy compile followed by the
    structured opt-in must RECOMPILE (the flip reaches the backend),
    never silently serve the legacy-era kernel."""
    probe = _probe()
    warm, flip = probe.AXES["parse_flip"]
    a = probe._run_inner(cache_root, "parse_flip", "warm", capture=False, **warm)
    assert a.get("captured") == "0" and a.get("outcome") == "ok", a
    b = probe._run_inner(cache_root, "parse_flip", "flip", capture=True, **flip)
    assert b.get("captured") == "1", (
        f"parse flip STALE-SERVED (backend compile never ran): {b}"
    )
    assert b.get("parse_backend") == "structured", b


def test_sencores_flip_recompiles_not_stale(cache_root):
    """SENCORES 8 -> 4 across a warm cache must RECOMPILE with
    num_cores=4 (T-SR-39's options-hash rekey), never serve the
    8-core-era kernel."""
    probe = _probe()
    warm, flip = probe.AXES["sencores_flip"]
    a = probe._run_inner(cache_root, "sencores_flip", "warm", capture=False, **warm)
    assert a.get("captured") == "0" and a.get("outcome") == "ok", a
    b = probe._run_inner(cache_root, "sencores_flip", "flip", capture=True, **flip)
    assert b.get("captured") == "1", (
        f"SENCORES flip STALE-SERVED (backend compile never ran): {b}"
    )
    assert b.get("num_cores") == "4", b


def test_semantically_identical_config_flip_stale_serves(cache_root):
    """Control axis (the mechanism discriminator): None -> "legacy" is
    semantically identical, and the cache correctly SERVES the warm
    kernel (one triton entry, same options hash). This is why the
    parse/sencores recompiles are attributed to the options hash, not
    to cache-layer keying on config-module state."""
    probe = _probe()
    warm, flip = probe.AXES["config_value_flip"]
    a = probe._run_inner(cache_root, "config_value_flip", "warm", capture=False, **warm)
    assert a.get("captured") == "0" and a.get("outcome") == "ok", a
    b = probe._run_inner(cache_root, "config_value_flip", "flip", capture=True, **flip)
    assert b.get("captured") == "0", (
        f"semantically identical flip unexpectedly recompiled: {b}"
    )
    # Prove execution reached the launch seam (genuine cache-layer serve,
    # not a pre-compile failure mislabeled as stale — R1 MINOR 1).
    # Box-independent (R2 MINOR 1 refuted): the subprocess env pins
    # TORCH_SPYRE_CODEGEN_ONLY=1, and require_runtime raises
    # SpyreRuntimeUnavailableError on that env var ALONE
    # (_runtime.py:54-60, _CODEGEN_ONLY_ENV) — whether or not the full
    # runtime .so is installed on the box.
    assert "SpyreRuntimeUnavailableError" in str(b.get("outcome")), b


def test_sentient_effort_flip_rekeys(cache_root):
    """RE-KEYED (T-SR-43, 2026-10-18): flipping SENTIENT_AUTOTUNE_EFFORT
    across a warm cache now RECOMPILES — the backend compile runs
    (captured=1). Formerly the measured T-SR-41 defect: the env var rides
    no SpyreOptions field and the fork triton cache key hashed only the
    C++ CACHE_INVALIDATING_ENV_VARS set, so the flip stale-served. Fixed
    fork-side: triton/runtime/cache.py merges SENTIENT_* env vars (minus
    the observation-only neutral set) into get_cache_key, and the
    autotuner's check_disk_cache key does the same."""
    probe = _probe()
    warm, flip = probe.AXES["effort_flip"]
    a = probe._run_inner(cache_root, "effort_flip", "warm", capture=False, **warm)
    assert a.get("captured") == "0" and a.get("outcome") == "ok", a
    b = probe._run_inner(cache_root, "effort_flip", "flip", capture=True, **flip)
    assert b.get("captured") == "1", (
        "SENTIENT_AUTOTUNE_EFFORT flip STALE-SERVED (backend compile "
        f"never ran) — the T-SR-43 fork re-key regressed: {b}"
    )
    # captured=1 means the capture hook ran inside the backend compile —
    # the recompile itself is the proof the flip reached compilation (no
    # launch-seam reachability check applies on the recompile path).
