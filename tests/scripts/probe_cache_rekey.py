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

"""T-SR-41: warm-compile -> flip -> recompile-or-stale-serve measurement.

Measures whether each flip axis survives the two cache layers above the
backend (the T-SB-23 one-window law: the measurement IS the deliverable):

  1. FX-graph cache (Inductor) — keyed without any Spyre knob input?
     SpyreHeuristics.uuid() is the CONSTANT "torch_spyre...v1"
     (choices.py:27-28), so a warm FX hit could serve an artifact
     compiled under the other setting.
  2. Triton cache — key includes backend_options.hash() (num_cores and
     ttir_parse_backend ride SpyreOptions -> splits), but env_vars hashes
     only TRITON_* (SENTIENT_AUTOTUNE_EFFORT is flip-blind there).

Axes measured (each: warm compile under setting A, then the SAME cache
dirs under setting B; detection via a SpyreBackend._make_spyre_ir
capture — reached-and-carries-B = recompiled; not-reached-and-succeeded
= STALE SERVE at a cache layer above the backend):

  - parse_flip:   ttir_parse_backend None -> "structured"
  - sencores_flip: SENCORES 8 -> 4
  - effort_flip:   SENTIENT_AUTOTUNE_EFFORT unset -> "full"

Usage (run from the torch-spyre repo root):
  PYTHONPATH=<kb>/codegen:<kb>/codegen/tests:<triton-spyre>/python \
  python tests/scripts/probe_cache_rekey.py [--root DIR]

Prints one verdict line per axis; --root controls where the shared
per-axis cache dirs are created (default: a fresh tempfile root).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timezone

# The inner compile script. Prints a machine-readable RESULT line:
#   captured=<0|1> num_cores=<int> parse_backend=<str> refusal=<substr>
# `captured=1` means _make_spyre_ir ran (a compile happened this run);
# the options values it saw are printed so the flip's arrival is
# verifiable, not inferred.
_INNER = """
import os
import torch

capture = os.environ.get("TS41_CAPTURE") == "1"
captured = []

if capture:
    from triton.backends.spyre.compiler import SpyreBackend

    def capture_spyre_ir(mod, metadata, options):
        captured.append(options)
        raise RuntimeError('tsr41-captured')

    SpyreBackend._make_spyre_ir = staticmethod(capture_spyre_ir)
else:
    # Warm phase: no-op the launch seam so the compile SUCCEEDS and the
    # FX cache stores a real entry (codegen-only env has no C extension;
    # the driver imports launch_kernel_from_bytes lazily from
    # torch_spyre._C at driver.py:449, so the module attribute patch
    # holds). Compilation and FX-cache write happen before launch.
    import torch_spyre._C

    torch_spyre._C.launch_kernel_from_bytes = lambda *a, **k: None

{knob_setup}

@torch.compile
def add(x):
    return x + 1

x = torch.empty((512, 128), dtype=torch.float16).to('spyre')
outcome = 'ok'
try:
    add(x)
except Exception as e:
    outcome = type(e).__name__ + ':' + str(e)[:120]

if captured:
    o = captured[0]
    print(f"RESULT captured=1 num_cores={o.num_cores} "
          f"parse_backend={o.ttir_parse_backend} outcome={outcome}")
else:
    print(f"RESULT captured=0 num_cores=-1 parse_backend=unobserved "
          f"outcome={outcome}")
"""


def _run_inner(cache_root: str, axis: str, phase: str, *, sencores: str | None,
               effort: str | None, knob: str | None, capture: bool) -> dict:
    env = os.environ.copy()
    env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    if capture:
        # Only the FLIP phase captures: the warm phase must compile
        # successfully so the FX cache actually stores an entry (a
        # failed compile is not cached — measuring against it would be
        # the confound this probe's first run exposed).
        env["TS41_CAPTURE"] = "1"
    else:
        env.pop("TS41_CAPTURE", None)
    # Shared across A and B of the SAME axis (that is the measurement);
    # fresh per axis (axes must not contaminate each other — the T-SR-40
    # R2 lesson).
    for var in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                "SENTIENT_CACHE_DIR"):
        env[var] = os.path.join(cache_root, f"{axis}_{var.lower()}")
        os.makedirs(env[var], exist_ok=True)
    env.pop("SENCORES", None)
    env.pop("SENTIENT_AUTOTUNE_EFFORT", None)
    env.pop("SPYRE_INJECT_BINARY", None)
    env.pop("SPYRE_INJECT_BINARY_PATH", None)
    env.pop("SPYRE_INJECT_BINARY_ACK", None)
    if sencores is not None:
        env["SENCORES"] = sencores
    if effort is not None:
        env["SENTIENT_AUTOTUNE_EFFORT"] = effort

    knob_setup = ""
    if knob is not None:
        knob_setup = (
            "from torch_spyre._inductor import config as spyre_config\n"
            f"spyre_config.ttir_parse_backend = {knob!r}\n"
        )
    script = _INNER.replace("{knob_setup}", knob_setup)

    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    result = {"phase": phase, "rc": proc.returncode}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            body = line[len("RESULT "):]
            # captured/num_cores/parse_backend are space-free tokens;
            # outcome is the free-text tail (may contain spaces).
            for key in ("captured", "num_cores", "parse_backend"):
                marker = f"{key}="
                start = body.find(marker)
                if start >= 0:
                    end = body.find(" ", start)
                    result[key] = (
                        body[start + len(marker):]
                        if end < 0 else body[start + len(marker):end]
                    )
            marker = "outcome="
            start = body.find(marker)
            if start >= 0:
                result["outcome"] = body[start + len(marker):]
    if "captured" not in result:
        result["captured"] = "?"
        result["raw_stderr_tail"] = proc.stderr[-400:]
    return result


def _measure(cache_root: str, axis: str, warm: dict, flip: dict) -> str:
    """Run the warm->flip pair; return the verdict line."""
    a = _run_inner(cache_root, axis, "warm", capture=False, **warm)
    b = _run_inner(cache_root, axis, "flip", capture=True, **flip)

    if a.get("captured") != "0" or a.get("outcome") != "ok":
        # The warm compile must SUCCEED without capture, or there is no
        # cached artifact to measure against.
        verdict = "INVALID (warm compile did not succeed cleanly)"
        detail = str(a)
    elif b.get("captured") == "1":
        # A compile ran under the flipped setting. The flip's arrival is
        # proven by the captured options; report the values.
        verdict = "RECOMPILED (backend compile ran under the flip)"
        detail = (f"num_cores={b.get('num_cores')} "
                  f"parse_backend={b.get('parse_backend')} "
                  f"outcome={b.get('outcome')}")
    elif b.get("captured") == "0" and "SpyreRuntimeUnavailableError" in str(
        b.get("outcome")
    ):
        # No backend compile ran AND execution reached the launch seam
        # (the served wrapper replayed down to the no-op'd C call): a
        # genuine stale serve at a cache layer ABOVE the backend. NOTE:
        # captured=0 alone cannot distinguish an FX-graph hit from a
        # triton-layer hit — both skip _make_spyre_ir; the layer
        # attribution rests on the entry-count/key forensics, not the
        # probe verdict. A flip run that failed BEFORE compile (import
        # error, config refusal) must not be mislabeled stale.
        verdict = "STALE SERVE (cache-layer hit; backend compile never ran)"
        detail = f"outcome={b.get('outcome')}"
    else:
        verdict = "INCONCLUSIVE (inner script failed)"
        detail = str(b)
    return (f"[{axis}] warm: captured={a.get('captured')} "
            f"outcome={a.get('outcome')} | flip: {verdict} | {detail}")


AXES = {
    "parse_flip": (
        dict(sencores=None, effort=None, knob=None),
        dict(sencores=None, effort=None, knob="structured"),
    ),
    "sencores_flip": (
        dict(sencores="8", effort=None, knob=None),
        dict(sencores="4", effort=None, knob=None),
    ),
    "effort_flip": (
        dict(sencores=None, effort=None, knob=None),
        dict(sencores=None, effort="full", knob=None),
    ),
    # Mechanism probe: None -> "legacy" is SEMANTICALLY IDENTICAL (both
    # parse legacy). If this also recompiles, the cache key tracks the
    # torch_spyre config module state itself (any config-value change
    # rekeys), not the knob's downstream effect.
    "config_value_flip": (
        dict(sencores=None, effort=None, knob=None),
        dict(sencores=None, effort=None, knob="legacy"),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None,
                        help="cache root dir (default: fresh tempfile)")
    args = parser.parse_args()

    if args.root:
        root = args.root
        os.makedirs(root, exist_ok=True)
    else:
        root = tempfile.mkdtemp(prefix="tsr41_cache_")

    print(f"# T-SR-41 cache-rekey measurement "
          f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# cache root: {root}")
    print(f"# python: {sys.executable}")
    for axis, (warm, flip) in AXES.items():
        print(_measure(root, axis, warm, flip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
