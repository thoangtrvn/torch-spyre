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

"""Random numbers on the AIU — the short version.

Run with the device path switched on, from an installed torch-spyre:

    TORCH_SPYRE_DEVICE_RNG=1 python examples/rng.py

Running it from this source tree instead needs BOTH trees on PYTHONPATH — the
torch-spyre tree so the aten kernels register from source, and the codegen tree
so the backend registers the ``rng_kat`` pattern the direct draw launches
(verified device-clean 2026-11-21, exit 0):

    PYTHONPATH=<this-tree>:<spyre-knowledgebase>/codegen \
      TORCH_SPYRE_DEVICE_RNG=1 python examples/rng.py

What you get
------------
``torch.spyre.manual_seed(seed)`` seeds the *device* generator. A whole-tile
float32 fill on a spyre tensor is then produced by the AIU: one
Threefry2x32-20 kernel launch generates 64 words per tile, and one launch may
carry 1..63 tiles (up to 4032 values). The same seed reproduces the same
values; successive draws advance the generator's offset.

Honest bounds (doc 54) — do not read past them
----------------------------------------------
* The on-device direct bind needs: float32, contiguous, the identity map
  ``[0, 1)``, and a device layout of exactly ``(2n, 32)`` for ``n`` tiles.
  ``n`` is ``numel // 64`` for a whole number of 64-word tiles (1..63), or
  ``ceil(numel / 64)`` for a TAIL fill (``numel > 64``, not a multiple of 64)
  whose stick-padded allocation already covers the tile write — the tile loop
  writes whole tiles, so the slots past ``numel`` are padding outside the
  logical view. Then the values never round-trip through the host.
* Everything else — ``numel`` 1..63, non-fp32, an affine ``(from_, to)``, a
  non-``(2n, 32)`` device layout — takes the unchanged host-map path (device
  bits mapped to floats on the host). That is **not** a refusal and must not
  be turned into one.
* Typed refusals: a ``numel > 64`` tail whose allocation does not cover the
  tile write (a stick-only-padded tail, or a slice into a larger buffer), more
  than 63 tiles, any per-call ``generator=<obj>`` (the device path uses the
  single per-device default generator — multi-stream is unbuilt), and any use
  without the ``rng_kat`` pattern registered.
* There is no device ``normal_``, so ``torch.randn(..., device="spyre")``
  refuses with the switch on and host-offloads with it off.
"""

import torch

import torch_spyre  # noqa: F401  - registers the "spyre" device
import torch_spyre.ops.eager  # noqa: F401  - registers the aten kernels on spyre

# The explicit ops.eager import matters when this tree is imported from source
# (PYTHONPATH) rather than as an installed distribution: torch's device-backend
# autoload resolves the INSTALLED torch_spyre, so the source tree's aten
# kernels would otherwise never be registered and torch.rand would raise
# "Could not run 'aten::uniform_' with arguments from the 'spyre' backend".
from torch_spyre.rng import SpyreRngError, device_rng_enabled

DEVICE = "spyre"
SEED = 12345


def main() -> None:
    if not device_rng_enabled():
        raise SystemExit(
            "Device RNG is off. Run with TORCH_SPYRE_DEVICE_RNG=1 to exercise "
            "the device path; with the switch off, uniform_() is filled by the "
            "host CPU and copied down."
        )

    torch.spyre.manual_seed(SEED)  # type: ignore[attr-defined]

    # The compact form: torch.rand is empty()+uniform_(0,1) fused by the
    # dispatcher, so it lands in the same spyre uniform_ kernel
    # (torch_spyre/ops/eager.py:189).
    # One tile: 64 uniform float32 values, generated on the AIU.
    a = torch.rand(64, device=DEVICE)

    # Sixteen tiles in the SAME launch: 1024 values.
    b = torch.rand(1024, device=DEVICE)

    # An affine range keeps the explicit fill; the identity map above is the
    # only range the device writes directly, so this one takes the host-map
    # fallback (device bits, host float map — not a refusal).
    d = torch.empty(64, dtype=torch.float32, device=DEVICE).uniform_(-1.0, 1.0)

    ac, bc, dc = a.cpu(), b.cpu(), d.cpu()
    assert bool(((ac >= 0.0) & (ac < 1.0)).all()), "uniform_ out of [0, 1)"
    assert bool(((bc >= 0.0) & (bc < 1.0)).all()), "uniform_ out of [0, 1)"
    assert bool(((dc >= -1.0) & (dc < 1.0)).all()), "uniform_ out of [-1, 1)"

    # The same seed reproduces the same values.
    torch.spyre.manual_seed(SEED)  # type: ignore[attr-defined]
    c = torch.rand(64, device=DEVICE)
    assert torch.equal(ac, c.cpu()), "same seed must reproduce the same values"

    print(f"1 tile   (64 values, torch.rand):         {ac[:8].tolist()}")
    print(f"16 tiles (1024 values, torch.rand):       {bc[:8].tolist()}")
    print(f"affine [-1, 1) (host-map fallback):       {dc[:8].tolist()}")

    # A TAIL fill (not a whole number of 64-word tiles) draws on device when the
    # caller's stick-padded allocation covers the tile write: torch.rand(100)
    # allocates (4,32) and the 2-tile loop writes 4 sticks, so it is direct.
    # Exact values are NOT comparable to torch.rand(128)[:100]: the tail draw
    # reserves a different generator offset than a 128-word draw would.
    tail = torch.rand(100, device=DEVICE)
    assert tuple(tail.shape) == (100,), tail.shape
    tv = tail.cpu().tolist()
    assert all(0.0 <= v < 1.0 for v in tv), "tail values must lie in [0, 1)"
    print(f"tail     (100 values, torch.rand):        {tv[:8]}")

    # A tail whose allocation is one stick SHORT of the tile write refuses,
    # typed: torch.rand(96) allocates (3,32) but ceil(96/64) = 2 tiles = 4 sticks.
    try:
        torch.rand(96, device=DEVICE)
    except SpyreRngError as exc:
        print(f"refused, as documented: {exc}")
    else:
        raise AssertionError("numel=96 on a (3,32) allocation must refuse typed")


if __name__ == "__main__":
    # No catch-all here on purpose: SpyreRngError IS a NotImplementedError
    # (torch_spyre/rng.py:133), so a broad `except NotImplementedError` would
    # swallow a typed refusal and relabel it. Let the real traceback out.
    main()
