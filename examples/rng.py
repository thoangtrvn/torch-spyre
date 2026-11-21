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

Run with the device path switched on:

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
  ``[0, 1)``, and a whole number of 64-word tiles (1..63) whose device layout
  is ``(2n, 32)``. Then the values never round-trip through the host.
* Everything else — ``numel`` 1..63, non-fp32, an affine ``(from_, to)``, a
  non-``(2n, 32)`` device layout — takes the unchanged host-map path (device
  bits mapped to floats on the host). That is **not** a refusal and must not
  be turned into one.
* Typed refusals: ``numel > 64`` that is not a whole number of tiles, more
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

    # A fill that is not a whole number of 64-word tiles refuses, typed.
    try:
        torch.rand(100, device=DEVICE)
    except SpyreRngError as exc:
        print(f"refused, as documented: {exc}")
    else:
        raise AssertionError("numel=100 must refuse typed, not draw")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError:
        raise SystemExit(
            "No spyre runtime in this environment (the spyre kernels are not "
            "registered). Run this on the AIU box."
        ) from None
