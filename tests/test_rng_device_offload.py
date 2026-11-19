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

"""Characterization + defect pin: Spyre "RNG" today is HOST-CPU offload.

What the tree does on the DEFAULT path (device RNG switch OFF; characterized
2026-11-18, tree ``152c8848``, still the default after the Path A landing):

* The registered ``aten::uniform_`` / ``aten::normal_`` "spyre" kernels fill a
  **CPU** tensor with the host generator and then ``self.copy_(cpu_tmp)`` to the
  device — see ``torch_spyre/ops/eager.py`` (``spyre__normal_``, ``spyre__uniform_``)
  and the twin registrations in ``torch_spyre/codegen_ops.py``. With the switch
  off, **no AIU RNG engine is used**: the values are correct and land on the
  device, but they are produced on the host CPU.
* NOTE: Path A adds a **device** RNG path behind the ``DeviceRngSwitch`` env
  (`uniform_` fills from device Threefry bits at T=1). That path is exercised by
  ``tests/test_rng_uniform_device.py`` and ``tests/test_rng_kat_device.py`` — NOT
  here. This file pins the switch-off (default) behaviour.
* ``torch.spyre.manual_seed(seed)`` is a **silent no-op**:
  ``torch_spyre/__init__.py:386`` only calls ``self._C.manual_seed`` *if* it
  exists, and there is no C++ ``manual_seed`` binding in the tree — so the call
  returns without effect. The seed that actually steers the offloaded values is
  the **global CPU** seed set by ``torch.manual_seed``.

The ``test_spyre_manual_seed_steers_device_rng`` case is an **XFAIL defect pin**:
it asserts the DESIRED behaviour (torch.spyre.manual_seed reproducibly steers the
device stream). It fails today because the seed path is inert. When the seed is
wired to the offload (or a real device RNG lands), it flips to XPASS — remove the
xfail marker then.

This file asserts only the switch-off default. It is NOT a capability claim and
does NOT exercise the silicon-witnessed device Threefry path (that path lives in
test_rng_uniform_device.py / test_rng_kat_device.py).
"""

import pytest
import torch

torch_spyre = pytest.importorskip("torch_spyre")


def _runtime_available() -> bool:
    """True only with the REAL spyre runtime, not codegen-only mode (where a
    device tensor allocates but a fill raises NotImplementedError)."""
    try:
        from torch_spyre._runtime import codegen_only_available

        return not codegen_only_available()
    except Exception:
        return False


def _spyre_available() -> bool:
    try:
        torch.zeros(1, dtype=torch.float32, device="spyre")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_runtime_available() and _spyre_available()),
    reason="no real spyre runtime (codegen-only allocates but cannot fill a "
    "device tensor)",
)

_SEED = 0xC0FFEE


def _device_uniform(n: int = 64) -> torch.Tensor:
    return torch.empty(n, dtype=torch.float32, device="spyre").uniform_(0.0, 1.0)


class TestDeviceRandomIsCpuOffload:
    """Characterization: the device fill produces valid values (host-generated)."""

    def test_uniform_fill_is_in_range(self):
        torch.manual_seed(_SEED)
        x = _device_uniform()
        assert x.device.type == "spyre"
        vals = x.cpu()
        assert torch.all(vals >= 0.0) and torch.all(vals < 1.0)

    def test_normal_fill_is_finite(self):
        torch.manual_seed(_SEED)
        y = torch.empty(64, dtype=torch.float32, device="spyre").normal_(0.0, 1.0)
        assert torch.isfinite(y.cpu()).all()

    def test_global_cpu_seed_reproducibly_drives_the_device_fill(self):
        """The ACTUAL control path is the global CPU seed: two runs under the
        same ``torch.manual_seed(S)`` produce identical device values, because
        the fill is done on the CPU with the default CPU generator. If this ever
        fails, the RNG path changed and this file's premise needs revisiting."""
        torch.manual_seed(_SEED)
        a = _device_uniform().cpu()
        torch.manual_seed(_SEED)
        b = _device_uniform().cpu()
        assert torch.equal(a, b), (
            "device fill is expected to be host-CPU offload and reproducible "
            "from the global CPU seed; the RNG path changed"
        )


class TestSpyreManualSeedDefect:
    """Pins the defect: torch.spyre.manual_seed does not steer the device RNG."""

    @pytest.mark.xfail(
        strict=False,
        reason="DEFECT: torch.spyre.manual_seed is a silent no-op (no C++ "
        "binding; torch_spyre/__init__.py:386). Device RNG is not seeded "
        "by it. Remove this xfail when the seed path is wired.",
    )
    def test_spyre_manual_seed_steers_device_rng(self):
        """DESIRED behaviour, asserted. Fails today because the seed is inert:
        reseeding with torch.spyre.manual_seed(S) does NOT reset the stream, so
        the two blocks differ."""
        torch.manual_seed(_SEED)  # deterministic starting CPU-gen state
        torch.spyre.manual_seed(1234)  # DEFECT: no-op today
        a = _device_uniform().cpu()
        torch.spyre.manual_seed(1234)  # DEFECT: no-op today -> does NOT reset
        b = _device_uniform().cpu()
        assert torch.equal(a, b), (
            "torch.spyre.manual_seed should reproducibly steer the device RNG; "
            "today it is a no-op, so the second block differs"
        )
