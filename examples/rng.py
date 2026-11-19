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

"""Truthful RNG example — what actually works on a Spyre device TODAY.

Verified against the tree on 2026-11-18. There is NO device (AIU) RNG engine
wired into torch-spyre yet. The registered ``aten::uniform_`` / ``aten::normal_``
"spyre" kernels fill a **CPU** tensor with the host generator and copy it to the
device (``torch_spyre/ops/eager.py`` — ``spyre__uniform_`` / ``spyre__normal_``;
twins in ``torch_spyre/codegen_ops.py``). So the values are correct and land on
the device, but they are generated on the HOST CPU.

Consequences for callers:

* Seed with ``torch.manual_seed(...)`` — the global **CPU** seed — because that
  is the generator the offload actually uses.
* ``torch.spyre.manual_seed(...)`` is a **silent no-op** today (no C++ binding;
  ``torch_spyre/__init__.py:386`` only calls it ``if`` it exists). Do NOT rely on
  it to make device draws reproducible. The defect is pinned by
  ``tests/test_rng_device_offload.py``.
* The silicon-witnessed device Threefry path is not reachable from torch
  (``rng_kat`` is unregistered). When registration (T-RN-10) lands, this example
  should be updated to exercise the real device RNG.

Run:  python examples/rng.py
"""

import torch
import torch_spyre  # noqa: F401  (registers the "spyre" device + kernels)

torch.manual_seed(0xAFFE)  # the seed that actually controls the RNG today (host CPU)

# In-place random fill on a device tensor. Generated on the CPU, copied over.
u = torch.empty(16, dtype=torch.float32, device="spyre").uniform_(0.0, 1.0)
n = torch.empty(16, dtype=torch.float32, device="spyre").normal_(0.0, 1.0)

print("uniform_(0,1) on spyre  [host-generated, copied to device]:")
print(u.cpu())
print("normal_(0,1) on spyre   [host-generated, copied to device]:")
print(n.cpu())

uc = u.cpu()
assert bool(((uc >= 0.0) & (uc < 1.0)).all()), "uniform_ values out of [0,1)"
print("ok — values valid (but produced on the host CPU, not the AIU)")
