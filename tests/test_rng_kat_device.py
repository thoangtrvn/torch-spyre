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

"""ON-DEVICE T=1 RNG KAT — the 3 published Random123 2x32-20 anchors on silicon.

Device-only: SKIPS unless a real ``spyre`` device is present AND ``rng_kat`` is
registered in the box's ``sentient_codegen`` (i.e. the codegen tree is at a
staging tip carrying the rng-kat registration). Run on the box with that
codegen on ``PYTHONPATH``.

This is the END-TO-END device proof, two ways:
* ``test_published_anchor_bit_exact_on_device`` — builds each anchor's EXPLICIT
  ``(counter, key)`` params (NOT a seed-derived stream), launches
  ``launch_raw_state_draw`` on the real device, reconstructs ``x0/x1`` from the
  raw egress, and compares to the frozen vectors. This is the device analogue of
  the offline emulator replay and is the strong witness.
* ``test_draw_raw_state_is_reproducible_on_device`` — the seed→stream smoke:
  ``draw_raw_state(seed)`` twice is identical (exercises the whole wired path
  and asserts params reach LX: a params swizzle shows as wrong/zero values).

Transport caveat (why the anchor test matters): ``stream_to_sticks`` is recipe
order; on silicon a multi-stick fp32 params tensor past the first stick may need
device word-order, which the launcher does not model. If the anchor test fails
with garbage/zeros past stick 0 while the emulator replay passed, that is the
transport swizzle, not the arithmetic.

Relationship to the codegen KAT (the two roles are DISTINCT):
  * The CANONICAL device *kernel* witness is the codegen slice-6 raw-state KAT
    in spyre-knowledgebase (receipt silicon-71dcf6e2ef41; binary e697b657; the
    3 published anchors bit-exact). THAT is what proves the SFP kernel.
  * THIS file is the torch-spyre *framework-integration* regression: it drives
    torch_spyre's own launch path (RngLaunchPlan ABI -> launch_raw_state_draw ->
    _C.launch_kernel_from_bytes -> egress reconstruct) and checks it reproduces
    the same bits. It lives in torch-spyre because the framework depends on the
    backend, never the reverse — a test that imports torch_spyre cannot live in
    the codegen repo without inverting that dependency.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
hr = pytest.importorskip("torch_spyre.rng")

_VECTORS = json.loads(
    (Path(__file__).parent / "data" / "rng_kat_raw_state_2x32_20.json").read_text()
)["vectors"]


def _runtime_available() -> bool:
    """True only with the REAL spyre runtime, not codegen-only mode (where
    device tensors allocate but ``launch_kernel_from_bytes`` refuses)."""
    try:
        from torch_spyre._runtime import codegen_only_available

        return not codegen_only_available()
    except Exception:
        return False


def _rng_kat_registered() -> bool:
    try:
        from sentient_codegen.scheduler.matmul import PATTERN_REGISTRY

        return "rng_kat" in PATTERN_REGISTRY
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_runtime_available() and _rng_kat_registered()),
    reason="needs a spyre device AND rng_kat registered in the box's "
    "sentient_codegen (codegen tree at a staging tip)",
)


def _anchor_plan(vector: dict) -> "hr.RngLaunchPlan":
    """Build the launch plan for one anchor's EXPLICIT (counter, key) params."""
    c0, c1 = vector["counter"]
    k0, k1 = vector["key"]
    blocks = np.arange(hr.RNG_KAT_LANES, dtype=np.uint32)
    stream = hr.build_band_lxlu_stream(
        blocks,
        x0_base=(c0 + k0) & 0xFFFFFFFF,
        x1=(c1 + k1) & 0xFFFFFFFF,
        rotations=[hr.ROTATIONS_2X32[r % 8] for r in range(hr.RNG_KAT_ROUNDS)],
        injections=hr.injection_words_2x32((k0, k1), hr.RNG_KAT_ROUNDS),
        rounds=hr.RNG_KAT_ROUNDS,
        egress=hr.EgressMode.RAW_STATE,
    )
    params = hr.stream_to_sticks(stream)
    return hr.RngLaunchPlan(
        params=params,
        egress=hr.EgressMode.RAW_STATE,
        sticks=int(params.shape[0]),
        lanes=int(params.shape[1]),
        rounds=hr.RNG_KAT_ROUNDS,
        xlat_segments={"input": 0, "output": 1},
        declared_segment_count=2,
        output_sticks=hr.band_egress_stick_count(
            hr.RNG_KAT_ROUNDS, egress=hr.EgressMode.RAW_STATE
        ),
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_published_anchor_bit_exact_on_device(index):
    vector = _VECTORS[index]
    out = hr.launch_raw_state_draw(_anchor_plan(vector))
    x0, x1 = hr.reconstruct_raw_state_from_tensor(out)
    assert [int(v) for v in x0] == [int(v) for v in vector["expected_x0"]], (
        f"anchor {index} x0 mismatch on device"
    )
    assert [int(v) for v in x1] == [int(v) for v in vector["expected_x1"]], (
        f"anchor {index} x1 mismatch on device"
    )


def test_draw_raw_state_is_reproducible_on_device():
    a0, a1 = hr.draw_raw_state(12345, 0)
    b0, b1 = hr.draw_raw_state(12345, 0)
    assert (np.asarray(a0) == np.asarray(b0)).all()
    assert (np.asarray(a1) == np.asarray(b1)).all()


def test_resident_device_tensor_equals_full_upload_after_n_draws():
    """Device residency on silicon: N draws off the resident params tensor
    produce identical output to the full-upload path, and the resident device
    buffer is byte-identical to a fresh full upload at the final offset.

    This exercises the REAL device bridge (``torch.Tensor`` on spyre) — the
    offline accounting/byte-identity gates live in ``test_rng_residency.py``.
    """
    dev = torch.device("spyre")
    hr.reset_rng_resources()
    res = hr.get_rng_resources()
    seed = 20241119

    plan = None
    for i in range(8):
        plan = hr.build_rng_launch_plan(
            seed=seed, offset=i * 64, egress=hr.EgressMode.RAW_STATE
        )
        resident_out = hr.launch_raw_state_draw(plan, device=dev)
        full_out = hr.launch_raw_state_draw(plan, device=dev, use_resident=False)
        assert torch.equal(resident_out.cpu(), full_out.cpu()), (
            f"resident vs full-upload mismatch at draw {i}"
        )

    # The resident device tensor must equal a fresh full upload of the same
    # (seed, offset) — no stale rows carried across the offset sweep.
    resident = res.ensure_device_params(plan.spec, dev)
    fresh = hr._TorchDeviceBridge().upload(plan.params, dev)
    assert torch.equal(resident.cpu(), fresh.cpu())


def test_resident_offset_advance_writes_three_rows_on_device():
    """On silicon, the device-side update after the first draw is a 3-row write.

    Asserts the accounting seam directly against the real bridge: instrument
    ``write_rows``/``upload`` on a real-device bridge and count the rows moved
    across one offset advance.
    """
    dev = torch.device("spyre")
    hr.reset_rng_resources()
    res = hr.get_rng_resources()
    seed = 424242

    class _CountingTorchBridge(hr._TorchDeviceBridge):
        def __init__(self):
            self.upload_rows = 0
            self.written_rows = 0

        def upload(self, host_words, device):
            self.upload_rows += int(host_words.shape[0])
            return super().upload(host_words, device)

        def write_rows(self, resident, row_indices, host_rows):
            self.written_rows += len(tuple(row_indices))
            return super().write_rows(resident, row_indices, host_rows)

    bridge = _CountingTorchBridge()
    res.ensure_device_params(
        hr.RngDrawSpec.from_seed_offset(seed, 0), dev, bridge=bridge
    )
    assert bridge.upload_rows == hr.band_lxlu_pop_count(
        hr.RNG_KAT_ROUNDS, egress=hr.EgressMode.RAW_STATE
    ), "first draw uploads the full params tensor"
    assert bridge.written_rows == 0

    res.ensure_device_params(
        hr.RngDrawSpec.from_seed_offset(seed, 64), dev, bridge=bridge
    )
    assert bridge.written_rows == 3, "an offset advance writes only 3 rows on device"
    assert bridge.upload_rows == hr.band_lxlu_pop_count(
        hr.RNG_KAT_ROUNDS, egress=hr.EgressMode.RAW_STATE
    ), "the full params tensor is NOT re-uploaded on an offset advance"
