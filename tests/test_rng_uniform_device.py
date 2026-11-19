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

"""Device uniform fill (Path A): device-generated bits -> ``[from_, to)``.

Two tiers:

* OFFLINE (always run): the pure map ``u = (w>>8)*2**-24`` on known raw words,
  and the T>=2 typed refusal (which fires before any backend/device need).
* DEVICE-GATED (skipped unless a real spyre runtime is present AND ``rng_kat``
  is registered): the end-to-end ``uniform_`` fill on a device tensor — range,
  reproducibility, successive-draw divergence, agreement with the host map of
  ``draw_raw_state``, and the n=65 typed refusal.

The random BITS are device-generated (``draw_raw_state``); the map to
``[from_, to)`` runs on the host because spyre has no registered ``>>``/``&``
elementwise ops. The witnessed uniform *epilogue* (P5) remains future work.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
hr = pytest.importorskip("torch_spyre.rng")


# ---------------------------------------------------------------------------
# OFFLINE: the pure reconstruct+map.
# ---------------------------------------------------------------------------


def test_raw_words_to_uniform_known_words():
    words = np.array(
        [0x00000000, 0xFFFFFFFF, 0x000000FF, 0x00ABCDEF, 0x80000000],
        dtype=np.uint32,
    )
    out = hr.raw_words_to_uniform(words, 0.0, 1.0)
    # u = (w >> 8) * 2**-24
    expected = (words.astype(np.uint64) >> np.uint64(8)).astype(np.float64) * (2.0**-24)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=0, atol=1e-7)
    # Bounds: top word gives 1 - 2^-24, floor word gives 0.
    assert out[0] == 0.0
    assert out[1] < 1.0
    assert out[1] == np.float32(1.0 - 2.0**-24)


def test_raw_words_to_uniform_scales_to_range():
    words = np.array([0x00000000, 0xFFFFFFFF, 0x00800000], dtype=np.uint32)
    out = hr.raw_words_to_uniform(words, -2.0, 3.0)
    u = (words.astype(np.uint64) >> np.uint64(8)).astype(np.float64) * (2.0**-24)
    np.testing.assert_allclose(out, (-2.0 + u * 5.0).astype(np.float32), atol=1e-6)
    assert (out >= -2.0).all() and (out < 3.0).all()


def test_route_refuses_more_than_one_tile_typed():
    """n > 64 is T>=2: a typed refusal, before any backend/device is needed."""
    with pytest.raises(hr.SpyreRngError) as excinfo:
        hr.route_uniform_draw(0.0, 1.0, 65)
    assert "T>=2" in str(excinfo.value) or "T-DV-34" in str(excinfo.value)
    assert isinstance(excinfo.value, NotImplementedError)


def test_route_empty_is_noop():
    out = hr.route_uniform_draw(0.0, 1.0, 0)
    assert out.shape == (0,)


# ---------------------------------------------------------------------------
# DEVICE-GATED: the end-to-end fill.
# ---------------------------------------------------------------------------


def _runtime_available() -> bool:
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


device_only = pytest.mark.skipif(
    not (_runtime_available() and _rng_kat_registered()),
    reason="needs a spyre device AND rng_kat registered in the box's "
    "sentient_codegen (codegen tree at a staging tip)",
)


@pytest.fixture
def device_rng_on(monkeypatch):
    monkeypatch.setenv(hr.DeviceRngSwitch.ENV, "1")
    assert hr.device_rng_enabled() is True


@device_only
def test_uniform_fills_unit_interval(device_rng_on):
    t = torch.empty(64, device="spyre")
    t.uniform_()
    vals = t.cpu().numpy()
    assert vals.shape == (64,)
    assert (vals >= 0.0).all() and (vals < 1.0).all()


@device_only
def test_uniform_fills_requested_range(device_rng_on):
    t = torch.empty(32, device="spyre")
    t.uniform_(-3.0, 7.0)
    vals = t.cpu().numpy()
    assert (vals >= -3.0).all() and (vals < 7.0).all()


@device_only
def test_uniform_reproducible_with_same_seed(device_rng_on):
    spyre = getattr(torch, "spyre")
    spyre.manual_seed(2024)
    a = torch.empty(64, device="spyre")
    a.uniform_()
    spyre.manual_seed(2024)
    b = torch.empty(64, device="spyre")
    b.uniform_()
    assert np.array_equal(a.cpu().numpy(), b.cpu().numpy())


@device_only
def test_successive_draws_differ(device_rng_on):
    spyre = getattr(torch, "spyre")
    spyre.manual_seed(7)
    a = torch.empty(64, device="spyre")
    a.uniform_()
    b = torch.empty(64, device="spyre")
    b.uniform_()
    # Offset advanced across draws, so the second draw differs.
    assert not np.array_equal(a.cpu().numpy(), b.cpu().numpy())


@device_only
def test_fill_matches_host_map_of_device_bits(device_rng_on):
    """The fill equals the host map of draw_raw_state for the same (seed, offset)."""
    spyre = getattr(torch, "spyre")
    spyre.manual_seed(31337)
    t = torch.empty(64, device="spyre")
    t.uniform_()
    got = t.cpu().numpy()

    # Reproduce exactly what route_uniform_draw did: reserve 64 from the SAME
    # state, then draw and map on the host.
    spyre.manual_seed(31337)
    state = spyre._impl._rng_state(None)
    offset = state.reserve(64)
    x0, x1 = hr.draw_raw_state(state.seed, offset)
    words = np.concatenate([np.asarray(x0, np.uint32), np.asarray(x1, np.uint32)])
    expected = hr.raw_words_to_uniform(words, 0.0, 1.0)
    np.testing.assert_array_equal(got, expected)


@device_only
def test_uniform_65_refuses_typed(device_rng_on):
    t = torch.empty(65, device="spyre")
    with pytest.raises(hr.SpyreRngError):
        t.uniform_()
