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

``route_uniform_draw`` remains the host-map path (device raw bits, host map).
The P5 route (``route_uniform_fill`` -> ``launch_uniform_draw``) writes the
silicon-witnessed 2-stick UNIFORM egress straight into the caller tensor (or a
(2,32) scratch + one device→device copy) with zero host round trip, only for
the identity ``[0, 1)`` map. The pure binding rule, the route refusals, and the
uniform binary digest are host-testable and run in the OFFLINE tier below.
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
def test_uniform_fills_requested_range_refuses_typed(device_rng_on):
    """Non-identity [from_,to) is NOT a device route (no on-device affine).

    The witnessed egress writes the bare [0,1) grid; affine scaling needs a
    device kernel that does not exist, so the switch-ON path refuses typed
    rather than emit out-of-request values."""
    t = torch.empty(32, device="spyre")
    with pytest.raises(hr.SpyreRngError) as excinfo:
        t.uniform_(-3.0, 7.0)
    assert "identity" in str(excinfo.value).lower() or "[0, 1)" in str(excinfo.value)


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


# ---------------------------------------------------------------------------
# OFFLINE: the caller-tensor binding rule (pure), the route refusals, and the
# uniform binary digest — no device needed.
# ---------------------------------------------------------------------------


def test_select_uniform_binding_direct_and_scratch():
    """The 2-stick binding rule: direct only for a dense (2,32) 64-elem caller;
    every other allocation-safe case takes the scratch + device-copy path."""
    from torch_spyre.rng import _select_uniform_binding

    kw = dict(is_float32=True, is_contiguous=True)
    # numel 0 is a no-op.
    assert _select_uniform_binding(numel=0, device_size=(0,), **kw) == "empty"
    # numel 64 with the exact (2,32) dense layout binds the caller directly.
    assert _select_uniform_binding(numel=64, device_size=(2, 32), **kw) == "direct"
    # numel <= 32 allocates ONE stick — a 2-stick store would overrun -> scratch.
    assert _select_uniform_binding(numel=32, device_size=(1, 32), **kw) == "scratch"
    assert _select_uniform_binding(numel=1, device_size=(1, 32), **kw) == "scratch"
    # numel 33..63 is 2 sticks but not the full 64-element direct case -> scratch.
    assert _select_uniform_binding(numel=33, device_size=(2, 32), **kw) == "scratch"
    assert _select_uniform_binding(numel=63, device_size=(2, 32), **kw) == "scratch"
    # A non-(2,32) device layout (e.g. an (8,8) row-padded tensor) -> scratch.
    assert _select_uniform_binding(numel=64, device_size=(8, 1, 32), **kw) == "scratch"
    # Unknown layout (None) never guesses direct -> scratch.
    assert _select_uniform_binding(numel=64, device_size=None, **kw) == "scratch"


def test_select_uniform_binding_refuses_typed():
    """T>64 / non-fp32 / non-contiguous all refuse typed (never silent)."""
    from torch_spyre.rng import _select_uniform_binding

    with pytest.raises(hr.SpyreRngError) as e:
        _select_uniform_binding(
            numel=65, is_float32=True, is_contiguous=True, device_size=(2, 32)
        )
    assert "T>=2" in str(e.value) or "T-DV-34" in str(e.value)
    with pytest.raises(hr.SpyreRngError) as e:
        _select_uniform_binding(
            numel=64, is_float32=False, is_contiguous=True, device_size=(2, 32)
        )
    assert "float32" in str(e.value)
    with pytest.raises(hr.SpyreRngError) as e:
        _select_uniform_binding(
            numel=64, is_float32=True, is_contiguous=False, device_size=(2, 32)
        )
    assert "contiguous" in str(e.value)


class _FakeCaller:
    """Minimal tensor stand-in for the route guards reached before any device."""

    def __init__(self, numel, device="cpu"):
        self._n = int(numel)
        self.device = device

    def numel(self):
        return self._n


def test_route_uniform_fill_refuses_affine_and_t_ge_2_typed():
    from torch_spyre.rng import route_uniform_fill

    # numel 0 is a no-op (returns the caller).
    f0 = _FakeCaller(0)
    assert route_uniform_fill(f0, 0.0, 1.0) is f0
    # Non-identity bounds refuse BEFORE any device/registration need.
    with pytest.raises(hr.SpyreRngError) as e:
        route_uniform_fill(_FakeCaller(32), -3.0, 7.0)
    assert "identity" in str(e.value).lower() or "[0, 1)" in str(e.value)
    # numel > 64 refuses (T-DV-34) BEFORE any device need.
    with pytest.raises(hr.SpyreRngError) as e:
        route_uniform_fill(_FakeCaller(65), 0.0, 1.0)
    assert "T>=2" in str(e.value) or "T-DV-34" in str(e.value)


def test_uniform_binary_digest_matches_the_silicon_witness():
    """The uniform route compiles to the witnessed ``d0930bd5…`` and declares
    the fp32 output symbol (T-LR-7 gate armed)."""
    import hashlib

    try:
        from sentient_codegen.scheduler.matmul import PATTERN_REGISTRY
    except Exception:
        pytest.skip("sentient_codegen not importable (source env.sh)")
    if "rng_kat" not in PATTERN_REGISTRY:
        pytest.skip("rng_kat not registered in the importable codegen tree")

    binary, plan = hr.build_rng_kat_uniform_binary()
    assert hashlib.sha256(binary).hexdigest() == (
        "d0930bd58e87cd0094159096cf4338a7572e997a51cf6ba647d7f9f0750cea68"
    )
    assert len(binary) == 4608
    # Segment 1 (output) declares fp32 / IEEE_FP32 / 32 bits / contiguous.
    out_sym = plan.symbols[1]
    assert out_sym.logical_dtype == "f32"
    assert out_sym.device_dtype == "IEEE_FP32"
    assert out_sym.device_element_bits == 32
    assert out_sym.contiguous_stick_layout is True
