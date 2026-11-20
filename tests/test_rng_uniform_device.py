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
silicon-witnessed 2-stick UNIFORM egress straight into the caller tensor with
zero host round trip, for the direct cell — float32, contiguous, identity
``[0, 1)``, device layout ``(2,32)``, ``numel == 64`` (the extent-exact cell;
doc 54 v7/v8). A ``(2,32)`` scratch + device→device copy would cover the other
callers but the f32 d2d copy is refused on 1p0 (pointer-grain gate), so
``1..63`` along with non-identity bounds / non-fp32 / non-contiguous fall back
to the unchanged host map, byte-identically; only ``numel > 64`` refuses. The
pure binding rule, the route refusals, and the uniform binary digest are
host-testable and run in the OFFLINE tier below.
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
    """Non-identity [from_,to) takes the host-map path (device bits, host
    affine) — it still fills correctly; this is NOT a refusal (doc 54 §4.1)."""
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


@device_only
def test_uniform_fp16_still_succeeds(device_rng_on):
    """NON-REGRESSION (doc 54 §4.1): fp16 keeps today's host-map path and fills.
    Established by execution 2026-11-19 that fp16 succeeds today; this pins it so
    an added dtype refusal would go red."""
    t = torch.empty(32, dtype=torch.float16, device="spyre")
    t.uniform_()
    v = t.cpu().numpy()
    assert v.dtype == np.float16
    assert (v >= 0.0).all() and (v < 1.0).all()


@device_only
def test_uniform_strided_behaves_as_today(device_rng_on):
    """NON-REGRESSION (doc 54 §4.1/§8): a strided destination fills successfully
    today (host-map path via ``spyre__uniform_``'s ``self.copy_``). Pin the
    OBSERVED behaviour (it succeeds, values in range) — do NOT pin an assumed
    layout/strided outcome; the exact scatter semantics are unverified."""
    base = torch.empty(128, dtype=torch.float32, device="spyre")
    t = base[::2]
    assert t.shape == (64,) and not t.is_contiguous()
    t.uniform_()
    v = t.cpu().numpy()
    assert v.shape == (64,)
    assert (v >= 0.0).all() and (v < 1.0).all()


@device_only
def test_uniform_direct_bind_stream_identical_to_host_map(device_rng_on):
    """§4.3 stream contract: the DIRECT device bind (numel=64) yields the SAME
    stream as today's host map — proving the device stick order == the host
    concatenation order (a wrong order would pass a range check and still be
    wrong)."""
    spyre = getattr(torch, "spyre")
    spyre.manual_seed(99)
    a = torch.empty(64, dtype=torch.float32, device="spyre")
    a.uniform_()  # direct device bind
    got = a.cpu().numpy()
    spyre.manual_seed(99)
    exp = np.asarray(hr.route_uniform_draw(0.0, 1.0, 64), dtype=np.float32)
    np.testing.assert_array_equal(got.view(np.uint32), exp.view(np.uint32))


# ---------------------------------------------------------------------------
# OFFLINE: the caller-tensor binding rule (pure), the route refusals, and the
# uniform binary digest — no device needed.
# ---------------------------------------------------------------------------


def test_select_uniform_binding_direct_and_fallback():
    """doc 54 §4.1: direct ONLY for a dense (2,32) two-stick caller with the
    identity map + fp32 + contiguous; every other caller takes the host path
    (fallback), never a refusal. Select shapes by computed device_size."""
    from torch_spyre.rng import _select_uniform_binding

    kw = dict(from_=0.0, to=1.0, is_float32=True, is_contiguous=True)
    assert _select_uniform_binding(numel=0, device_size=(0,), **kw) == "empty"
    # DIRECT only at numel==64 on a dense (2,32) layout (extent-exact).
    assert _select_uniform_binding(numel=64, device_size=(2, 32), **kw) == "direct"
    # 33..63 is allocation-safe (two sticks) but NOT extent-exact -> fallback.
    for n in (33, 63):
        assert _select_uniform_binding(numel=n, device_size=(2, 32), **kw) == (
            "fallback"
        )
    # 1..32 is a (1,32) one-stick allocation -> host path (scratch f32-blocked).
    for n in (1, 32):
        assert _select_uniform_binding(numel=n, device_size=(1, 32), **kw) == (
            "fallback"
        )
    # Multi-tile device loop: a witnessed whole-tile width on its exact (2n,32)
    # layout is direct; the same numel on the wrong layout falls back.
    for tiles in (2, 4, 16):
        numel = tiles * 64
        assert (
            _select_uniform_binding(numel=numel, device_size=(2 * tiles, 32), **kw)
            == "direct"
        )
        assert (
            _select_uniform_binding(numel=numel, device_size=(2, 32), **kw)
            == "fallback"
        )
    # A non-whole width (65) or an unwitnessed tile count (3 tiles = 192)
    # refuses typed, never a silent host draw of an unwitnessed geometry.
    import pytest as _pytest
    from torch_spyre.rng import SpyreRngError

    for bad in (65, 192):
        with _pytest.raises(SpyreRngError):
            _select_uniform_binding(numel=bad, device_size=(2, 32), **kw)
    # A non-(2,32) layout (e.g. a 2-D (8,8) is EIGHT sticks) -> host path.
    assert _select_uniform_binding(numel=64, device_size=(8, 1, 32), **kw) == (
        "fallback"
    )
    assert _select_uniform_binding(numel=64, device_size=None, **kw) == "fallback"
    # Non-identity map / non-fp32 / non-contiguous -> host path, NOT a refusal.
    assert (
        _select_uniform_binding(
            numel=64,
            device_size=(2, 32),
            from_=-3.0,
            to=7.0,
            is_float32=True,
            is_contiguous=True,
        )
        == "fallback"
    )
    assert (
        _select_uniform_binding(
            numel=64,
            device_size=(2, 32),
            from_=0.0,
            to=1.0,
            is_float32=False,
            is_contiguous=True,
        )
        == "fallback"
    )
    assert (
        _select_uniform_binding(
            numel=64,
            device_size=(2, 32),
            from_=0.0,
            to=1.0,
            is_float32=True,
            is_contiguous=False,
        )
        == "fallback"
    )


def test_select_uniform_binding_refuses_only_t_gt_64():
    """The ONLY typed refusal added/kept here is numel > 64 (T-DV-34)."""
    from torch_spyre.rng import _select_uniform_binding

    with pytest.raises(hr.SpyreRngError) as e:
        _select_uniform_binding(
            numel=65,
            from_=0.0,
            to=1.0,
            is_float32=True,
            is_contiguous=True,
            device_size=(2, 32),
        )
    assert "T>=2" in str(e.value) or "T-DV-34" in str(e.value)


def test_route_uniform_fill_fallback_and_t_gt_64():
    """route_uniform_fill returns False (host path) for every non-direct case —
    affine / fp16 / strided / 1-stick — and raises ONLY for numel > 64. No new
    refusals (doc 54 §4.1). CPU tensors: device_size is None -> fallback."""
    from torch_spyre.rng import route_uniform_fill

    assert route_uniform_fill(torch.empty(0, dtype=torch.float32), 0.0, 1.0) is True
    # Non-identity bounds -> host path (host map applies the affine).
    assert route_uniform_fill(torch.empty(64, dtype=torch.float32), -3.0, 7.0) is False
    # fp16 -> host path (spyre__uniform_ casts on the host today).
    assert route_uniform_fill(torch.empty(32, dtype=torch.float16), 0.0, 1.0) is False
    # strided -> host path (whatever today does; not narrowed).
    assert (
        route_uniform_fill(torch.empty(128, dtype=torch.float32)[::2], 0.0, 1.0)
        is False
    )
    # 1-stick (<=32) / non-(2,32) contiguous -> host path (scratch f32-blocked).
    assert route_uniform_fill(torch.empty(32, dtype=torch.float32), 0.0, 1.0) is False
    # ONLY numel > 64 refuses typed (T-DV-34) — unchanged from today.
    with pytest.raises(hr.SpyreRngError) as e:
        route_uniform_fill(torch.empty(65, dtype=torch.float32), 0.0, 1.0)
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


@pytest.mark.parametrize(
    "make, from_, to",
    [
        (lambda: torch.empty(32, dtype=torch.float32, device="spyre"), 0.0, 1.0),
        (lambda: torch.empty(63, dtype=torch.float32, device="spyre"), 0.0, 1.0),
        (lambda: torch.empty(64, dtype=torch.float32, device="spyre"), 0.0, 2.0),
        (lambda: torch.empty(32, dtype=torch.float16, device="spyre"), 0.0, 1.0),
    ],
    ids=["numel32", "numel63", "affine_0_2", "fp16_32"],
)
@device_only
def test_fallback_stream_identical_to_route_uniform_draw(
    device_rng_on, make, from_, to
):
    """NON-REGRESSION (doc 54 §4.1/§5): every FALLBACK caller (1..63, affine,
    fp16, strided) goes through today's host-map path and yields the SAME value
    stream as the pre-change ``route_uniform_draw`` for the same seed — i.e. the
    change is an optimisation of the direct cell, not a narrowing. Bit-exact."""
    spyre = getattr(torch, "spyre")
    probe = make()
    n = probe.numel()
    dtype = probe.dtype
    spyre.manual_seed(123)
    t = make()
    t.uniform_(from_, to)  # fallback path
    got = t.cpu().numpy()
    # Reproduce the pre-change stream exactly.
    spyre.manual_seed(123)
    raw = np.asarray(hr.route_uniform_draw(from_, to, n), dtype=np.float32)
    exp = torch.from_numpy(raw).to(dtype=dtype).cpu().numpy()
    view = np.uint16 if dtype == torch.float16 else np.uint32
    np.testing.assert_array_equal(got.view(view), exp.view(view))
