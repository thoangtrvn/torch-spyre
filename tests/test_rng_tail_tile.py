"""A6 slice-2: RNG tail-tile (numel not a whole multiple of 64) admission.

The decision under test: ``_select_uniform_binding`` admits a tail tile as a
"direct" padded-caller bind ONLY when the caller's device allocation already
covers the whole-tile write (``device_size == (2*ceil(numel/64), 32)``) on the
identity/fp32/contiguous cell; otherwise it refuses typed. Coverage:

* the tail decision matrix (allocation-safe corridor vs under-allocation),
* the MEASURED device layouts that make it safe (a fresh 100-element tensor is
  ``(4,32)``; a slice into a larger buffer inherits the parent's ``(7,32)`` and
  refuses — so no direct bind ever writes past a caller's allocation),
* the launch-ABI extent fix (physical device-allocation extent passes gate 8; a
  logical-short extent refuses), and
* the binary is byte-identical (pinned to the silicon-witnessed digests).

No device: the launch itself is the hardware claim named in
``codegen/docs/rng-tail-tile-design.md`` §5.1; everything here runs offline.
"""

import hashlib

import pytest

torch = pytest.importorskip("torch")

from torch_spyre.rng import (  # noqa: E402
    SpyreRngError,
    _device_stick_layout,
    _select_uniform_binding,
)

KW = dict(from_=0.0, to=1.0, is_float32=True, is_contiguous=True)


@pytest.mark.parametrize(
    "numel, device_size",
    [
        (100, (4, 32)),  # 2 tiles (128 slots) on a 4-stick allocation: fits
        (120, (4, 32)),  # tail of the 2nd tile, 4-stick allocation
        (127, (4, 32)),  # last slot before the tile boundary
        (65, (4, 32)),  # 65 on an explicitly whole-tile-padded (4,32) alloc
        (96, (4, 32)),  # 96 on a deliberately padded (4,32) alloc
    ],
)
def test_tail_direct_admits_allocation_safe_corridor(numel, device_size):
    """A tail whose device allocation IS the 2*ceil(numel/64)-stick tile write
    is admitted as a direct padded bind."""
    assert (
        _select_uniform_binding(numel=numel, device_size=device_size, **KW) == "direct"
    )


@pytest.mark.parametrize(
    "numel, device_size",
    [
        (65, (3, 32)),  # real torch.empty(65): stick-padded to 3 sticks; the
        # 2-tile write needs 4 -> would overrun -> refuse
        (96, (3, 32)),  # real torch.empty(96): 3 sticks < 4 -> refuse
        (100, (2, 32)),  # under-allocated (2 sticks) < 4 -> refuse
        (200, (7, 32)),  # 200: ceil=4 tiles -> needs 8 sticks, has 7
    ],
)
def test_tail_refuses_under_allocation(numel, device_size):
    """A tail whose allocation does NOT cover the tile write refuses typed —
    never an overrun. This is the safety guard, not a limitation."""
    with pytest.raises(SpyreRngError) as e:
        _select_uniform_binding(numel=numel, device_size=device_size, **KW)
    assert "T-DV-34" in str(e.value)


@pytest.mark.parametrize(
    "kw",
    [
        dict(from_=-3.0, to=7.0, is_float32=True, is_contiguous=True),
        dict(from_=0.0, to=1.0, is_float32=False, is_contiguous=True),
        dict(from_=0.0, to=1.0, is_float32=True, is_contiguous=False),
    ],
    ids=["affine", "non-fp32", "non-contiguous"],
)
def test_tail_refuses_non_direct_cell(kw):
    """A tail is admitted only on the identity/fp32/contiguous cell; any other
    cell refuses typed (the host map cannot serve numel > 64 either)."""
    with pytest.raises(SpyreRngError):
        _select_uniform_binding(numel=100, device_size=(4, 32), **kw)


def test_measured_device_layout_admits_fresh_refuses_slice():
    """Anchors the safety argument to MEASURED device layouts: a fresh
    100-element spyre tensor is (4,32) -> direct; a slice into a 200-element
    buffer inherits the parent's (7,32) -> refused, so the direct bind never
    writes into the neighbour skew[100:128]."""
    try:
        fresh = torch.empty(100, dtype=torch.float32, device="spyre")
        big = torch.empty(200, dtype=torch.float32, device="spyre")
    except Exception as exc:  # no spyre substrate available
        pytest.skip(f"cannot construct a spyre tensor: {exc}")

    ds_fresh = _device_stick_layout(fresh)
    assert ds_fresh == (4, 32), ds_fresh
    assert _select_uniform_binding(numel=100, device_size=ds_fresh, **KW) == "direct"

    ds_slice = _device_stick_layout(big[0:100])
    assert ds_slice == (7, 32), ds_slice
    with pytest.raises(SpyreRngError):
        _select_uniform_binding(numel=100, device_size=ds_slice, **KW)


def _uniform_plan(n_tiles):
    try:
        from sentient_codegen.scheduler.matmul import PATTERN_REGISTRY
    except Exception:
        pytest.skip("sentient_codegen not importable (source env.sh)")
    if "rng_kat" not in PATTERN_REGISTRY:
        pytest.skip("rng_kat not registered in the importable codegen tree")
    import torch_spyre.rng as hr

    return hr.build_rng_kat_uniform_binary(n_tiles=n_tiles)


@pytest.mark.parametrize(
    "n_tiles, digest",
    [
        (1, "d0930bd58e87cd0094159096cf4338a7572e997a51cf6ba647d7f9f0750cea68"),
        (2, "a8c85caa91c4f1fdc3b0b8193aa582b63b753d086a411903c554ee89906122f9"),
    ],
)
def test_tail_binary_is_byte_identical(n_tiles, digest):
    """The tail-tile route reuses the tile-loop binary unchanged: the emitted
    bytes are the silicon-witnessed digests (no digest moves)."""
    binary, _ = _uniform_plan(n_tiles)
    assert hashlib.sha256(binary).hexdigest() == digest
    # Build a second time (distinct call) and confirm the same bytes.
    binary2, _ = _uniform_plan(n_tiles)
    assert hashlib.sha256(binary2).hexdigest() == digest


def test_abi_gate_accepts_physical_extent_refuses_short():
    """The launch ABI is fed the DEVICE-ALLOCATION extent: for a 2-tile tail
    (required 512 B) the 512 B allocation passes gate 8 while a logical-short
    extent refuses; the 1-tile cell's physical extent equals its logical."""
    binary, plan = _uniform_plan(2)
    from sentient_codegen.backend.launch_abi import (
        LaunchAbiError,
        validate_positional_launch,
    )

    params_req = plan.symbols[0].required_extent_bytes
    out_req = plan.symbols[1].required_extent_bytes
    assert out_req == 512  # 2 tiles * 2 sticks * 128 B

    def specs(out_ext):
        return {
            plan.symbols[0].symbol_id: (params_req, "input"),
            plan.symbols[1].symbol_id: (out_ext, "output"),
        }

    # Physical device-allocation extent (a 4-stick allocation) == required.
    validate_positional_launch(plan, binary, specs(512))  # passes
    # A window one stick short but still stick-aligned (384 B) refuses at gate 8
    # (coverage): the physical extent must reach the whole-tile write.
    with pytest.raises(LaunchAbiError) as e:
        validate_positional_launch(plan, binary, specs(384))
    assert "WINDOW-EXTENT-UNCOVERED" in str(e.value)
    # A 100-element tensor's raw logical nbytes (400 B) also refuses — here at
    # gate 5 (misaligned), the other wall a logical-extent bind hits.
    with pytest.raises(LaunchAbiError):
        validate_positional_launch(plan, binary, specs(400))

    # The 1-tile cell: physical == logical (256 == 64*4), unchanged.
    binary1, plan1 = _uniform_plan(1)
    params_req1 = plan1.symbols[0].required_extent_bytes
    out_req1 = plan1.symbols[1].required_extent_bytes
    assert out_req1 == 256
    validate_positional_launch(
        plan1,
        binary1,
        {
            plan1.symbols[0].symbol_id: (params_req1, "input"),
            plan1.symbols[1].symbol_id: (out_req1, "output"),
        },
    )  # passes, identical to today
