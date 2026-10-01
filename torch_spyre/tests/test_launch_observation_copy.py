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
"""Regression: copying a native launch observation yields a diagnostic mapping.

A compiled-op launch returns a native ``RuntimeLaunchObservation`` from
``torch_spyre._C.launch_kernel_from_bytes``. The Triton launcher deep-copies that
result and then indexes it as a mapping (``observation["launched_binary_sha256"]
= ...``; ``observation.get(...)``). The native object has no mapping protocol, so
``__copy__``/``__deepcopy__`` single-use-CONSUME it into a plain ``dict`` of its
own fields. That keeps the launcher working while deliberately NOT authenticating
provenance:

* the dict is a plain mapping, not the pybind-registered native type, so it can
  never reach the native validator; and
* consuming through the copy spends the single-use original, so a later native
  validation of that original fails closed.

A genuine verdict is therefore obtainable only by validating the native original
BEFORE it is copied. Pickling stays rejected (reconstruction from a byte stream
would mint an observation outside the native launch path that roots trust).

This exercises the real pybind bindings, so it needs a built extension and a
device: there is no Python constructor for the observation, and the only genuine
producer is a real launch. It skips cleanly off-hardware.
"""
import copy
import hashlib
import pickle
import struct

import pytest
import torch

_SPYRE_AVAILABLE = False
try:
    import torch_spyre  # noqa: F401 — autoloads the "spyre" device
    _SPYRE_AVAILABLE = torch.zeros(1).to("spyre") is not None
except Exception:
    _SPYRE_AVAILABLE = False

requires_hw = pytest.mark.skipif(
    not _SPYRE_AVAILABLE, reason="spyre device not available (no hardware)"
)

# One shape that is actually routed end-to-end on 1p0: M=1024, N=64, C=32, DL16.
_M, _N, _NUM_CORES, _DTYPE_BITS = 1024, 64, 32, 16

# Default launch context (torch_spyre._C.launch_kernel_from_bytes defaults).
_TARGET, _GENERATION = "rcudd1a", "AIU 1.0"


def _emitted_sha256(binary: bytes) -> str:
    return "sha256:" + hashlib.sha256(binary).hexdigest()


def _cache_identity_prefixed(binary: bytes, target: str, generation: str) -> str:
    """Reproduce launch_identity.cpp CacheIdentityContext independently.

    Fields are framed in the fixed order (target, generation, binary), each as an
    8-byte little-endian length followed by its bytes. This Python side must stay
    byte-identical to the C++ side or the native validator's cross-check fails
    closed -- which is exactly what this reproduction guards against.
    """
    def framed(field: bytes) -> bytes:
        return struct.pack("<Q", len(field)) + field

    context = (
        framed(target.encode("utf-8"))
        + framed(generation.encode("utf-8"))
        + framed(binary)
    )
    return "sha256:" + hashlib.sha256(context).hexdigest()


def _launch_real_observation():
    """Launch the witnessed softmax kernel; return (observation, type, binary).

    Skips (not fails) when the built extension lacks the symbol or codegen is not
    importable, so a stale local build does not masquerade as a regression
    failure.
    """
    try:
        from torch_spyre._C import (  # type: ignore[attr-defined]
            RuntimeLaunchObservation,
            launch_kernel_from_bytes,
        )
    except Exception as exc:  # unbuilt / stale extension
        pytest.skip(
            f"torch_spyre._C lacks the launch-observation binding: {exc!r} "
            "(rebuild the extension on the box)"
        )

    try:
        from sentient_codegen.encoder.softmax_dispatch import (
            build_softmax_launch_plan,
        )
    except Exception:
        pytest.skip("sentient_codegen not importable (source env.sh)")

    plan = build_softmax_launch_plan(
        _M, _N, num_cores=_NUM_CORES, dtype_bits=_DTYPE_BITS
    )
    binary = bytes(plan.binary)

    # DEVICE-RESIDENT, HANDLES KEPT (see torch_spyre/ops/softmax.py): the kernel
    # writes into these exact tensors. Positional XLAT: seg0=scratch, seg1=x,
    # seg2=out.
    scratch = torch.zeros(
        (plan.scratch_sticks, plan.elems_per_stick),
        dtype=torch.float16,
        device="spyre",
    )
    x = torch.zeros((_M, _N), dtype=torch.float16, device="spyre")
    out = torch.zeros((_M, _N), dtype=torch.float16, device="spyre")

    observation = launch_kernel_from_bytes(binary, [scratch, x, out])
    assert isinstance(observation, RuntimeLaunchObservation)
    return observation, RuntimeLaunchObservation, binary


def _native_validator():
    try:
        from torch_spyre._C import (  # type: ignore[attr-defined]
            validate_runtime_launch_observation,
        )
    except Exception as exc:
        pytest.skip(
            f"torch_spyre._C lacks validate_runtime_launch_observation: {exc!r}"
        )
    return validate_runtime_launch_observation


@requires_hw
def test_deepcopy_consumes_native_observation_into_dict():
    """deepcopy(native observation) -> a plain diagnostic dict the launcher can index."""
    observation, RuntimeLaunchObservation, _ = _launch_real_observation()

    diagnostic = copy.deepcopy(observation)

    # A plain mapping, NOT the native type.
    assert isinstance(diagnostic, dict)
    assert not isinstance(diagnostic, RuntimeLaunchObservation)
    assert diagnostic["schema_version"] == "spyre-launch-observation-v2"
    assert diagnostic["launch_target"] == _TARGET

    # The exact mapping operations the Triton launcher performs after deepcopy.
    diagnostic["launched_binary_sha256"] = "sha256:" + "0" * 64  # __setitem__
    assert diagnostic.get("cached_binary_matches_launched_bytes") in (True, False)

    # A deepcopy of the diagnostic dict is now an ordinary dict copy.
    again = copy.deepcopy(diagnostic)
    assert isinstance(again, dict)
    assert again["launched_binary_sha256"] == "sha256:" + "0" * 64


@requires_hw
def test_native_validator_refuses_observation_already_consumed_by_deepcopy():
    """Once deepcopy has consumed the original, native validation fails closed."""
    observation, _, _ = _launch_real_observation()
    validate = _native_validator()

    # deepcopy single-use-consumes the native original into a diagnostic dict.
    consumed_dict = copy.deepcopy(observation)
    assert isinstance(consumed_dict, dict)

    # The validator can no longer consume the original: the single-use flag is set.
    dummy = "sha256:" + "0" * 64
    with pytest.raises(RuntimeError):
        validate(observation, dummy, _TARGET, _GENERATION, dummy)


@requires_hw
def test_native_validator_valid_without_deepcopy():
    """The native original validates to a genuine verdict when it is NOT copied first."""
    observation, _, binary = _launch_real_observation()
    validate = _native_validator()

    emitted = _emitted_sha256(binary)
    cache_identity = _cache_identity_prefixed(binary, _TARGET, _GENERATION)

    verdict = validate(observation, emitted, _TARGET, _GENERATION, cache_identity)
    assert verdict.origin_verified is True, list(verdict.mismatches())
    assert list(verdict.mismatches()) == []

    # The verdict is single-use too: a second validation of the (now consumed)
    # original fails closed.
    with pytest.raises(RuntimeError):
        validate(observation, emitted, _TARGET, _GENERATION, cache_identity)


@requires_hw
def test_native_observation_cannot_be_pickled():
    """Pickling stays rejected with a catchable TypeError, not an arity error."""
    observation, _, _ = _launch_real_observation()

    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(observation)
    with pytest.raises(TypeError):
        observation.__reduce__()
    with pytest.raises(TypeError):
        observation.__reduce_ex__(pickle.HIGHEST_PROTOCOL)

    # Rejecting pickling must not have consumed the single-use observation: a
    # deepcopy afterward still succeeds and yields the diagnostic dict.
    diagnostic = copy.deepcopy(observation)
    assert isinstance(diagnostic, dict)
