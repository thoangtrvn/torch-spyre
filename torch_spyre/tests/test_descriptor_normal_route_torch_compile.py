# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""On-box torch.compile boundary proof for the bounded descriptor-negation route.

Run on the Spyre host after sourcing the development environment:

TORCHINDUCTOR_COMPILE_THREADS=1 TORCH_SPYRE_EXPERIMENTAL_DESCRIPTOR_POINTWISE=1 \
PYTHONPATH=codegen:codegen/tests:<triton-spyre>/python:<torch-spyre> \
python -m pytest torch_spyre/tests/test_descriptor_normal_route_torch_compile.py -v

The local codegen-only Python ``torch_spyre._C`` cannot allocate or launch a
Spyre tensor and therefore cannot establish this boundary. The test skips only
after an explicit prerequisite probe records which native capability is absent.
It never treats that skip as runtime or hardware evidence.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch


def _native_boundary_prerequisite():
    if os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") != "1":
        return "TORCHINDUCTOR_COMPILE_THREADS=1 is required"
    try:
        import torch_spyre
        torch_spyre._autoload()
    except Exception as exc:
        return f"torch-spyre autoload failed: {type(exc).__name__}: {exc}"
    try:
        from torch_spyre import _C
    except Exception as exc:
        return f"native torch_spyre._C import failed: {type(exc).__name__}: {exc}"
    required = (
        "RuntimeLaunchObservation",
        "RuntimeLaunchIdentityVerdict",
        "launch_kernel_from_bytes",
        "validate_runtime_launch_observation",
    )
    missing = tuple(name for name in required if not hasattr(_C, name))
    if missing:
        return "native torch_spyre._C is missing: " + ", ".join(missing)
    try:
        probe = torch.empty((1, 64), dtype=torch.float16, device="spyre")
        torch.spyre.synchronize()
    except Exception as exc:
        return f"Spyre allocation/runtime probe failed: {type(exc).__name__}: {exc}"
    if probe.device.type != "spyre":
        return f"Spyre allocation returned device {probe.device}"
    return None


_PREREQUISITE_FAILURE = _native_boundary_prerequisite()
requires_native_boundary = pytest.mark.skipif(
    _PREREQUISITE_FAILURE is not None,
    reason=_PREREQUISITE_FAILURE or "native Spyre boundary unavailable",
)


def _negation_normal_route_authority():
    import sentient_codegen
    from sentient_codegen.backend.descriptor_negation_artifact import (
        descriptor_negation_artifact_evidence_from_contract,
        descriptor_negation_artifact_from_evidence,
        descriptor_negation_artifact_report,
    )
    from sentient_codegen.backend.descriptor_negation_hardware import (
        DescriptorNegationHardwareKat,
    )
    from sentient_codegen.backend.descriptor_negation_route_authority import (
        build_descriptor_negation_normal_route_authority,
    )
    from test_semantic_to_schedule import (
        _descriptor_pointwise_negation_schedule_projection,
        _descriptor_pointwise_negation_segment_binding_facts,
    )

    package_file = getattr(sentient_codegen, "__file__", None)
    assert isinstance(package_file, str) and package_file, (
        "sentient_codegen package has no filesystem origin; the descriptor "
        "negation integration fixture cannot be resolved"
    )
    fixture = (
        Path(package_file).resolve().parent.parent
        / "tests"
        / "fixtures"
        / "ttir_descriptor"
        / "structured_descriptor_pointwise_negation_512x128.ttir"
    )
    assert fixture.is_file(), (
        "descriptor negation integration fixture is missing relative to the "
        f"imported sentient_codegen package: {fixture}"
    )
    source = fixture.read_text(encoding="utf-8")
    source_fingerprint = hashlib.sha256(source.encode()).hexdigest()
    projection = _descriptor_pointwise_negation_schedule_projection(
        artifact_fingerprint=source_fingerprint,
        segment_binding_facts=_descriptor_pointwise_negation_segment_binding_facts(
            artifact_fingerprint=source_fingerprint,
        ),
    )
    assert projection.contract is not None
    evidence = descriptor_negation_artifact_evidence_from_contract(
        projection.contract,
        source_ttir=source,
    )
    artifact = descriptor_negation_artifact_from_evidence(evidence)
    hardware = DescriptorNegationHardwareKat(
        schema_version="descriptor-negation-hardware-kat-v1",
        status="hardware_correctness_observed",
        target="rcudd1a",
        generation="AIU 1.0",
        emitted_binary_sha256=artifact.emitted_binary_sha256,
        runtime_identity_status="runtime_launch_identity_validated",
        proves_runtime_identity=True,
        element_count=512 * 128,
        mismatch_count=0,
        maximum_absolute_difference=0.0,
        proves_hardware_execution=True,
        proves_hardware_correctness=True,
        production_authority=False,
    )
    return build_descriptor_negation_normal_route_authority(
        artifact_report=descriptor_negation_artifact_report(artifact),
        trusted_evidence=evidence,
        hardware_kat=hardware,
    )


def _relu_normal_route_authority():
    import sentient_codegen
    from sentient_codegen.backend.descriptor_relu_artifact import (
        descriptor_relu_artifact_evidence_from_contract,
        descriptor_relu_artifact_from_evidence,
        descriptor_relu_artifact_report,
    )
    from sentient_codegen.backend.descriptor_relu_hardware import DescriptorReluHardwareKat
    from sentient_codegen.backend.descriptor_relu_route_authority import (
        build_descriptor_relu_normal_route_authority,
    )
    from test_semantic_to_schedule import (
        _descriptor_pointwise_negation_segment_binding_facts,
        _descriptor_pointwise_relu_schedule_projection,
    )

    package_file = getattr(sentient_codegen, "__file__", None)
    assert isinstance(package_file, str) and package_file
    fixture = (
        Path(package_file).resolve().parent.parent
        / "tests"
        / "fixtures"
        / "ttir_descriptor"
        / "structured_descriptor_pointwise_relu_512x128.ttir"
    )
    assert fixture.is_file()
    source = fixture.read_text(encoding="utf-8")
    source_fingerprint = hashlib.sha256(source.encode()).hexdigest()
    projection = _descriptor_pointwise_relu_schedule_projection(
        artifact_fingerprint=source_fingerprint,
        segment_binding_facts=_descriptor_pointwise_negation_segment_binding_facts(
            artifact_fingerprint=source_fingerprint,
        ),
    )
    assert projection.contract is not None
    evidence = descriptor_relu_artifact_evidence_from_contract(
        projection.contract,
        source_ttir=source,
    )
    artifact = descriptor_relu_artifact_from_evidence(evidence)
    hardware = DescriptorReluHardwareKat(
        schema_version="descriptor-relu-hardware-kat-v1",
        status="hardware_correctness_observed",
        target="rcudd1a",
        generation="AIU 1.0",
        emitted_binary_sha256=artifact.emitted_binary_sha256,
        emitted_binary_size=len(evidence.emitted_binary),
        runtime_identity_status="runtime_identity_authenticated",
        cache_status="bytes_cache_miss",
        program_copied_this_launch=True,
        proves_runtime_identity=True,
        element_count=512 * 128,
        mismatch_count=0,
        maximum_absolute_difference=0.0,
        signed_zero_input_count=13108,
        signed_zero_expected_count=39324,
        signed_zero_observed_count=39324,
        signed_zero_mismatch_count=0,
        input_nan_count=0,
        expected_nan_count=0,
        observed_nan_count=0,
        input_corpus="finite_dl16_exact_with_signed_zeros",
        proves_hardware_execution=True,
        proves_hardware_correctness=True,
        production_authority=False,
    )
    return build_descriptor_relu_normal_route_authority(
        artifact_report=descriptor_relu_artifact_report(artifact),
        trusted_evidence=evidence,
        hardware_kat=hardware,
    )


@requires_native_boundary
def test_descriptor_negation_torch_compile_cold_and_warm():
    """Cross parent authorization, Inductor, registered backend, cache, and launch."""

    import torch._dynamo
    import torch._inductor.config as inductor_config
    from torch_spyre._inductor import config as spyre_config
    from triton.backends.spyre import driver as spyre_driver

    assert inductor_config.compile_threads == 1
    spyre_config.disable_descriptor_normal_route()
    spyre_config.authorize_descriptor_negation_normal_route(
        _negation_normal_route_authority()
    )
    torch._dynamo.reset()

    compiled = torch.compile(lambda value: -value, backend="inductor")
    host = torch.arange(512 * 128, dtype=torch.int32).remainder(17)
    host = (host - 8).to(torch.float16).div(2).reshape(512, 128)
    expected = -host
    assert bool(torch.isfinite(host).all())
    assert bool(torch.isfinite(expected).all())
    value = host.to("spyre")

    cold = compiled(value)
    torch.spyre.synchronize()
    cold_observation = spyre_driver.last_launch_observation()
    warm = compiled(value)
    torch.spyre.synchronize()
    warm_observation = spyre_driver.last_launch_observation()

    assert cold.numel() == expected.numel() == 512 * 128
    assert warm.numel() == expected.numel() == 512 * 128
    assert torch.equal(cold.cpu(), expected)
    assert torch.equal(warm.cpu(), expected)
    for observation in (cold_observation, warm_observation):
        assert observation["native_origin_verified"] is True
        assert observation["launched_binary_sha256"] == observation["cached_binary_sha256"]
        assert observation["launched_binary_sha256"] == observation["submitted_binary_sha256"]
        assert observation["submitted_binary_matches_cached"] is True
        assert observation["submitted_binary_matches_launched"] is True
    assert cold_observation["bytes_cache_status"] == "bytes_cache_miss"
    assert warm_observation["bytes_cache_status"] == "bytes_cache_hit"


@requires_native_boundary
def test_descriptor_relu_torch_compile_cold_and_warm():
    """Prove ReLU through the real default-off normal route and native cache."""

    import torch._dynamo
    import torch._inductor.config as inductor_config
    from torch_spyre._inductor import config as spyre_config
    from triton.backends.spyre import driver as spyre_driver

    assert inductor_config.compile_threads == 1
    spyre_config.disable_descriptor_normal_route()
    spyre_config.authorize_descriptor_relu_normal_route(_relu_normal_route_authority())
    torch._dynamo.reset()

    compiled = torch.compile(torch.relu, backend="inductor")
    values = (-4.0, -2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, 4.0)
    element_count = 512 * 128
    host = torch.tensor(
        (values * ((element_count + len(values) - 1) // len(values)))[:element_count],
        dtype=torch.float16,
    ).reshape(512, 128)
    zero = torch.tensor(0.0, dtype=torch.float16)
    expected = torch.where(host > zero, host, zero)
    value = host.to("spyre")

    cold = compiled(value)
    torch.spyre.synchronize()
    cold_observation = spyre_driver.last_launch_observation()
    warm = compiled(value)
    torch.spyre.synchronize()
    warm_observation = spyre_driver.last_launch_observation()
    cold_host = cold.cpu()
    warm_host = warm.cpu()

    for observed in (cold_host, warm_host):
        assert torch.equal(observed, expected)
        expected_zero = expected == 0
        assert torch.equal(
            torch.signbit(observed)[expected_zero],
            torch.signbit(expected)[expected_zero],
        )
    for observation in (cold_observation, warm_observation):
        assert observation["native_origin_verified"] is True
        assert observation["launched_binary_sha256"] == observation["cached_binary_sha256"]
        assert observation["launched_binary_sha256"] == observation["submitted_binary_sha256"]
        assert observation["submitted_binary_matches_cached"] is True
        assert observation["submitted_binary_matches_launched"] is True
    assert cold_observation["bytes_cache_status"] == "bytes_cache_miss"
    assert warm_observation["bytes_cache_status"] == "bytes_cache_hit"
