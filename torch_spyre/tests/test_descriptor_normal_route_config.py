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

from types import SimpleNamespace

import pytest
import torch._inductor.config as inductor_config

from torch_spyre._inductor import config


def test_descriptor_normal_route_authorization_is_typed_and_default_off(monkeypatch):
    import sentient_codegen.backend.descriptor_negation_route_authority as authority_module

    config.disable_descriptor_normal_route()
    monkeypatch.setattr(inductor_config, "compile_threads", 1)
    assert config.descriptor_normal_route_operation == "off"
    authority = SimpleNamespace()
    monkeypatch.setattr(
        authority_module,
        "register_descriptor_negation_normal_route_authority",
        lambda observed: "sha256:registered" if observed is authority else "",
    )

    fingerprint = config.authorize_descriptor_negation_normal_route(authority)

    assert fingerprint == "sha256:registered"
    assert config.descriptor_normal_route_operation == "descriptor_negation"
    assert config.descriptor_normal_route_rollout == "canary_512x128_negation"
    assert config.descriptor_normal_route_kill_switch == "off"
    assert config.descriptor_normal_route_authority_fingerprint == fingerprint

    config.disable_descriptor_normal_route()
    assert config.descriptor_normal_route_operation == "off"
    assert config.descriptor_normal_route_authority_fingerprint == ""


def test_descriptor_relu_normal_route_authorization_is_distinct(monkeypatch):
    import sentient_codegen.backend.descriptor_relu_route_authority as authority_module

    config.disable_descriptor_normal_route()
    monkeypatch.setattr(inductor_config, "compile_threads", 1)
    authority = SimpleNamespace()
    monkeypatch.setattr(
        authority_module,
        "register_descriptor_relu_normal_route_authority",
        lambda observed: "sha256:relu" if observed is authority else "",
    )

    fingerprint = config.authorize_descriptor_relu_normal_route(authority)

    assert fingerprint == "sha256:relu"
    assert config.descriptor_normal_route_operation == "descriptor_relu"
    assert config.descriptor_normal_route_rollout == "canary_512x128_relu"
    assert config.descriptor_normal_route_kill_switch == "off"


def test_descriptor_normal_route_refuses_async_compiler_process(monkeypatch):
    import sentient_codegen.backend.descriptor_negation_route_authority as authority_module

    config.disable_descriptor_normal_route()
    monkeypatch.setattr(inductor_config, "compile_threads", 2)
    register_calls = []
    monkeypatch.setattr(
        authority_module,
        "register_descriptor_negation_normal_route_authority",
        lambda authority: register_calls.append(authority),
    )

    with pytest.raises(RuntimeError, match="TORCHINDUCTOR_COMPILE_THREADS=1"):
        config.authorize_descriptor_negation_normal_route(SimpleNamespace())

    assert register_calls == []
    assert config.descriptor_normal_route_operation == "off"
    assert config.descriptor_normal_route_authority_fingerprint == ""
