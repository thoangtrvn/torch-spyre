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

import os
import sys

from torch.utils._config_module import install_config_module

from torch_spyre.core_budget import validated_sencores

# T-SR-39: single SENCORES read, validated loud ([1, 32]) — a malformed
# env value refuses at import instead of poisoning every downstream
# core-count decision. The value flows to SpyreOptions.num_cores via
# SpyreHeuristics.triton_kernel_kwargs (the problem_m/n/k channel).
sencores: int = validated_sencores(os.getenv("SENCORES", "32"))

# The bounded descriptor-negation route is deliberately process-local and
# default-off. A trusted caller registers typed authority in sentient_codegen
# and sets only its returned fingerprint here; the mutable authority object is
# never serialized into generated wrappers or Triton's JSON cache.
descriptor_normal_route_operation: str = "off"
descriptor_normal_route_rollout: str = "off"
descriptor_normal_route_kill_switch: str = "off"
descriptor_normal_route_authority_fingerprint: str = ""


def _authorize_descriptor_normal_route(
    authority: object,
    *,
    operation: str,
    rollout: str,
    register,
) -> str:
    """Enable one fixed canary only for enforced same-process compilation."""

    import torch._inductor.config as inductor_config
    from torch._inductor.async_compile import get_compile_threads

    compile_threads = get_compile_threads()
    if compile_threads != 1 or inductor_config.compile_threads != 1:
        raise RuntimeError(
            "descriptor normal route requires same-process Inductor compilation "
            "with TORCHINDUCTOR_COMPILE_THREADS=1"
        )
    fingerprint = register(authority)
    global descriptor_normal_route_operation
    global descriptor_normal_route_rollout
    global descriptor_normal_route_kill_switch
    global descriptor_normal_route_authority_fingerprint
    descriptor_normal_route_operation = operation
    descriptor_normal_route_rollout = rollout
    descriptor_normal_route_kill_switch = "off"
    descriptor_normal_route_authority_fingerprint = fingerprint
    return fingerprint


def authorize_descriptor_negation_normal_route(authority: object) -> str:
    """Enable the fixed negation canary."""

    from sentient_codegen.backend.descriptor_negation_route_authority import (
        register_descriptor_negation_normal_route_authority,
    )

    return _authorize_descriptor_normal_route(
        authority,
        operation="descriptor_negation",
        rollout="canary_512x128_negation",
        register=register_descriptor_negation_normal_route_authority,
    )


def authorize_descriptor_relu_normal_route(authority: object) -> str:
    """Enable the fixed exact-ReLU canary."""

    from sentient_codegen.backend.descriptor_relu_route_authority import (
        register_descriptor_relu_normal_route_authority,
    )

    return _authorize_descriptor_normal_route(
        authority,
        operation="descriptor_relu",
        rollout="canary_512x128_relu",
        register=register_descriptor_relu_normal_route_authority,
    )


def disable_descriptor_normal_route() -> None:
    """Disable normal descriptor routing without deleting registered evidence."""

    global descriptor_normal_route_operation
    global descriptor_normal_route_rollout
    global descriptor_normal_route_kill_switch
    global descriptor_normal_route_authority_fingerprint
    descriptor_normal_route_operation = "off"
    descriptor_normal_route_rollout = "off"
    descriptor_normal_route_kill_switch = "off"
    descriptor_normal_route_authority_fingerprint = ""


install_config_module(sys.modules[__name__])
