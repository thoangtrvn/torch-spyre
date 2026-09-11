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

from __future__ import annotations

from torch_spyre._runtime import codegen_only_available, require_runtime

if not codegen_only_available():
    raise ImportError(
        "torch_spyre._hooks runtime extension is not built. Set "
        "TORCH_SPYRE_CODEGEN_ONLY=1 only for compiler/import/TTIR capture workflows, "
        "or install the full Spyre runtime build."
    )

__all__ = ["device_count"]


def device_count() -> int:
    require_runtime("torch_spyre._hooks.device_count")
    return 0
