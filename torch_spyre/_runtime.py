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

import os

_CODEGEN_ONLY_ENV = "TORCH_SPYRE_CODEGEN_ONLY"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class SpyreRuntimeUnavailableError(RuntimeError):
    """Raised when a runtime-only Spyre API is used in codegen-only mode."""


def codegen_only_available() -> bool:
    """Return True when torch-spyre is intentionally running without runtime extensions."""
    return os.environ.get(_CODEGEN_ONLY_ENV, "0").strip().lower() in _TRUE_VALUES


def require_runtime(operation: str) -> None:
    """Fail loudly if a runtime-only operation is attempted in codegen-only mode."""
    if codegen_only_available():
        raise SpyreRuntimeUnavailableError(
            f"{operation} requires the Spyre runtime extension, but "
            f"{_CODEGEN_ONLY_ENV}=1 selected codegen-only mode. This mode is for "
            f"compiler/import/TTIR capture workflows only; unset {_CODEGEN_ONLY_ENV} "
            f"and install the full runtime build to allocate tensors or launch kernels."
        )
