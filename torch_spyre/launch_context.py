# Copyright 2026 The Torch-Spyre Authors.
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

"""Launch-context provider seam for bounded-synchronize fault surfacing.

A bounded synchronize that times out raises a typed error carrying stream,
device, timeout, and FIFO state. The identity of WHAT was launched — kernel
name, binary digest, tensor count, submission time — lives at the compiler
stack (triton-spyre launcher), not in the stream layer. Rather than coupling
the stream layer to the compiler, the compiler registers a context provider
here; when a timeout fires, the Python synchronize wrapper calls the provider
and appends its text to the error message. With no provider registered
(plain torch-spyre use without triton-spyre), the timeout error still fires
with the stream-layer facts alone.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

# Returns a short human-readable description of the most recent launch(es) to
# append to a synchronize-timeout error, or None when nothing is known.
LaunchContextProvider = Callable[[], Optional[str]]

_provider_lock = threading.Lock()
_provider: Optional[LaunchContextProvider] = None


def set_launch_context_provider(provider: Optional[LaunchContextProvider]) -> None:
    """Register (or clear, with None) the compiler-side launch-context provider.

    The provider is called lazily, only when a bounded synchronize times out,
    so registration is cheap and the provider itself may consult live state.
    It runs synchronously on the timeout-error path and must return promptly
    (no blocking calls or I/O) — a slow provider extends the caller's wait by
    its own latency, unbounded. No watchdog is deliberately implemented; the
    contract is documentation-level to keep the seam simple.
    """
    global _provider
    with _provider_lock:
        _provider = provider


def describe_recent_launches() -> Optional[str]:
    """Return the provider's description of recent launches, if any.

    Never raises: a broken provider must not mask the timeout error it is
    meant to enrich — a provider failure is reported inline instead. Runs
    synchronously on the timeout-error path; see
    set_launch_context_provider for the promptness contract.
    """
    with _provider_lock:
        provider = _provider
    if provider is None:
        return None
    try:
        return provider()
    except Exception as exc:  # noqa: BLE001 - surfaced inline, never re-raised
        return f"(launch-context provider failed: {exc!r})"
