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

"""Single authority for the SENCORES core budget (T-SR-39).

Every consumer of the Spyre core budget validates through
:func:`validated_sencores` — a malformed or out-of-range value refuses
LOUD at the point of read, never silently falls back to a default. The
silent-fallback variant was a real defect: ``device/interface.py``
caught the ``ValueError`` from a garbage ``SENCORES`` env value and
reported ``multi_processor_count=32`` with no signal, so every
downstream core-count decision was built on an unrequested budget.

The ceiling is AIU 1.0's physical core count (32). SEN1P5 revisits the
bound; until then the 1p0 bound is the honest one — refusing >32 on a
1p0 device is correct even if a future generation raises it.

This module is a deliberate leaf (stdlib-only imports) so
``_inductor/config.py``, ``device/interface.py``, and the ops layer can
share it without import cycles.
"""

from __future__ import annotations

SPYRE_1P0_CORE_COUNT = 32


def validated_sencores(raw: object, *, source: str = "SENCORES") -> int:
    """Validate a raw SENCORES value into a ``[1, 32]`` core budget.

    Refuses loud (``ValueError``) on non-integer values and on integers
    outside ``[1, SPYRE_1P0_CORE_COUNT]`` — the same fail-loud law as
    the fork's ``parse_options`` unknown-option refusal. Callers pass
    ``source`` so the error names where the value came from (env var,
    config field, function argument).
    """
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(
            f"{source} must be an integer in [1, {SPYRE_1P0_CORE_COUNT}] "
            f"(AIU 1.0 core count), got {raw!r}"
        ) from None
    if not 1 <= value <= SPYRE_1P0_CORE_COUNT:
        raise ValueError(
            f"{source} must be an integer in [1, {SPYRE_1P0_CORE_COUNT}] "
            f"(AIU 1.0 core count), got {value}"
        )
    return value
