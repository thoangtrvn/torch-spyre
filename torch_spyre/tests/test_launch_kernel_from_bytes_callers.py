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
"""The direct-C++ launch route must not gain new callers silently.

``torch_spyre._C.launch_kernel_from_bytes`` binds tensor allocs POSITIONALLY
(``xlat[i] = tensor_allocs[i]``) and performs no plan validation of its own.
The T-LR-7 deferral record (C++/flex enforcement of the XlatRelocationPlan)
promises that, until that seam lands, an in-repo contract assertion guards
that no new bypass route appears silently. This file IS that assertion.

Every call site of the C++ launch in the ``torch_spyre`` package is enumerated
DYNAMICALLY from the checked-out package source at test time (no hardcoded
file list), then classified against the sanctioned-caller table below:

* ``ops/softmax.py`` — the plan-validating caller. Its tensor list is derived
  from the named ``tensors_by_symbol`` mapping and gated by
  ``validate_positional_launch`` before the C++ call (the T-LR-2B consume
  contract).
* ``ops/embedding.py`` — the recorded planless legacy caller. Embedding
  binaries carry no relocation plan today (IBR-indirect addressing, recorded
  deferral), so its positional call binds nothing a plan would describe.

A call site that is neither — a new op calling the C++ launch directly, or an
existing one re-routing around the validating wrapper — flips this test red
with the offending ``file:line``. That is the bypass-route signal; do not
resolve it by adding the new caller to the table without deciding whether it
must instead route through plan validation.
"""

from __future__ import annotations

import ast
import os

import torch_spyre

# The C++ launch's module-level name, as imported at call sites.
_LAUNCH_FUNCTION_NAME = "launch_kernel_from_bytes"


class LaunchBypassRouteError(AssertionError):
    """An unclassified caller of the direct C++ launch route."""

    # Reason code, in the repo's SENTIENT-LAUNCH-ABI family, naming this
    # contract: a caller of launch_kernel_from_bytes outside the sanctioned
    # table is a new bypass route around plan validation.
    reason_code = "SENTIENT-LAUNCH-ABI-UNCLASSIFIED-CXX-LAUNCH-CALLER"


def _package_root() -> str:
    return os.path.dirname(os.path.abspath(torch_spyre.__file__))


def _enumerate_cxx_launch_call_sites():
    """Every call of ``launch_kernel_from_bytes(...)`` under torch_spyre/.

    Walks the package source (skipping this test directory) and finds call
    nodes whose callee name is the C++ launch, however it was imported —
    a direct ``launch_kernel_from_bytes(...)`` or through a module alias.
    Definitions (``def launch_kernel_from_bytes`` in ``_C.py``, whose body
    only ``require_runtime``s) and doc/comment mentions are not calls.
    """
    root = _package_root()
    sites = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        if os.path.commonpath([dirpath, os.path.join(root, "tests")]) == os.path.join(
            root, "tests"
        ):
            continue
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == _LAUNCH_FUNCTION_NAME
                ):
                    sites.append((path, node.lineno))
    return sites


# The sanctioned-caller table. Keys are package-relative paths; values state
# the classification and its justification, so a reader can audit each entry
# against the deferral record that grants it.
_SANCTIONED_CALLERS = {
    "rng.py": (
        "plan-validating caller — the T=1 raw-state KAT launch derives an "
        "XLAT relocation plan beside the binary (build_xlat_relocation_plan on "
        "the SAME scheduled program) and gates the [params, out] binding with "
        "validate_positional_launch before the C++ call (T-LR-2B); the frozen "
        "2-segment ABI (segment 0 = params, segment 1 = raw output) is the "
        "rng_kat contract"
    ),
    "ops/softmax.py": (
        "plan-validating caller — the tensor list is derived from the "
        "named tensors_by_symbol mapping and gated by "
        "validate_positional_launch before the C++ call (T-LR-2B)"
    ),
    "ops/embedding.py": (
        "recorded planless legacy caller — embedding binaries carry no "
        "relocation plan (IBR-indirect addressing, recorded deferral), "
        "so the positional call binds nothing a plan would describe"
    ),
    "ops/attention.py": (
        "SPLIT classification (attention-aten-exposure): the CONTAINER "
        "path is a plan-validating caller — the 7-tensor list is gated by "
        "validate_positional_launch with named specs derived from the "
        "runtime allocations and roles read from the container dispatch's "
        "own segment classification (the softmax T-LR-2B pattern, "
        "_launch_container_validated); the HOST_SERIAL path is a recorded "
        "planless legacy caller — the multiblock owner's per-phase "
        "binaries (q_scale, kv_paged_read ingress, SP1 matmuls, the "
        "softmax bundle) are compiled via compile_scheduled_program and "
        "carry no relocation plan (the a4 KAT's witnessed launch form, "
        "recorded deferral until per-phase plans exist)"
    ),
}


def test_every_cxx_launch_call_site_is_sanctioned():
    sites = _enumerate_cxx_launch_call_sites()
    assert sites, (
        "enumeration found ZERO call sites of launch_kernel_from_bytes "
        "under torch_spyre/ — the walk is broken, not the package "
        "planless; fix the enumeration before trusting this guard"
    )
    root = _package_root()
    violations = []
    for path, lineno in sites:
        rel = os.path.relpath(path, root)
        if rel not in _SANCTIONED_CALLERS:
            violations.append(f"{rel}:{lineno}")
    assert not violations, (
        f"{LaunchBypassRouteError.reason_code}: direct C++ launch route "
        f"gained unclassified callers: {violations}. Each caller of "
        f"launch_kernel_from_bytes must either route through plan "
        f"validation (the ops/softmax.py named-mapping pattern) or be a "
        f"recorded planless legacy caller in _SANCTIONED_CALLERS with a "
        f"deferral-record justification. Adding a caller to the table "
        f"without making that decision opens a bypass route around the "
        f"launch ABI."
    )


def test_sanctioned_table_names_only_real_callers():
    # The table must not rot: every entry must correspond to at least one
    # LIVE call site. When a sanctioned caller is deleted (or stops calling
    # the C++ launch), its table entry becomes a stale grant that a future
    # re-added file could silently inherit.
    sites = _enumerate_cxx_launch_call_sites()
    root = _package_root()
    live = {os.path.relpath(path, root) for path, _ in sites}
    stale = [rel for rel in _SANCTIONED_CALLERS if rel not in live]
    assert not stale, (
        "stale sanctioned-caller entries (no live call site): "
        f"{stale} — remove them so the table names only real callers"
    )
