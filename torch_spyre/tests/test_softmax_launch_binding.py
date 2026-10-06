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
"""Softmax op launch-binding plan consumption (T-LR-2B part 3).

``torch_spyre.ops.softmax`` gates its launch through codegen's
``validate_positional_launch`` with NAMED tensor specs derived from the
actual runtime allocations — {arg0: workspace, arg1: input, arg2: output}.
The declared roles are the op's own by-construction knowledge (it knows
which tensor is its scratch, input and output); a generic launcher could
only echo the plan's access modes (tautology), which is why role validation
lives at this surface and NOT in the triton launcher.

These tests exercise the gate logic off-box against a REAL plan from
``build_softmax_launch_plan`` (the same cached call the op makes). The spec
derivation is the op's OWN module-level helper (``_softmax_launch_specs``),
not a test-side re-implementation — a drifted derivation in softmax.py must
FAIL here, not silently pass beside a parallel copy in the test file.

The shape is derived from codegen's ``SOFTMAX_WITNESSED_SHAPES`` (the sole
source of truth for which shapes route), never a literal pinned here.
"""
import pytest


def _witnessed_shape():
    """One admitted (M, N, C, dtype_bits) from codegen's witnessed set.

    Derived from ``SOFTMAX_WITNESSED_SHAPES`` so this test can never pin a
    shape codegen has since removed from the routed envelope.
    """
    try:
        from sentient_codegen.encoder.softmax_dispatch import (
            SOFTMAX_WITNESSED_SHAPES,
        )
    except ImportError:
        pytest.skip("sentient_codegen not importable (source env.sh)")
    # Deterministic pick: the original witnessed (1024, 64) point.
    for m, n, c, dtype_bits in sorted(SOFTMAX_WITNESSED_SHAPES):
        if (m, n) == (1024, 64):
            return m, n, c, dtype_bits
    m, n, c, dtype_bits = sorted(SOFTMAX_WITNESSED_SHAPES)[0]
    return m, n, c, dtype_bits


def _build_plan(m, n, c, dtype_bits):
    try:
        from sentient_codegen.encoder.softmax_dispatch import (
            build_softmax_launch_plan,
        )
    except ImportError:
        pytest.skip("sentient_codegen not importable (source env.sh)")
    return build_softmax_launch_plan(m, n, num_cores=c, dtype_bits=dtype_bits)


class _FakeTensor:
    """Minimal numel/element_size carrier — the helper reads nothing else."""

    def __init__(self, numel, element_size):
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


def _op_tensors(m, n, dtype_bits, plan):
    """The op's own allocations, built exactly as softmax.py builds them."""
    elem_bytes = dtype_bits // 8
    scratch = _FakeTensor(plan.scratch_sticks * plan.elems_per_stick, elem_bytes)
    xc = _FakeTensor(m * n, elem_bytes)
    out = _FakeTensor(m * n, elem_bytes)
    return scratch, xc, out


def test_op_spec_helper_derives_real_allocation_extents():
    # The op's module-level spec derivation (relocated for off-box
    # testability — the _resolve_num_cores rationale): extents must be the
    # runtime allocation sizes, with the op's own role knowledge.
    from torch_spyre.ops.softmax import (
        _softmax_launch_specs,
        _softmax_launch_tensors_by_symbol,
    )

    m, n, c, dtype_bits = _witnessed_shape()
    plan = _build_plan(m, n, c, dtype_bits)
    scratch, xc, out = _op_tensors(m, n, dtype_bits, plan)
    specs = _softmax_launch_specs(
        _softmax_launch_tensors_by_symbol(scratch, xc, out)
    )
    assert specs == {
        "arg0": (scratch.numel() * scratch.element_size(), "workspace"),
        "arg1": (xc.numel() * xc.element_size(), "input"),
        "arg2": (out.numel() * out.element_size(), "output"),
    }


def test_admitted_softmax_specs_pass_the_consume_gate():
    from sentient_codegen.backend.launch_abi import validate_positional_launch
    from torch_spyre.ops.softmax import (
        _softmax_launch_specs,
        _softmax_launch_tensors_by_symbol,
    )

    m, n, c, dtype_bits = _witnessed_shape()
    plan = _build_plan(m, n, c, dtype_bits)
    tensors = _op_tensors(m, n, dtype_bits, plan)
    specs = _softmax_launch_specs(_softmax_launch_tensors_by_symbol(*tensors))
    result = validate_positional_launch(plan.xlat_plan, plan.binary, specs)
    assert result.template_fingerprint == plan.xlat_plan.template_fingerprint
    assert sorted(result.validated_symbol_ids) == ["arg0", "arg1", "arg2"]


def test_permuted_input_output_roles_refuse():
    # x and out have IDENTICAL byte footprints, so extent coverage cannot
    # catch this swap — only the role gate (op-side knowledge) can. The
    # refusal fires on a PERMUTED ROLES TABLE (the labels swapped between
    # the equal-extent symbols): that is what the gate compares against the
    # plan's mode-derived access modes.
    from sentient_codegen.backend.launch_abi import (
        LaunchAbiError,
        validate_positional_launch,
    )
    from torch_spyre.ops.softmax import (
        _softmax_launch_specs,
        _softmax_launch_tensors_by_symbol,
    )

    m, n, c, dtype_bits = _witnessed_shape()
    plan = _build_plan(m, n, c, dtype_bits)
    tensors = _op_tensors(m, n, dtype_bits, plan)
    specs = _softmax_launch_specs(_softmax_launch_tensors_by_symbol(*tensors))
    # Swap the role LABELS between the equal-extent input/output symbols.
    specs["arg1"], specs["arg2"] = (
        (specs["arg2"][0], "output"),
        (specs["arg1"][0], "input"),
    )
    with pytest.raises(LaunchAbiError):
        validate_positional_launch(plan.xlat_plan, plan.binary, specs)


def test_equal_extent_tensor_swap_is_structurally_not_gate_detectable():
    # NEGATIVE-SPACE PIN (honest limit of the seam): swapping the INPUT and
    # OUTPUT TENSORS while keeping the canonical role labels does NOT
    # refuse — the two allocations have identical byte footprints, so no
    # numeric gate at this boundary can tell them apart, and role labels
    # are correct by construction. This permutation is closed STRUCTURALLY,
    # not by the gate: softmax.py derives both the specs and the C++
    # positional list from ONE named mapping, so a second, permuted
    # ordering cannot exist to be drifted (see
    # test_the_cxx_positional_list_is_derived_from_the_named_mapping). If
    # this test ever FAILS, the gate gained a check this seam never
    # promised — re-derive the honest-checker split before trusting it.
    from sentient_codegen.backend.launch_abi import validate_positional_launch
    from torch_spyre.ops.softmax import (
        _softmax_launch_specs,
        _softmax_launch_tensors_by_symbol,
    )

    m, n, c, dtype_bits = _witnessed_shape()
    plan = _build_plan(m, n, c, dtype_bits)
    scratch, xc, out = _op_tensors(m, n, dtype_bits, plan)
    specs = _softmax_launch_specs(
        _softmax_launch_tensors_by_symbol(scratch, out, xc)
    )
    # Must NOT raise.
    validate_positional_launch(plan.xlat_plan, plan.binary, specs)


def test_missing_spec_refuses():
    from sentient_codegen.backend.launch_abi import (
        LaunchAbiError,
        validate_positional_launch,
    )
    from torch_spyre.ops.softmax import (
        _softmax_launch_specs,
        _softmax_launch_tensors_by_symbol,
    )

    m, n, c, dtype_bits = _witnessed_shape()
    plan = _build_plan(m, n, c, dtype_bits)
    tensors = _op_tensors(m, n, dtype_bits, plan)
    specs = _softmax_launch_specs(_softmax_launch_tensors_by_symbol(*tensors))
    del specs["arg0"]
    with pytest.raises(LaunchAbiError):
        validate_positional_launch(plan.xlat_plan, plan.binary, specs)


def test_extra_spec_refuses():
    from sentient_codegen.backend.launch_abi import (
        LaunchAbiError,
        validate_positional_launch,
    )
    from torch_spyre.ops.softmax import (
        _softmax_launch_specs,
        _softmax_launch_tensors_by_symbol,
    )

    m, n, c, dtype_bits = _witnessed_shape()
    plan = _build_plan(m, n, c, dtype_bits)
    tensors = _op_tensors(m, n, dtype_bits, plan)
    specs = _softmax_launch_specs(_softmax_launch_tensors_by_symbol(*tensors))
    specs["arg3"] = (128, "input")
    with pytest.raises(LaunchAbiError):
        validate_positional_launch(plan.xlat_plan, plan.binary, specs)


def test_tampered_binary_refuses_on_fingerprint_first():
    from sentient_codegen.backend.launch_abi import (
        LaunchAbiError,
        validate_positional_launch,
    )
    from torch_spyre.ops.softmax import (
        _softmax_launch_specs,
        _softmax_launch_tensors_by_symbol,
    )

    m, n, c, dtype_bits = _witnessed_shape()
    plan = _build_plan(m, n, c, dtype_bits)
    tensors = _op_tensors(m, n, dtype_bits, plan)
    specs = _softmax_launch_specs(_softmax_launch_tensors_by_symbol(*tensors))
    tampered = bytearray(plan.binary)
    tampered[0] ^= 0xFF
    with pytest.raises(LaunchAbiError, match="fingerprint"):
        validate_positional_launch(plan.xlat_plan, bytes(tampered), specs)


def test_op_launch_spec_derivation_matches_the_plan_requirements():
    # The op derives extents from the allocations it is about to launch;
    # those must satisfy the plan's own window requirements, or every
    # launch would refuse. Guards the derivation against drift between
    # the op's allocation shapes and the plan's declared extents.
    from torch_spyre.ops.softmax import (
        _softmax_launch_specs,
        _softmax_launch_tensors_by_symbol,
    )

    m, n, c, dtype_bits = _witnessed_shape()
    plan = _build_plan(m, n, c, dtype_bits)
    tensors = _op_tensors(m, n, dtype_bits, plan)
    specs = _softmax_launch_specs(_softmax_launch_tensors_by_symbol(*tensors))
    by_symbol = {s.symbol_id: s for s in plan.xlat_plan.symbols}
    for symbol_id, (extent, _role) in specs.items():
        assert extent >= by_symbol[symbol_id].required_extent_bytes, (
            f"{symbol_id}: derived allocation {extent} bytes does not "
            f"cover the plan's required "
            f"{by_symbol[symbol_id].required_extent_bytes} bytes"
        )


def test_the_cxx_positional_list_is_derived_from_the_named_mapping():
    # Defect-class pin: the C++ positional list and the named specs must be
    # derived from the SAME mapping (the single source of truth), so a
    # hand-maintained second ordering cannot exist. Verifies the op's own
    # derivation produces [scratch, xc, out] in the plan's symbol order.
    from torch_spyre.ops.softmax import _softmax_launch_tensors_by_symbol

    m, n, c, dtype_bits = _witnessed_shape()
    plan = _build_plan(m, n, c, dtype_bits)
    tensors = _op_tensors(m, n, dtype_bits, plan)
    mapping = _softmax_launch_tensors_by_symbol(*tensors)
    cxx_list = [mapping[s.symbol_id] for s in plan.xlat_plan.symbols]
    assert cxx_list == [tensors[0], tensors[1], tensors[2]]


# ---------------------------------------------------------------------------
# T-LR-2C: route-aware positional-arg permutation fuzz (torch-op route)
# ---------------------------------------------------------------------------

def _witnessed_shapes():
    """The plan-bearing population, DERIVED from codegen's authority.

    ``SOFTMAX_WITNESSED_SHAPES`` is the only plan-bearing binary set today
    (embedding is IBR-indirected, matmul qginit bundles are plan-less by
    recorded deferral). A loud skip — never silence — when codegen is not
    importable: an off-box run must report the missing population, not
    masquerade as coverage.
    """
    try:
        from sentient_codegen.encoder.softmax_dispatch import (
            SOFTMAX_WITNESSED_SHAPES,
        )
    except ImportError:
        pytest.skip(
            "sentient_codegen not importable (source env.sh) — the "
            "plan-bearing permutation population cannot be derived; "
            "skipping the torch-op-route fuzz LOUDLY rather than "
            "reporting untested permutations as green."
        )
    return sorted(SOFTMAX_WITNESSED_SHAPES)


@pytest.mark.parametrize("m,n,c,dtype_bits", _witnessed_shapes())
def test_softmax_permutation_fuzz_through_the_op_gate(m, n, c, dtype_bits):
    # THE T-LR-2C TORCH-OP-ROUTE FUZZ. Every permutation of the positional
    # tensor list, exercised the way a DRIFTED caller would hit the op's
    # gate: the op's own module-level helpers build specs from the PERMUTED
    # tensors, with the op's canonical role labels (the mapping
    # ``_softmax_launch_tensors_by_symbol`` is the single named source of
    # truth — a permuted CALL is structurally impossible today, and that
    # structural closure is itself pinned by
    # ``test_the_cxx_positional_list_is_derived_from_the_named_mapping``).
    # The per-permutation contract is bidirectional:
    #   * identity and the equal-extent x/out swap PASS BY DESIGN — the
    #     pinned negative space: identical byte footprints mean no numeric
    #     gate at this boundary can tell them apart, and the canonical role
    #     labels are correct for the tensors they name (the swap is closed
    #     STRUCTURALLY by the single mapping, not by the gate);
    #   * every permutation that moves the (strictly larger) scratch tensor
    #     REFUSES on window extent;
    #   * a permutation refusing where the contract pins a pass, or passing
    #     where it must refuse, is a failure either way.
    import itertools

    from sentient_codegen.backend.launch_abi import (
        LaunchAbiError,
        validate_positional_launch,
    )
    from torch_spyre.ops.softmax import (
        _softmax_launch_specs,
        _softmax_launch_tensors_by_symbol,
    )

    plan = _build_plan(m, n, c, dtype_bits)
    canonical = _op_tensors(m, n, dtype_bits, plan)  # (scratch, xc, out)
    symbols = list(plan.xlat_plan.symbols)
    assert [s.symbol_id for s in symbols] == ["arg0", "arg1", "arg2"]
    requirements = {s.symbol_id: s.required_extent_bytes for s in symbols}

    for perm in itertools.permutations(range(len(canonical))):
        # The permuted positional list: canonical tensor perm[i] lands in
        # slot i. Specs are derived through the OP's own helpers from the
        # named mapping the permuted list implies — the drifted-caller
        # shape: same helpers, same roles, different tensor↔slot pairing.
        mapping = {
            symbols[slot].symbol_id: canonical[perm[slot]]
            for slot in range(len(canonical))
        }
        specs = _softmax_launch_specs(mapping)
        # Derive the prediction from the actual tensor footprints vs the
        # plan's requirements — the same fact the extent gate compares.
        must_refuse = any(
            specs[symbol_id][0] < requirements[symbol_id]
            for symbol_id in requirements
        )
        if must_refuse:
            with pytest.raises(LaunchAbiError):
                validate_positional_launch(plan.xlat_plan, plan.binary, specs)
        else:
            # PINNED NEGATIVE SPACE: must NOT raise (identity and the
            # equal-extent x/out swap). If it does, the gate gained a check
            # this seam never promised.
            validate_positional_launch(plan.xlat_plan, plan.binary, specs)
