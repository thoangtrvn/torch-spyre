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

# Import torch before any torch_spyre submodule. PyTorch discovers torch_spyre as
# a backend entrypoint during its own import; reversing this order makes that
# entrypoint re-enter a partially initialized torch_spyre._inductor package.
import torch  # noqa: F401

from torch_spyre._inductor.descriptor_route import (
    DescriptorNegationRouteFacts,
    DescriptorUnaryRouteFacts,
    descriptor_negation_route_eligibility,
    descriptor_relu_route_eligibility,
    descriptor_unary_route_eligibility,
)


_CANONICAL_SOURCE = """
def triton_unk_fused_neg_0(in_ptr0, out_ptr0, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    source_desc = tl.make_tensor_descriptor(
        in_ptr0,
        shape=[512, 128],
        strides=[128, 1],
        block_shape=[(127 + XBLOCK) // 128, 128 * (128 <= XBLOCK) + XBLOCK * (XBLOCK < 128)],
    )
    target_desc = tl.make_tensor_descriptor(
        out_ptr0,
        shape=[512, 128],
        strides=[128, 1],
        block_shape=[(127 + XBLOCK) // 128, 128 * (128 <= XBLOCK) + XBLOCK * (XBLOCK < 128)],
    )
    tmp0 = tl.reshape(source_desc.load([xoffset // 128, xoffset % 128]), [XBLOCK])
    tmp1 = -tmp0
    target_desc.store(
        [xoffset // 128, xoffset % 128],
        tl.reshape(tmp1, [(127 + XBLOCK) // 128, 128 * (128 <= XBLOCK) + XBLOCK * (XBLOCK < 128)]),
    )
"""

_RELU_SOURCE = _CANONICAL_SOURCE.replace(
    "def triton_unk_fused_neg_0",
    "def triton_unk_fused_relu_0",
).replace(
    "tmp1 = -tmp0",
    "zero = 0.0\n    tmp1 = tl.maximum(tmp0, zero, fast_math=(\"nnan\",))",
)

_REAL_GENERATED_RELU_SOURCE = _CANONICAL_SOURCE.replace(
    "def triton_unk_fused_neg_0",
    "def triton_unk_fused_relu_0",
).replace(
    "tmp1 = -tmp0",
    "zero = tl.full([1], 0, tl.int32)\n"
    "    tmp1 = triton_helpers.maximum(zero, tmp0)",
).replace(
    "tl.reshape(tmp1, [(127 + XBLOCK) // 128, "
    "128 * (128 <= XBLOCK) + XBLOCK * (XBLOCK < 128)])",
    "tl.reshape(tl.broadcast_to(tmp1, [XBLOCK]), [(127 + XBLOCK) // 128, "
    "128 * (128 <= XBLOCK) + XBLOCK * (XBLOCK < 128)]).to(tl.float16)",
)


def test_descriptor_route_eligibility_refuses_raw_pointer_fallback():
    source = """
def triton_unk_fused_neg_0(in_ptr0, out_ptr0, XBLOCK: tl.constexpr):
    value = tl.load(in_ptr0 + tl.arange(0, XBLOCK))
    tl.store(out_ptr0 + tl.arange(0, XBLOCK), -value)
"""

    facts, problems = descriptor_negation_route_eligibility(
        source,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts is None
    assert any("raw-pointer" in problem for problem in problems)
    assert any("two tensor descriptors" in problem for problem in problems)


def test_descriptor_route_eligibility_recovers_canonical_contract_from_structure():
    facts, problems = descriptor_negation_route_eligibility(
        _CANONICAL_SOURCE,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts == DescriptorNegationRouteFacts(
        shape=(512, 128),
        strides=(128, 1),
        block_shape=(1, 64),
        problem_dimensions=(512, 128, 1),
    )
    assert problems == ()


def test_descriptor_route_eligibility_refuses_swapped_descriptor_direction():
    source = (
        _CANONICAL_SOURCE
        .replace("        in_ptr0,\n        shape=[512, 128],", "        swapped_ptr,\n        shape=[512, 128],", 1)
        .replace("        out_ptr0,\n        shape=[512, 128],", "        in_ptr0,\n        shape=[512, 128],", 1)
        .replace("        swapped_ptr,\n        shape=[512, 128],", "        out_ptr0,\n        shape=[512, 128],", 1)
    )

    facts, problems = descriptor_negation_route_eligibility(
        source,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts is None
    assert "descriptor load must read from the first kernel pointer argument" in problems
    assert "descriptor store must write to the second kernel pointer argument" in problems


def test_descriptor_route_eligibility_refuses_wrong_descriptor_contract():
    source = _CANONICAL_SOURCE.replace("strides=[128, 1]", "strides=[64, 1]")

    facts, problems = descriptor_negation_route_eligibility(
        source,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts is None
    assert any("strides must be (128, 1)" in problem for problem in problems)


def test_descriptor_route_eligibility_refuses_disconnected_negation():
    source = _CANONICAL_SOURCE.replace("tmp1 = -tmp0", "tmp1 = -other")

    facts, problems = descriptor_negation_route_eligibility(
        source,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts is None
    assert "unary negation does not consume the descriptor load" in problems


def test_descriptor_route_eligibility_refuses_store_not_fed_by_negation():
    source = _CANONICAL_SOURCE.replace("tl.reshape(tmp1,", "tl.reshape(tmp0,")

    facts, problems = descriptor_negation_route_eligibility(
        source,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts is None
    assert "descriptor store does not consume the unary negation" in problems


def test_descriptor_route_eligibility_refuses_unresolved_block_shape():
    facts, problems = descriptor_negation_route_eligibility(
        _CANONICAL_SOURCE,
        constexprs={},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts is None
    assert any("block_shape must be (1, 64)" in problem for problem in problems)


def test_descriptor_route_eligibility_refuses_wrong_frontend_dtype():
    facts, problems = descriptor_negation_route_eligibility(
        _CANONICAL_SOURCE,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*bf16", "out_ptr0": "*bf16"},
    )

    assert facts is None
    assert "descriptor negation route requires two DL16 frontend pointer arguments" in problems


def test_relu_route_eligibility_recovers_exact_operation_contract():
    facts, problems = descriptor_relu_route_eligibility(
        _RELU_SOURCE,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts == DescriptorUnaryRouteFacts(
        shape=(512, 128),
        strides=(128, 1),
        block_shape=(1, 64),
        problem_dimensions=(512, 128, 1),
        operation="descriptor_relu",
    )
    assert problems == ()


def test_relu_route_accepts_direct_qualified_maximum_form():
    source = _RELU_SOURCE.replace(
        "zero = 0.0",
        "zero = tl.zeros([64], tl.float16)",
    ).replace(
        'tl.maximum(tmp0, zero, fast_math=("nnan",))',
        "tl.maximum(tmp0, zero, tl.PropagateNan.ALL)",
    )

    facts, problems = descriptor_relu_route_eligibility(
        source,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts is not None
    assert problems == ()


def test_relu_route_accepts_real_generated_helper_form():
    facts, problems = descriptor_relu_route_eligibility(
        _REAL_GENERATED_RELU_SOURCE,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts == DescriptorUnaryRouteFacts(
        shape=(512, 128),
        strides=(128, 1),
        block_shape=(1, 64),
        problem_dimensions=(512, 128, 1),
        operation="descriptor_relu",
    )
    assert problems == ()


def test_relu_route_refuses_reversed_operands_negative_zero_and_nsz():
    mutations = (
        (
            _RELU_SOURCE.replace("tl.maximum(tmp0, zero", "tl.maximum(zero, tmp0"),
            "data operand must consume only the descriptor load",
        ),
        (
            _RELU_SOURCE.replace("zero = 0.0", "zero = -0.0"),
            "zero operand must be positive zero",
        ),
        (
            _RELU_SOURCE.replace('(\"nnan\",)', '(\"nnan\", \"nsz\")'),
            "fastmath must be exactly nnan without nsz",
        ),
        (
            _RELU_SOURCE.replace(
                'tl.maximum(tmp0, zero, fast_math=("nnan",))',
                "tl.maximum(tmp0, zero, tl.PropagateNan.NONE)",
            ),
            "must use the qualified propagating form",
        ),
        (
            _REAL_GENERATED_RELU_SOURCE.replace(
                "triton_helpers.maximum(zero, tmp0)",
                "triton_helpers.maximum(tmp0, zero)",
            ),
            "data operand must consume only the descriptor load",
        ),
        (
            _REAL_GENERATED_RELU_SOURCE.replace(
                "tl.full([1], 0, tl.int32)",
                "tl.full([1], -0.0, tl.int32)",
            ),
            "zero operand must be positive zero",
        ),
    )
    for source, expected_problem in mutations:
        facts, problems = descriptor_relu_route_eligibility(
            source,
            constexprs={"XBLOCK": 64},
            kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
        )
        assert facts is None
        assert any(expected_problem in problem for problem in problems)


def test_relu_route_refuses_arithmetic_before_maximum():
    source = _RELU_SOURCE.replace(
        "tl.maximum(tmp0, zero",
        "tl.maximum(tmp0 + 1.0, zero",
    )

    facts, problems = descriptor_relu_route_eligibility(
        source,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts is None
    assert any("shape-only wrappers" in problem for problem in problems)


def test_relu_route_refuses_arithmetic_after_maximum():
    source = _RELU_SOURCE.replace(
        'tmp1 = tl.maximum(tmp0, zero, fast_math=("nnan",))',
        'maximum = tl.maximum(tmp0, zero, fast_math=("nnan",))\n    tmp1 = maximum + 1.0',
    )

    facts, problems = descriptor_relu_route_eligibility(
        source,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts is None
    assert any("shape-only wrappers" in problem for problem in problems)


def test_relu_route_refuses_unsupported_transcendental_between_load_and_maximum():
    source = _RELU_SOURCE.replace(
        "tl.maximum(tmp0, zero",
        "tl.maximum(tl.exp(tmp0), zero",
    )

    facts, problems = descriptor_relu_route_eligibility(
        source,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts is None
    assert any("shape-only wrappers" in problem for problem in problems)


def test_operation_dispatch_refuses_cross_wiring():
    facts, problems = descriptor_unary_route_eligibility(
        "descriptor_relu",
        _CANONICAL_SOURCE,
        constexprs={"XBLOCK": 64},
        kernel_signature={"in_ptr0": "*fp16", "out_ptr0": "*fp16"},
    )

    assert facts is None
    assert "generated Triton kernel must contain exactly one ReLU maximum" in problems
