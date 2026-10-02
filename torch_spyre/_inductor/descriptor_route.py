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

"""Import-light classifiers for descriptor route option admission."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass


_EXPECTED_SHAPE = (512, 128)
_EXPECTED_STRIDES = (128, 1)
_EXPECTED_BLOCK_SHAPE = (1, 64)


@dataclass(frozen=True)
class DescriptorUnaryRouteFacts:
    shape: tuple[int, int]
    strides: tuple[int, int]
    block_shape: tuple[int, int]
    problem_dimensions: tuple[int, int, int]
    operation: str = "descriptor_negation"


# Preserve the public negation fact name and equality behavior.
DescriptorNegationRouteFacts = DescriptorUnaryRouteFacts


def descriptor_unary_route_eligibility(
    operation: str,
    source: str,
    *,
    constexprs: Mapping[str, object] | None = None,
    kernel_signature: Mapping[str, object] | None = None,
) -> tuple[DescriptorUnaryRouteFacts | None, tuple[str, ...]]:
    """Recover one exact descriptor-unary contract from generated syntax."""

    validators = {
        "descriptor_negation": _descriptor_negation_structure_problems,
        "descriptor_relu": _descriptor_relu_structure_problems,
    }
    validate = validators.get(operation)
    if validate is None:
        return None, ("unsupported descriptor normal route operation",)
    problems = validate(source, constexprs or {}, kernel_signature or {})
    if problems:
        return None, problems
    return (
        DescriptorUnaryRouteFacts(
            operation=operation,
            shape=_EXPECTED_SHAPE,
            strides=_EXPECTED_STRIDES,
            block_shape=_EXPECTED_BLOCK_SHAPE,
            problem_dimensions=(*_EXPECTED_SHAPE, 1),
        ),
        (),
    )


def descriptor_negation_route_eligibility(
    source: str,
    *,
    constexprs: Mapping[str, object] | None = None,
    kernel_signature: Mapping[str, object] | None = None,
) -> tuple[DescriptorNegationRouteFacts | None, tuple[str, ...]]:
    """Recover the fixed route contract from generated syntax and dataflow."""

    problems = _descriptor_negation_structure_problems(
        source,
        constexprs or {},
        kernel_signature or {},
    )
    if problems:
        return None, problems
    facts, problems = descriptor_unary_route_eligibility(
        "descriptor_negation",
        source,
        constexprs=constexprs,
        kernel_signature=kernel_signature,
    )
    return facts, problems


def descriptor_relu_route_eligibility(
    source: str,
    *,
    constexprs: Mapping[str, object] | None = None,
    kernel_signature: Mapping[str, object] | None = None,
) -> tuple[DescriptorUnaryRouteFacts | None, tuple[str, ...]]:
    """Recover the fixed exact-ReLU route contract."""

    return descriptor_unary_route_eligibility(
        "descriptor_relu",
        source,
        constexprs=constexprs,
        kernel_signature=kernel_signature,
    )


def _descriptor_negation_structure_problems(
    source: str,
    constexprs: Mapping[str, object],
    kernel_signature: Mapping[str, object],
) -> tuple[str, ...]:
    try:
        module = ast.parse(source)
    except (SyntaxError, TypeError) as exc:
        return (f"generated Triton source is not valid Python: {exc}",)

    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        return ("generated Triton source must contain exactly one kernel function",)
    function = functions[0]
    assignments = {
        target.id: statement.value
        for statement in function.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    raw_accesses = [
        call
        for call in calls
        if _qualified_name(call.func) in {"tl.load", "tl.store"}
    ]
    descriptors = [
        call for call in calls if _qualified_name(call.func) == "tl.make_tensor_descriptor"
    ]
    descriptor_names = {
        name
        for name, expression in assignments.items()
        if expression in descriptors
    }
    descriptor_loads = [
        call
        for call in calls
        if _descriptor_method_receiver(call, "load") in descriptor_names
    ]
    descriptor_stores = [
        call
        for call in calls
        if _descriptor_method_receiver(call, "store") in descriptor_names
    ]
    negations = [node for node in ast.walk(function) if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)]

    problems: list[str] = []
    pointer_types = tuple(
        kernel_signature.get(argument.arg)
        for argument in function.args.args[:2]
    )
    if pointer_types != ("*fp16", "*fp16"):
        problems.append(
            "descriptor negation route requires two DL16 frontend pointer arguments"
        )
    if raw_accesses:
        problems.append("generated Triton kernel contains raw-pointer load/store access")
    if len(descriptors) != 2:
        problems.append(
            "generated Triton kernel must construct exactly two tensor descriptors"
        )
    if len(descriptor_loads) != 1:
        problems.append("generated Triton kernel must contain exactly one descriptor load")
    if len(descriptor_stores) != 1:
        problems.append("generated Triton kernel must contain exactly one descriptor store")
    if len(negations) != 1:
        problems.append("generated Triton kernel must contain exactly one unary negation")

    for descriptor in descriptors:
        descriptor_problems = _descriptor_contract_problems(descriptor, constexprs)
        problems.extend(descriptor_problems)

    if len(descriptor_loads) == 1 and len(descriptor_stores) == 1:
        load_receiver = _descriptor_method_receiver(descriptor_loads[0], "load")
        store_receiver = _descriptor_method_receiver(descriptor_stores[0], "store")
        if load_receiver == store_receiver:
            problems.append("descriptor load and store must use distinct descriptors")

        function_args = tuple(argument.arg for argument in function.args.args)
        load_descriptor = assignments.get(load_receiver) if load_receiver else None
        store_descriptor = assignments.get(store_receiver) if store_receiver else None
        load_base = _descriptor_base_name(load_descriptor)
        store_base = _descriptor_base_name(store_descriptor)
        if len(function_args) < 2 or load_base != function_args[0]:
            problems.append(
                "descriptor load must read from the first kernel pointer argument"
            )
        if len(function_args) < 2 or store_base != function_args[1]:
            problems.append(
                "descriptor store must write to the second kernel pointer argument"
            )

    if len(descriptor_loads) == 1 and len(negations) == 1:
        if not _depends_on(negations[0].operand, descriptor_loads[0], assignments):
            problems.append("unary negation does not consume the descriptor load")
    if len(descriptor_stores) == 1 and len(negations) == 1:
        store = descriptor_stores[0]
        stored_value = store.args[-1] if store.args else None
        if stored_value is None or not _depends_on(stored_value, negations[0], assignments):
            problems.append("descriptor store does not consume the unary negation")

    return tuple(problems)


def _descriptor_relu_structure_problems(
    source: str,
    constexprs: Mapping[str, object],
    kernel_signature: Mapping[str, object],
) -> tuple[str, ...]:
    try:
        module = ast.parse(source)
    except (SyntaxError, TypeError) as exc:
        return (f"generated Triton source is not valid Python: {exc}",)

    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        return ("generated Triton source must contain exactly one kernel function",)
    function = functions[0]
    assignments = {
        target.id: statement.value
        for statement in function.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    descriptors = [
        call for call in calls if _qualified_name(call.func) == "tl.make_tensor_descriptor"
    ]
    descriptor_names = {
        name for name, expression in assignments.items() if expression in descriptors
    }
    descriptor_loads = [
        call for call in calls if _descriptor_method_receiver(call, "load") in descriptor_names
    ]
    descriptor_stores = [
        call for call in calls if _descriptor_method_receiver(call, "store") in descriptor_names
    ]
    raw_accesses = [
        call for call in calls if _qualified_name(call.func) in {"tl.load", "tl.store"}
    ]
    maximums = [
        call
        for call in calls
        if _qualified_name(call.func)
        in {"tl.maximum", "tl.maximumf", "triton_helpers.maximum"}
    ]
    negations = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
    ]

    problems: list[str] = []
    pointer_types = tuple(
        kernel_signature.get(argument.arg) for argument in function.args.args[:2]
    )
    if pointer_types != ("*fp16", "*fp16"):
        problems.append("descriptor ReLU route requires two DL16 frontend pointer arguments")
    if raw_accesses:
        problems.append("generated Triton kernel contains raw-pointer load/store access")
    if len(descriptors) != 2:
        problems.append("generated Triton kernel must construct exactly two tensor descriptors")
    if len(descriptor_loads) != 1:
        problems.append("generated Triton kernel must contain exactly one descriptor load")
    if len(descriptor_stores) != 1:
        problems.append("generated Triton kernel must contain exactly one descriptor store")
    if len(maximums) != 1:
        problems.append("generated Triton kernel must contain exactly one ReLU maximum")
    if negations:
        problems.append("generated Triton ReLU kernel must not contain unary negation")

    for descriptor in descriptors:
        problems.extend(_descriptor_contract_problems(descriptor, constexprs))

    if len(descriptor_loads) == 1 and len(descriptor_stores) == 1:
        load_receiver = _descriptor_method_receiver(descriptor_loads[0], "load")
        store_receiver = _descriptor_method_receiver(descriptor_stores[0], "store")
        if load_receiver == store_receiver:
            problems.append("descriptor load and store must use distinct descriptors")
        function_args = tuple(argument.arg for argument in function.args.args)
        load_descriptor = assignments.get(load_receiver) if load_receiver else None
        store_descriptor = assignments.get(store_receiver) if store_receiver else None
        if len(function_args) < 2 or _descriptor_base_name(load_descriptor) != function_args[0]:
            problems.append("descriptor load must read from the first kernel pointer argument")
        if len(function_args) < 2 or _descriptor_base_name(store_descriptor) != function_args[1]:
            problems.append("descriptor store must write to the second kernel pointer argument")

    if len(maximums) == 1:
        maximum = maximums[0]
        maximum_name = _qualified_name(maximum.func)
        data_operands = maximum.args[:2]
        expected_argument_count = 2 if maximum_name == "triton_helpers.maximum" else (2, 3)
        if (
            len(maximum.args) != expected_argument_count
            if isinstance(expected_argument_count, int)
            else len(maximum.args) not in expected_argument_count
        ):
            problems.append("ReLU maximum has an unsupported argument count")
        else:
            load_operands = tuple(
                index
                for index, operand in enumerate(data_operands)
                if len(descriptor_loads) == 1
                and _is_shape_only_path(operand, descriptor_loads[0], assignments)
            )
            zero_operands = tuple(
                index
                for index, operand in enumerate(data_operands)
                if _is_positive_zero(operand, assignments)
            )
            if maximum_name == "triton_helpers.maximum":
                expected_load_operands = (1,)
                expected_zero_operands = (0,)
            else:
                expected_load_operands = (0,)
                expected_zero_operands = (1,)
            if load_operands != expected_load_operands:
                problems.append(
                    "ReLU maximum data operand must consume only the descriptor "
                    "load through shape-only wrappers"
                )
            if zero_operands != expected_zero_operands:
                problems.append("ReLU maximum zero operand must be positive zero")
        if maximum_name != "triton_helpers.maximum":
            propagate_nan = _call_keyword_value(maximum, "propagate_nan")
            if propagate_nan is None and len(maximum.args) >= 3:
                propagate_nan = maximum.args[2]
            if propagate_nan is not None:
                if _qualified_name(propagate_nan) != "tl.PropagateNan.ALL":
                    problems.append(
                        "ReLU maximum must use the qualified propagating form"
                    )
            else:
                fastmath = _call_keyword_value(maximum, "fast_math")
                if fastmath is None:
                    fastmath = _call_keyword_value(maximum, "fastmath")
                if fastmath is None:
                    problems.append("ReLU maximum must declare exact nnan fastmath")
                else:
                    flags = _string_values(fastmath, assignments)
                    if flags != ("nnan",):
                        problems.append(
                            "ReLU maximum fastmath must be exactly nnan without nsz"
                        )

    if len(descriptor_stores) == 1 and len(maximums) == 1:
        stored_value = descriptor_stores[0].args[-1] if descriptor_stores[0].args else None
        if stored_value is None or not _is_shape_only_path(
            stored_value,
            maximums[0],
            assignments,
        ):
            problems.append(
                "descriptor store must consume only the ReLU maximum through "
                "shape-only wrappers"
            )

    return tuple(problems)


def _call_keyword_value(call: ast.Call, name: str) -> ast.AST | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _string_values(
    node: ast.AST,
    assignments: Mapping[str, ast.AST],
) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name) and node.id in assignments:
        return _string_values(assignments[node.id], assignments)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = tuple(
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        )
        return values if len(values) == len(node.elts) else None
    return None


def _is_positive_zero(
    node: ast.AST,
    assignments: Mapping[str, ast.AST],
) -> bool:
    if isinstance(node, ast.Name) and node.id in assignments:
        return _is_positive_zero(assignments[node.id], assignments)
    if isinstance(node, ast.Constant):
        if type(node.value) not in {int, float} or node.value != 0:
            return False
        if isinstance(node.value, float):
            import math

            return math.copysign(1.0, node.value) > 0
        return True
    if isinstance(node, ast.Call) and not node.keywords:
        function_name = _qualified_name(node.func)
        if (
            function_name == "tl.full"
            and len(node.args) == 3
            and _evaluate_int_sequence(node.args[0], {}) in {(), (1,), (64,)}
            and _is_positive_zero(node.args[1], assignments)
            and _qualified_name(node.args[2]) in {"tl.int32", "tl.float16"}
        ):
            return True
        if (
            function_name == "tl.zeros"
            and len(node.args) == 2
            and _evaluate_int_sequence(node.args[0], {}) == (64,)
            and _qualified_name(node.args[1]) == "tl.float16"
        ):
            return True
    return False


def _descriptor_contract_problems(
    descriptor: ast.Call,
    constexprs: Mapping[str, object],
) -> tuple[str, ...]:
    keywords = {keyword.arg: keyword.value for keyword in descriptor.keywords}
    expected = {
        "shape": _EXPECTED_SHAPE,
        "strides": _EXPECTED_STRIDES,
        "block_shape": _EXPECTED_BLOCK_SHAPE,
    }
    problems = []
    if len(descriptor.args) != 1:
        problems.append("tensor descriptor must have exactly one base pointer")
    for field, required in expected.items():
        value = keywords.get(field)
        observed = _evaluate_int_sequence(value, constexprs) if value is not None else None
        if observed != required:
            problems.append(
                f"tensor descriptor {field} must be {required!r}; observed {observed!r}"
            )
    return tuple(problems)


def _is_shape_only_path(
    expression: ast.AST,
    target: ast.AST,
    assignments: Mapping[str, ast.AST],
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Follow assignments and reshape calls only; reject all arithmetic/fused ops."""

    if expression is target:
        return True
    if isinstance(expression, ast.Name) and expression.id in assignments:
        if expression.id in seen:
            return False
        return _is_shape_only_path(
            assignments[expression.id],
            target,
            assignments,
            seen | {expression.id},
        )
    if isinstance(expression, ast.Call):
        function_name = _qualified_name(expression.func)
        if function_name in {"tl.reshape", "tl.broadcast_to"}:
            if len(expression.args) != 2 or expression.keywords:
                return False
            return _is_shape_only_path(
                expression.args[0],
                target,
                assignments,
                seen,
            )
        if (
            isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "to"
            and len(expression.args) == 1
            and not expression.keywords
            and _qualified_name(expression.args[0]) == "tl.float16"
        ):
            return _is_shape_only_path(
                expression.func.value,
                target,
                assignments,
                seen,
            )
    return False


def _depends_on(
    expression: ast.AST,
    target: ast.AST,
    assignments: Mapping[str, ast.AST],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if expression is target:
        return True
    if isinstance(expression, ast.Name) and expression.id in assignments:
        if expression.id in seen:
            return False
        return _depends_on(
            assignments[expression.id],
            target,
            assignments,
            seen | {expression.id},
        )
    return any(
        _depends_on(child, target, assignments, seen)
        for child in ast.iter_child_nodes(expression)
    )


def _descriptor_method_receiver(call: ast.Call, method: str) -> str | None:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != method:
        return None
    return _name(call.func.value)


def _descriptor_base_name(node: ast.AST | None) -> str | None:
    if (
        not isinstance(node, ast.Call)
        or _qualified_name(node.func) != "tl.make_tensor_descriptor"
        or len(node.args) != 1
    ):
        return None
    return _name(node.args[0])


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _evaluate_int_sequence(
    node: ast.AST,
    constexprs: Mapping[str, object],
) -> tuple[int, ...] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values = tuple(_evaluate_int(element, constexprs) for element in node.elts)
    if any(value is None for value in values):
        return None
    return tuple(int(value) for value in values)


def _evaluate_int(node: ast.AST, constexprs: Mapping[str, object]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int)):
        return int(node.value)
    if isinstance(node, ast.Name):
        value = constexprs.get(node.id)
        return int(value) if isinstance(value, (bool, int)) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _evaluate_int(node.operand, constexprs)
        if operand is None:
            return None
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_int(node.left, constexprs)
        right = _evaluate_int(node.right, constexprs)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv) and right != 0:
            return left // right
        if isinstance(node.op, ast.Mod) and right != 0:
            return left % right
        return None
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left = _evaluate_int(node.left, constexprs)
        right = _evaluate_int(node.comparators[0], constexprs)
        if left is None or right is None:
            return None
        operator = node.ops[0]
        if isinstance(operator, ast.Lt):
            return int(left < right)
        if isinstance(operator, ast.LtE):
            return int(left <= right)
        if isinstance(operator, ast.Gt):
            return int(left > right)
        if isinstance(operator, ast.GtE):
            return int(left >= right)
        if isinstance(operator, ast.Eq):
            return int(left == right)
    return None
