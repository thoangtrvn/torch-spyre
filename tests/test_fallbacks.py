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


import threading
import warnings

import torch
from torch.testing import assert_close
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
)
from torch_spyre.ops.fallbacks import (
    FallbackWarning,
    get_fallback_ledger,
    reset_fallback_ledger,
)


class Op(subtest):
    def __init__(self, name, fn, rtol=None, atol=None):
        super().__init__(self, name)
        self.fn = fn
        self.rtol = rtol
        self.atol = atol


class FactoryOp(Op):
    """
    Represents an operator that creates a new tensor without consuming an existing
    tensor as input.

    Examples:
       - torch.arange
       - torch.full
    """

    inputs = [(64,)]

    def __init__(self, name, fn, inputs=None, rtol=None, atol=None):
        super().__init__(name, fn, rtol, atol)
        self.inputs = inputs or type(self).inputs


class UnaryOp(Op):
    inputs = [torch.rand(64, dtype=torch.float16)]

    def __init__(self, name, fn, inputs=None, rtol=None, atol=None):
        super().__init__(name, fn, rtol, atol)
        self.inputs = inputs or type(self).inputs


_factory_ops = [
    FactoryOp("arange", torch.arange, [(64.0,), (1.0, 65.0), (0.0, 128.0, 2.0)]),
]

_unary_ops = [
    UnaryOp("sin", torch.sin),
    UnaryOp("cos", torch.cos),
]


@instantiate_parametrized_tests
class TestFallbacks(TestCase):
    def setUp(self):
        self.rtol = 1e-2
        self.atol = 1e-3
        self.dtype = torch.float16

        torch.random.manual_seed(42)

        warnings.simplefilter("ignore", FallbackWarning)

    def _assert_close(self, op, output_spyre, output_cpu):
        rtol = op.rtol or self.rtol
        atol = op.atol or self.atol
        assert_close(output_spyre, output_cpu, rtol=rtol, atol=atol)

    @parametrize("op", _factory_ops)
    def test_factory_op(self, op):
        for input in op.inputs:
            output_cpu = op.fn(*input, dtype=self.dtype, device="cpu")
            output_spyre = op.fn(*input, dtype=self.dtype, device="spyre")

            self._assert_close(op, output_spyre.cpu(), output_cpu)

    @parametrize("op", _factory_ops)
    def test_factory_op_out(self, op):
        for input in op.inputs:
            buffer_cpu = torch.empty(0)
            output_cpu = op.fn(*input, out=buffer_cpu)

            buffer_spyre = torch.empty_like(output_cpu, device="spyre")
            output_spyre = op.fn(*input, out=buffer_spyre)

            self._assert_close(op, output_spyre.cpu(), output_cpu)

    @parametrize("op", _unary_ops)
    def test_unary_op(self, op):
        for input in op.inputs:
            output_cpu = op.fn(input)
            output_spyre = op.fn(input.to("spyre"))

            self._assert_close(op, output_spyre.cpu(), output_cpu)

    @parametrize("op", _unary_ops)
    def test_unary_op_out(self, op):
        for input in op.inputs:
            buffer_cpu = torch.empty_like(input)
            output_cpu = op.fn(input, out=buffer_cpu)

            buffer_spyre = torch.empty_like(input, device="spyre")
            output_spyre = op.fn(input.to("spyre"), out=buffer_spyre)

            self._assert_close(op, output_spyre.cpu(), output_cpu)

    @parametrize("op", _unary_ops)
    def test_unary_op_out_alias(self, op):
        for input in op.inputs:
            buffer_cpu = torch.clone(input)
            output_cpu = op.fn(buffer_cpu, out=buffer_cpu)

            buffer_spyre = torch.clone(input).to(device="spyre")
            output_spyre = op.fn(buffer_spyre, out=buffer_spyre)

            self._assert_close(op, output_spyre.cpu(), output_cpu)
            self.assertEqual(id(buffer_spyre), id(output_spyre))


class TestFallbackLedger(TestCase):
    """T-CV-8 counted-ledger contract for register_fallback dispatches.

    The ledger counts EVERY dispatch reaching a registered fallback
    kernel, BEFORE the "once"-filtered FallbackWarning layer — so counts
    must track invocations, not emitted warnings. These tests pin that
    distinction plus reset and thread-safety.

    Spyre tensors are built with ``torch.empty(..., device="spyre")``
    (allocation is simulated under the codegen-only build) rather than
    ``.to("spyre")`` (a data copy, which the codegen-only build refuses).
    The fallback kernels themselves run on CPU and round-trip through
    the simulated allocator, so these tests do not need the hardware
    runtime.
    """

    LEDGER_OP = "aten::sin.default"
    THREADS = 4
    INVOCATIONS_PER_THREAD = 25

    def setUp(self):
        torch.random.manual_seed(42)
        # "error" makes the once-filter drop every warning: if counting
        # were gated on warn_fallback, all ledger tests below would see 0.
        warnings.simplefilter("error", FallbackWarning)
        reset_fallback_ledger()

    def tearDown(self):
        warnings.simplefilter("ignore", FallbackWarning)
        reset_fallback_ledger()

    def _spyre_input(self):
        return torch.empty((64,), dtype=torch.float16, device="spyre")

    def test_ledger_counts_per_invocation_not_per_warning(self):
        x = self._spyre_input()
        for _ in range(5):
            torch.sin(x)
        self.assertEqual(get_fallback_ledger().get(self.LEDGER_OP), 5)

    def test_ledger_reset(self):
        x = self._spyre_input()
        torch.sin(x)
        self.assertEqual(get_fallback_ledger().get(self.LEDGER_OP), 1)
        reset_fallback_ledger()
        self.assertEqual(get_fallback_ledger(), {})

    def test_ledger_out_variant_counted_once(self):
        x = self._spyre_input()
        buf = torch.empty_like(x)
        torch.sin(x, out=buf)
        self.assertEqual(get_fallback_ledger().get(self.LEDGER_OP), 1)

    def test_ledger_snapshot_is_a_copy(self):
        x = self._spyre_input()
        torch.sin(x)
        snapshot = get_fallback_ledger()
        snapshot[self.LEDGER_OP] = 999
        self.assertEqual(get_fallback_ledger().get(self.LEDGER_OP), 1)

    def test_ledger_thread_safety_smoke(self):
        def worker():
            x = self._spyre_input()
            for _ in range(self.INVOCATIONS_PER_THREAD):
                torch.sin(x)

        threads = [threading.Thread(target=worker) for _ in range(self.THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        expected = self.THREADS * self.INVOCATIONS_PER_THREAD
        self.assertEqual(get_fallback_ledger().get(self.LEDGER_OP), expected)


if __name__ == "__main__":
    run_tests()
