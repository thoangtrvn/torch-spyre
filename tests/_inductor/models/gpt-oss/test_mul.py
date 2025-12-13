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

# Owner(s): ["module: cpp"]
import torch
import torch.nn as nn
import random
import numpy as np
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
    parametrize,
    instantiate_parametrized_tests,
)
import pytest


def set_global_seed(seed: int):
    """Set all relevant RNGs to a fixed seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_op_callable():
    return torch.mul


class OpModule(nn.Module):
    def __init__(self):
        super().__init__()
        op = get_op_callable()
        self.op = op

    def forward(self, *args):
        return self.op(*args)


def run_op(net, inputs, device, backend="inductor"):
    """
    Run the module on the requested device.
    - CPU → eager
    - non-CPU (Spyre) → torch.compile(..., backend=inductor)
    """
    net = net.to(device)
    inputs = [t.to(device) for t in inputs]

    if device == "cpu":
        net_compile = net
    else:
        net_compile = torch.compile(net, backend=backend)

    out = net_compile(*inputs)
    if isinstance(out, tuple):
        out = out[0]
    return out.detach()


class TestOps(TestCase):
    @classmethod
    def setUpClass(cls):
        # Backend and type/tolerance configuration that you can extend
        cls.backend = "inductor"

        cls.dtype_map = {
            "fp16": torch.float16,
        }

        # Per-dtype tolerances for allclose (floating types)
        cls.tol_map = {
            "fp16": (1e-2, 1e-2),
        }

    def make_inputs(self, shape0, shape1, dtype_name, seed=123):
        """
        Factory fixture to create two random tensors of given shapes on CPU first,
        so they share the exact same values before moving to other devices.
        """
        set_global_seed(seed)
        dtype = self.dtype_map[dtype_name]
        var0 = torch.rand(torch.Size(shape0), dtype=dtype, device="cpu")
        var1 = torch.rand(torch.Size(shape1), dtype=dtype, device="cpu")
        return [var0, var1]

    def _mul_cases(self, case_name, shape0, shape1, dtype_name):
        """
        Single invocation per case:
        - Build inputs once on CPU (same values).
        - Run CPU eager → baseline.
        - Run Spyre compiled → device under test.
        - Compare results on CPU.
        """
        # Prepare identical inputs (same shapes, same values) on CPU first
        inputs_cpu = self.make_inputs(shape0, shape1, dtype_name)

        # CPU eager baseline
        out_cpu = run_op(net=OpModule(), inputs=inputs_cpu, device="cpu")

        # Spyre compiled path
        out_spyre = run_op(net=OpModule(), inputs=inputs_cpu, device="spyre")

        # Compare on CPU
        out_spyre_cpu = out_spyre.to("cpu")
        atol, rtol = self.tol_map[dtype_name]

        same = torch.allclose(out_cpu, out_spyre_cpu, atol=atol, rtol=rtol)
        max_diff = 0.0 if same else (out_cpu - out_spyre_cpu).abs().max().item()

        assert same, f"[{case_name}] Outputs differ: max diff={max_diff}"


@pytest.mark.paddedtensor
class TestOpsPad(TestOps):
    @parametrize(
        "case_name, shape0, shape1, dtype_name",
        [
            ("mul_ng_broadcast", [1, 64], [1, 1], "fp16"),
        ],
    )
    def test_mul_cases(self, case_name, shape0, shape1, dtype_name):
        return self._mul_cases(case_name, shape0, shape1, dtype_name)


class TestOpsNoPad(TestOps):
    @parametrize(
        "case_name, shape0, shape1, dtype_name",
        [
            ("mul_ok", [1, 64], [1, 64], "fp16"),
        ],
    )
    def test_mul_cases(self, case_name, shape0, shape1, dtype_name):
        return self._mul_cases(case_name, shape0, shape1, dtype_name)


instantiate_parametrized_tests(TestOpsPad)
instantiate_parametrized_tests(TestOpsNoPad)

if __name__ == "__main__":
    run_tests()
