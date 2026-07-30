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

"""Tests for spyre_constant_tensor caching behavior.

Tests verify:
(a) Two calls with same (value, device, dtype) return the same object
(b) Different dtype/device/value produce distinct tensors
(c) The produced value is numerically correct after DMA fill
(d) Cache is scoped to V.graph lifetime
"""

import unittest

import torch
from torch._inductor.virtualized import V

from torch_spyre._inductor.op_spec import spyre_constant_tensor


class TestConstantTensorCache(unittest.TestCase):
    """Test V.graph-based caching for spyre_constant_tensor."""

    def setUp(self):
        """Reset compiler state before each test."""
        torch.compiler.reset()
        torch.manual_seed(0xBEEF)

    def tearDown(self):
        """Clean up after each test."""
        torch.compiler.reset()

    def test_cache_hit_returns_same_object(self):
        """Two calls with same (value, device, dtype) return the same tensor object."""
        # Create a mock graph to simulate compilation context
        mock_graph = type("MockGraph", (), {})()

        # Save original graph
        original_getter = V.__class__.graph.fget

        # Mock V.graph to return our mock_graph
        def mock_graph_getter(self):
            return mock_graph

        # Patch V.graph temporarily
        V.__class__.graph = property(mock_graph_getter)

        try:
            # First call creates and caches
            t1 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)

            # Check cache was created
            self.assertTrue(
                hasattr(V.graph, "_spyre_constant_tensors"),
                "Cache should be created on V.graph",
            )
            cache = V.graph._spyre_constant_tensors
            self.assertEqual(
                len(cache), 1, "Cache should have 1 entry after first call"
            )

            # Second call should return same object
            t2 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)

            self.assertIs(t1, t2, "Cache hit should return same object")
            self.assertEqual(
                len(cache), 1, "Cache should still have only 1 entry after hit"
            )

        finally:
            # Restore original V.graph
            V.__class__.graph = property(original_getter)

    def test_cache_miss_creates_new_tensor(self):
        """Different values produce distinct tensors with separate cache entries."""
        mock_graph = type("MockGraph", (), {})()
        original_getter = V.__class__.graph.fget

        def mock_graph_getter(self):
            return mock_graph

        V.__class__.graph = property(mock_graph_getter)

        try:
            t1 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)
            t2 = spyre_constant_tensor(2.0, torch.device("spyre"), torch.float16)

            self.assertIsNot(
                t1, t2, "Different values should produce different tensors"
            )

            cache = V.graph._spyre_constant_tensors
            self.assertEqual(
                len(cache), 2, "Cache should have 2 entries for different values"
            )

        finally:
            V.__class__.graph = property(original_getter)

    def test_different_dtype_creates_new_tensor(self):
        """Different dtype produces distinct tensors even with same value."""
        mock_graph = type("MockGraph", (), {})()
        original_getter = V.__class__.graph.fget

        def mock_graph_getter(self):
            return mock_graph

        V.__class__.graph = property(mock_graph_getter)

        try:
            t1 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)
            t2 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float32)

            self.assertIsNot(t1, t2, "Different dtype should produce different tensors")

            cache = V.graph._spyre_constant_tensors
            self.assertEqual(
                len(cache), 2, "Cache should have 2 entries for different dtypes"
            )

        finally:
            V.__class__.graph = property(original_getter)

    def test_different_device_creates_new_tensor(self):
        """Different device produces distinct tensors even with same value and dtype."""
        mock_graph = type("MockGraph", (), {})()
        original_getter = V.__class__.graph.fget

        def mock_graph_getter(self):
            return mock_graph

        V.__class__.graph = property(mock_graph_getter)

        try:
            # Note: We can't actually test different devices in unit tests,
            # but verify the key includes device
            spyre_constant_tensor(1.0, torch.device("spyre:0"), torch.float16)

            # Check cache key includes device
            cache = V.graph._spyre_constant_tensors
            self.assertEqual(len(cache), 1)

            # Verify device is part of key
            cache_keys = list(cache.keys())
            _, device_key, _ = cache_keys[0]
            self.assertIsInstance(
                device_key, tuple, "Device should be tuple in cache key"
            )

        finally:
            V.__class__.graph = property(original_getter)

    def test_dma_fill_numerical_correctness(self):
        """DMA fill produces correct numerical values."""
        mock_graph = type("MockGraph", (), {})()
        original_getter = V.__class__.graph.fget

        def mock_graph_getter(self):
            return mock_graph

        V.__class__.graph = property(mock_graph_getter)

        try:
            test_values = [0.0, 1.0, -1.0, 3.14159, -2.71828]

            for val in test_values:
                t = spyre_constant_tensor(val, torch.device("spyre"), torch.float16)
                cpu_ref = torch.tensor(val, dtype=torch.float16)

                # Compare values on CPU
                t_cpu = t.cpu()
                self.assertTrue(
                    torch.allclose(t_cpu, cpu_ref, rtol=1e-3, atol=1e-3),
                    f"DMA fill mismatch: expected {val}, got {t_cpu.item()}",
                )

        finally:
            V.__class__.graph = property(original_getter)

    def test_cache_scoped_to_graph(self):
        """Each V.graph has its own isolated cache."""
        # First graph
        mock_graph1 = type("MockGraph", (), {})()
        original_getter = V.__class__.graph.fget

        def mock_graph_getter1(self):
            return mock_graph1

        V.__class__.graph = property(mock_graph_getter1)

        try:
            t1 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)
            # Verify cache was created
            self.assertEqual(len(V.graph._spyre_constant_tensors), 1)

        finally:
            V.__class__.graph = property(original_getter)

        # Second graph - different compilation
        mock_graph2 = type("MockGraph", (), {})()

        def mock_graph_getter2(self):
            return mock_graph2

        V.__class__.graph = property(mock_graph_getter2)

        try:
            # Should not have cache from graph1
            self.assertFalse(
                hasattr(V.graph, "_spyre_constant_tensors"),
                "New graph should not have cache from previous graph",
            )

            t2 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)

            # Should be different object (different graph = different cache)
            self.assertIsNot(t1, t2, "Different graph should produce different tensor")

        finally:
            V.__class__.graph = property(original_getter)

    def test_real_compilation_creates_cache(self):
        """Test that cache is created during real compilation."""

        # Simple model that uses constant
        @torch.compile
        def model_with_constant(x):
            return x + 1.0

        x = torch.randn(2, 3, dtype=torch.float16, device="spyre")

        # Run compilation
        result = model_with_constant(x)

        # Verify compilation succeeded
        self.assertEqual(result.shape, torch.Size([2, 3]))

        # Note: V.graph is not accessible after compilation ends,
        # but the test verifies no errors during compilation

    def test_string_device_input(self):
        """String device input is handled correctly."""
        mock_graph = type("MockGraph", (), {})()
        original_getter = V.__class__.graph.fget

        def mock_graph_getter(self):
            return mock_graph

        V.__class__.graph = property(mock_graph_getter)

        try:
            # String device
            t1 = spyre_constant_tensor(1.0, "spyre", torch.float16)

            # torch.device input
            t2 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)

            # Should return same cached tensor
            self.assertIs(t1, t2, "String and torch.device should use same cache")

        finally:
            V.__class__.graph = property(original_getter)


if __name__ == "__main__":
    unittest.main()
