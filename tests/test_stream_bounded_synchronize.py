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

# Owner(s): ["module: stream"]

"""Bounded-synchronize fault surfacing.

The stream layer raises a typed SpyreStreamSynchronizeTimeout carrying
stream/device/FIFO facts; the compiler-side launch context (kernel name,
binary digest, tensor count) is attached through the launch-context
provider seam. These tests exercise the Python seams with mocked C
bindings so they run without the Spyre runtime extension or hardware.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_spyre import launch_context
from torch_spyre import streams


class _FakeCStream:
    """SpyreStream._cdata stand-in recording synchronize calls.

    The timeout message reproduces the real C++ error shape including the
    FIFO-state sentence and the not-reset clause, so a C++ change dropping
    either breaks these tests rather than passing silently. (flex streams
    expose no id or in-flight count, so the message carries pool ids and
    query() state only.)
    """

    def __init__(self, hang: bool = False):
        self.hang = hang
        self.calls = []

    def synchronize(self, timeout_ms=None):
        self.calls.append(timeout_ms)
        if timeout_ms is not None and self.hang:
            raise RuntimeError(
                "SpyreStreamSynchronizeTimeout: stream 0 on "
                "device 0 did not become idle within 10ms (still busy at the "
                "poll deadline). "
                "FIFO state: query()=busy; the in-flight op count is not "
                "exposed by the stream "
                "API. The stream is NOT reset — the operation may still be "
                "running; a subsequent unbounded synchronize() or device "
                "reset is required to reclaim it."
            )


def _make_stream(hang: bool = False) -> streams.Stream:
    stream = streams.Stream.__new__(streams.Stream)
    stream._cdata = _FakeCStream(hang=hang)
    return stream


class TestBoundedSynchronizeSignature(TestCase):
    """Backward compatibility of the synchronize call surface."""

    def test_stream_synchronize_no_args(self):
        stream = _make_stream()
        stream.synchronize()
        self.assertEqual(stream._cdata.calls, [None])

    def test_stream_synchronize_timeout_threaded(self):
        stream = _make_stream()
        stream.synchronize(timeout_ms=5000)
        self.assertEqual(stream._cdata.calls, [5000])

    def test_device_synchronize_dispatches_without_timeout(self):
        calls = []

        def fake_c_sync(device):
            calls.append(device)

        original = streams._C.synchronize
        streams._C.synchronize = fake_c_sync
        try:
            streams.synchronize()
        finally:
            streams._C.synchronize = original
        self.assertEqual(len(calls), 1)

    def test_device_synchronize_bounded_dispatch(self):
        calls = []

        def fake_bounded(device, timeout_ms):
            calls.append((device, timeout_ms))

        original = streams._C.synchronize_bounded
        streams._C.synchronize_bounded = fake_bounded
        try:
            streams.synchronize(timeout_ms=250)
        finally:
            streams._C.synchronize_bounded = original
        self.assertEqual(calls, [(None, 250)])


class TestLaunchContextProviderSeam(TestCase):
    """The compiler-to-runtime fault-context seam."""

    def setUp(self):
        launch_context.set_launch_context_provider(None)

    def tearDown(self):
        launch_context.set_launch_context_provider(None)

    def test_timeout_without_provider_raises_stream_facts_only(self):
        stream = _make_stream(hang=True)
        with self.assertRaises(RuntimeError) as ctx:
            stream.synchronize(timeout_ms=10)
        self.assertIn("SpyreStreamSynchronizeTimeout", str(ctx.exception))
        # Pin the FIFO-state suffix: a C++ change dropping it must fail here.
        self.assertIn("FIFO state", str(ctx.exception))
        self.assertIn("NOT reset", str(ctx.exception))
        self.assertNotIn("Recent launches", str(ctx.exception))

    def test_timeout_raises_typed_subclass(self):
        stream = _make_stream(hang=True)
        with self.assertRaises(streams.SpyreStreamSynchronizeTimeout):
            stream.synchronize(timeout_ms=10)

    def test_unknown_runtime_error_propagates_unchanged(self):
        class _UnknownErrorCStream(_FakeCStream):
            def synchronize(self, timeout_ms=None):
                if timeout_ms is not None:
                    raise RuntimeError("some unrelated C-layer failure")

        stream = streams.Stream.__new__(streams.Stream)
        stream._cdata = _UnknownErrorCStream()
        with self.assertRaises(RuntimeError) as ctx:
            stream.synchronize(timeout_ms=10)
        self.assertNotIsInstance(ctx.exception, streams.SpyreStreamSynchronizeTimeout)
        self.assertEqual(str(ctx.exception), "some unrelated C-layer failure")

    def test_negative_timeout_passes_through_to_c_boundary(self):
        # Negative-timeout refusal lives in the C++ TORCH_CHECK (both the
        # single-stream overload and synchronizeDeviceBounded); the Python
        # wrapper deliberately does not duplicate it. Assert the wrapper
        # forwards the value unchanged rather than faking a refusal here.
        stream = _make_stream()
        stream.synchronize(timeout_ms=-5)
        self.assertEqual(stream._cdata.calls, [-5])

    def test_timeout_with_provider_appends_context(self):
        launch_context.set_launch_context_provider(
            lambda: "kernel 'neg' (158 bytes, 3 tensor args)"
        )
        stream = _make_stream(hang=True)
        with self.assertRaises(RuntimeError) as ctx:
            stream.synchronize(timeout_ms=10)
        message = str(ctx.exception)
        self.assertIn("SpyreStreamSynchronizeTimeout", message)
        self.assertIn("Recent launches", message)
        self.assertIn("kernel 'neg'", message)

    def test_broken_provider_does_not_mask_timeout(self):
        def broken():
            raise ValueError("provider bug")

        launch_context.set_launch_context_provider(broken)
        stream = _make_stream(hang=True)
        with self.assertRaises(RuntimeError) as ctx:
            stream.synchronize(timeout_ms=10)
        message = str(ctx.exception)
        self.assertIn("SpyreStreamSynchronizeTimeout", message)
        self.assertIn("provider failed", message)

    def test_describe_recent_launches_none_without_provider(self):
        self.assertIsNone(launch_context.describe_recent_launches())

    def test_successful_bounded_wait_never_calls_provider(self):
        calls = []
        launch_context.set_launch_context_provider(
            lambda: calls.append("called") or "should not appear"
        )
        stream = _make_stream(hang=False)
        stream.synchronize(timeout_ms=10)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    run_tests()
