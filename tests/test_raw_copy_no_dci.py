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

"""Contract tests for the bit-preserving raw-copy API (dci=nullptr).

Two layers:

1. C++ source contracts (red-first, in the spirit of the
   ``test_int16_escape_hatch_data_convert_contract`` probe in
   spyre-knowledgebase): they pin the seam that makes raw copies work on-box —
   the pybind binding, the ``dci=nullptr`` call into ``copyAsyncImpl``, and
   the loud precondition checks in ``SpyreStream::copyRawAsync``. They also
   pin that the NORMAL copy path still attaches a DCI (this API must stay an
   opt-in escape hatch, never a silent default).

2. Python-surface behavior: the ``torch_spyre.raw_copy_*`` wrappers fail
   loud and typed when the runtime extension (or the ``raw_copy_tensor``
   entry point) is unavailable, and reject wrong-direction / non-tensor
   inputs before touching the device.
"""

import re
from pathlib import Path

import pytest

CSRC = Path(__file__).resolve().parents[1] / "torch_spyre" / "csrc"


def _csrc(name: str) -> str:
    return (CSRC / name).read_text()


def _function_body(source: str, signature: str) -> str:
    """Return the full definition (header through closing brace) matching
    ``signature`` at the start of a line."""
    match = re.search(
        rf"^{re.escape(signature)}.*?\n\}}",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert match, f"did not find {signature!r} in source"
    return match.group(0)


# ---------------------------------------------------------------------------
# C++ source contracts: the seam
# ---------------------------------------------------------------------------


def test_module_binds_raw_copy_tensor_to_spyre_raw_copy():
    src = _csrc("module.cpp")
    assert 'm.def("raw_copy_tensor", &spyre::spyre_raw_copy' in src, (
        "torch_spyre._C.raw_copy_tensor must be bound to "
        "spyre::spyre_raw_copy in csrc/module.cpp"
    )


def test_raw_copy_entry_exists_and_synchronizes():
    src = _csrc("spyre_mem.cpp")
    body = _function_body(src, "at::Tensor spyre_raw_copy(")
    assert "copyRawAsync" in body
    assert "synchronize()" in body, "non_blocking=False must synchronize"


def test_copy_raw_async_sends_null_dci():
    src = _csrc("spyre_stream.cpp")
    body = _function_body(src, "void SpyreStream::copyRawAsync(")
    assert "copyAsyncImpl(cpu_ptr, &ctx->composite_addr, /*dci=*/nullptr, host2device)" in body, (
        "copyRawAsync must issue the DMA with dci=nullptr (the raw-word "
        "mechanism); any dci attachment routes through ConvertData and "
        "re-triggers DtException sen_data_convert.cpp:2505 for SENINT16"
    )
    assert "generate_dci" not in body


def test_copy_raw_async_declared_in_stream_header():
    src = _csrc("spyre_stream.h")
    assert "void copyRawAsync(const at::Tensor& src, const at::Tensor& dst) const;" in src


def test_copy_raw_async_rejects_format_mismatch():
    body = _function_body(_csrc("spyre_stream.cpp"), "void SpyreStream::copyRawAsync(")
    assert "scalar types must match" in body, (
        "raw copy is bit-preserving only; a dtype mismatch must fail loud "
        "instead of silently copying mismatched bytes"
    )


def test_copy_raw_async_rejects_non_contiguous_host():
    body = _function_body(_csrc("spyre_stream.cpp"), "void SpyreStream::copyRawAsync(")
    assert "requires a contiguous host tensor" in body


def test_copy_raw_async_checks_host_storage_covers_allocation():
    body = _function_body(_csrc("spyre_stream.cpp"), "void SpyreStream::copyRawAsync(")
    assert "storage().nbytes()" in body and "total_size()" in body, (
        "the transfer is allocation-sized (stick-padded); the host storage "
        "bound check must exist or the DMA reads/writes past the host buffer"
    )


def test_normal_copy_path_still_attaches_dci():
    """The automatic path must remain unchanged: copyAsync generates a DCI."""
    src = _csrc("spyre_stream.cpp")
    body = _function_body(src, "void SpyreStream::copyAsync(")
    assert "generate_dci" in body, (
        "the normal copy path must keep attaching a DataConversionInfo; the "
        "raw bypass is opt-in only"
    )
    assert "nullptr" not in body


# ---------------------------------------------------------------------------
# Python surface
# ---------------------------------------------------------------------------


def test_raw_copy_exported_from_torch_spyre():
    import torch_spyre

    assert callable(torch_spyre.raw_copy_to_device)
    assert callable(torch_spyre.raw_copy_from_device)


def test_c_shim_exposes_raw_copy_tensor():
    import torch_spyre._C as C

    assert callable(C.raw_copy_tensor)
    assert "raw_copy_tensor" in C.__all__


def test_raw_copy_to_device_refuses_loud_without_runtime():
    """No silent fallback: unavailable runtime/entry point raises, naming it."""
    import torch
    import torch_spyre
    from torch_spyre.raw_copy import _raw_copy_entry

    t = torch.zeros(8, dtype=torch.int16)
    with pytest.raises((NotImplementedError, RuntimeError)) as excinfo:
        torch_spyre.raw_copy_to_device(t)
    assert "raw_copy_tensor" in str(excinfo.value), (
        "the refusal must name the missing C++ entry point "
        "(torch_spyre._C.raw_copy_tensor)"
    )
    # The refusal fires before any device allocation is attempted.
    with pytest.raises((NotImplementedError, RuntimeError)):
        _raw_copy_entry()


def test_raw_copy_rejects_non_tensor_and_wrong_direction():
    import torch
    import torch_spyre

    with pytest.raises(TypeError):
        torch_spyre.raw_copy_to_device("not a tensor")
    with pytest.raises(TypeError):
        torch_spyre.raw_copy_from_device("not a tensor")
    # CPU tensor handed to the D2H wrapper: rejected on direction, before any
    # runtime/extension resolution.
    with pytest.raises(TypeError, match="expects a spyre tensor"):
        torch_spyre.raw_copy_from_device(torch.zeros(4, dtype=torch.int16))
