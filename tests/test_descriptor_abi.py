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

import pytest

from torch_spyre._inductor.descriptor_abi import (
    STICK_BYTES,
    SpyreDescriptorAlignmentError,
    check_descriptor_pointer_alignment,
    check_descriptor_pointer_alignments,
    describe_descriptor_pointer_alignment,
)


class TensorArg:
    def __init__(self, *, data_ptr: int, element_size: int, storage_offset: int):
        self._data_ptr = data_ptr
        self._element_size = element_size
        self._storage_offset = storage_offset

    def data_ptr(self):
        return self._data_ptr

    def element_size(self):
        return self._element_size

    def storage_offset(self):
        return self._storage_offset


def test_descriptor_abi_accepts_128b_effective_pointer_and_stick_offset():
    tensor = TensorArg(data_ptr=0x4000, element_size=2, storage_offset=64)

    report = describe_descriptor_pointer_alignment(tensor)

    assert report.effective_data_ptr % STICK_BYTES == 0
    assert report.storage_offset_bytes == STICK_BYTES
    assert report.pointer_aligned is True
    assert report.storage_offset_aligned is True
    assert report.aligned is True
    assert report.to_dict()["aligned"] is True


def test_descriptor_abi_rejects_effective_pointer_misalignment():
    tensor = TensorArg(data_ptr=0x4002, element_size=2, storage_offset=0)

    with pytest.raises(SpyreDescriptorAlignmentError, match="effective data_ptr"):
        check_descriptor_pointer_alignment(tensor)


def test_descriptor_abi_rejects_storage_offset_that_breaks_stick_boundary():
    tensor = TensorArg(data_ptr=0x4000, element_size=2, storage_offset=1)

    report = describe_descriptor_pointer_alignment(tensor)

    assert report.pointer_aligned is True
    assert report.storage_offset_bytes == 2
    assert report.storage_offset_aligned is False
    assert report.aligned is False
    assert any("64 elements" in note for note in report.notes)

    with pytest.raises(SpyreDescriptorAlignmentError, match="storage_offset=1"):
        check_descriptor_pointer_alignment(tensor)


def test_descriptor_abi_sequence_helper_can_run_as_shadow_diagnostic():
    aligned = TensorArg(data_ptr=0x4000, element_size=2, storage_offset=0)
    misaligned = TensorArg(data_ptr=0x4002, element_size=2, storage_offset=0)

    reports = check_descriptor_pointer_alignments([aligned, misaligned])

    assert [report.aligned for report in reports] == [True, False]
