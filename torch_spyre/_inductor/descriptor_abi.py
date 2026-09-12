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
"""Diagnostic-only Sentient stick-alignment contract for descriptor-admitted
Spyre tensor arguments.

Sentient's stick is 128 bytes (AIU 1.0 DL16 / f16: 64 elements * 2 bytes). A
descriptor-admitted (TensorDescriptor/TMA-style) tensor argument's *effective*
runtime pointer -- the address of element 0 of the view actually passed to a
kernel -- must land on a 128-byte boundary for the stick-addressed load/store
path to be well-formed. This module defines that contract as a standalone,
duck-typed check so it can be exercised against real ``torch.Tensor``
objects, ``FakeTensor`` objects, and plain test doubles alike.

Two things this module deliberately does NOT do:

  1. It does not gate production admission. Nothing here changes which ops or
     shapes route through the descriptor pointwise diagnostic
     (``TORCH_SPYRE_EXPERIMENTAL_DESCRIPTOR_POINTWISE``, see
     ``torch_spyre._inductor.patches``); it only reports on the tensors that
     already reach that opt-in path.
  2. It does not trust Triton's ``{tt.divisibility = 16}`` TTIR attribute as
     alignment proof. That attribute is a *compile-time lower bound* Inductor
     derives for a symbolic pointer expression (a load/store base is a
     multiple of 16 bytes) -- it says nothing about whether the concrete
     ``data_ptr()`` passed at a given call lands on a 128-byte Sentient stick
     boundary. A multiple of 16 need not be a multiple of 128, so this module
     checks the runtime value instead.

Effective pointer, not allocation base
---------------------------------------

The value this module checks is ``tensor.data_ptr()``. PyTorch defines that
as the address of the tensor's *first element* -- i.e. the storage's base
pointer plus ``storage_offset() * element_size()`` -- not
``tensor.untyped_storage().data_ptr()`` (the allocation base only). Checking
the storage's base pointer instead of ``data_ptr()`` would silently miss a
narrowed/sliced/viewed argument (nonzero ``storage_offset()``) whose EFFECTIVE
address breaks stick alignment even though the underlying allocation is
itself 128-byte aligned. ``storage_offset_aligned``/``storage_offset_bytes``
below isolate exactly that offset-composition failure mode, independent of
whatever the allocation base represents (see the next section for why that
independence matters on Spyre today).

Known limitation: SpyreAllocator's ``data_ptr()`` is not the device address
------------------------------------------------------------------------------

``torch_spyre/csrc/spyre_allocator.cpp`` (``SpyreAllocator::allocate``) hands
PyTorch a ``c10::DataPtr`` whose value is a pointer to the allocation's
host-side ``SharedOwnerCtx`` bookkeeping struct ("Use the SharedOwnerCtx
pointer as the unique data handle for c10::DataPtr. This pointer is never
dereferenced -- it serves only as a unique token for memory profiling."). The
real on-device address lives in ``flex::CompositeAddress`` inside that struct
and is never surfaced to Python; FlexAllocator does round IT to
``DEVICE_ALIGNMENT`` (128 bytes), but ``Tensor.data_ptr()`` on today's
PrivateUse1 ``spyre`` tensors returns the unrelated host token instead. So on
a real (non-fake) Spyre tensor, ``pointer_aligned`` below reflects the host
allocator's/``operator new``'s incidental alignment, not proof of on-device
stick alignment -- passing or failing that check today is not itself
evidence about the real device address. ``storage_offset_aligned`` remains a
real, allocator-independent signal (a narrowed view's element offset either
is or is not a multiple of the natural stick element count, regardless of
what the base token represents), which is why callers that only trust the
Spyre codegen-only/fake backend should treat that half of the report as the
load-bearing one until the effective on-device address is exposed to Python.

Also on today's backend, ``aten::as_strided`` is not implemented for the real
(non-fake) PrivateUse1 ``spyre`` device, so a real spyre tensor cannot
currently be narrowed to a nonzero ``storage_offset()`` at all (it raises
``NotImplementedError``) -- a genuinely misaligned REAL device tensor
argument cannot be constructed in this environment today. Tests therefore
exercise the misaligned path via ``FakeTensorMode`` tensors (whose
``data_ptr()``/``storage_offset()`` arithmetic is internally consistent, just
not device-backed) and via plain test doubles, and document this limitation
rather than assert past it.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Sequence

_log = logging.getLogger("torch_spyre.descriptor_abi")

# Sentient stick size in bytes. AIU 1.0 DL16/f16 packs 64 elements * 2 bytes
# per stick; 1.5 BF16 keeps the same 2-byte-element stick width. See
# feedback-l3-stick-units / stick layout conventions in the knowledgebase.
STICK_BYTES = 128

# 2-byte-element (DL16/f16/BF16) stick width, in elements.
_TWO_BYTE_STICK_ELEMENTS = STICK_BYTES // 2


class SpyreDescriptorAlignmentError(RuntimeError):
    """A descriptor-admitted Spyre tensor argument's effective pointer (or its
    storage_offset composition) fails the 128-byte Sentient stick alignment
    contract. Diagnostic-only: raising this never happens on the production
    admission path, only from an explicit call to
    ``check_descriptor_pointer_alignment``.
    """


@dataclasses.dataclass(frozen=True)
class DescriptorAlignmentReport:
    """Result of checking one tensor argument's effective-pointer alignment.

    ``pointer_aligned`` is the literal ABI contract: is ``data_ptr()`` (the
    effective pointer, base + storage_offset*itemsize) a multiple of
    ``stick_bytes``. ``storage_offset_aligned`` isolates whether the
    storage_offset composition ALONE preserves stick alignment, independent
    of whatever the allocation base represents -- see the module docstring's
    "Known limitation" section for why that independence matters on today's
    Spyre codegen-only backend.
    """

    effective_data_ptr: int
    element_size: int
    storage_offset: int
    storage_offset_bytes: int
    stick_bytes: int
    pointer_aligned: bool
    storage_offset_aligned: bool
    notes: tuple[str, ...] = ()

    @property
    def aligned(self) -> bool:
        return self.pointer_aligned and self.storage_offset_aligned

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe rendering (plain ints/bools/strings only)."""
        return {
            "effective_data_ptr": self.effective_data_ptr,
            "element_size": self.element_size,
            "storage_offset": self.storage_offset,
            "storage_offset_bytes": self.storage_offset_bytes,
            "stick_bytes": self.stick_bytes,
            "pointer_aligned": self.pointer_aligned,
            "storage_offset_aligned": self.storage_offset_aligned,
            "aligned": self.aligned,
            "notes": list(self.notes),
        }


def describe_descriptor_pointer_alignment(
    tensor: Any,
    *,
    stick_bytes: int = STICK_BYTES,
) -> DescriptorAlignmentReport:
    """Compute (without raising) the alignment report for ``tensor``.

    ``tensor`` is duck-typed: any object exposing ``data_ptr()``,
    ``storage_offset()``, and ``element_size()`` (a ``torch.Tensor``, a
    ``FakeTensor``, or a test double) works.
    """
    data_ptr = int(tensor.data_ptr())
    element_size = int(tensor.element_size())
    storage_offset = int(tensor.storage_offset())
    storage_offset_bytes = storage_offset * element_size

    pointer_aligned = (data_ptr % stick_bytes) == 0
    storage_offset_aligned = (storage_offset_bytes % stick_bytes) == 0

    notes: list[str] = []
    if not pointer_aligned:
        notes.append(
            f"effective data_ptr=0x{data_ptr:x} is not a multiple of "
            f"{stick_bytes} bytes"
        )
    if not storage_offset_aligned:
        notes.append(
            f"storage_offset={storage_offset} elements "
            f"({storage_offset_bytes} bytes) is not a multiple of "
            f"{stick_bytes} bytes"
        )
        if element_size == 2 and storage_offset % _TWO_BYTE_STICK_ELEMENTS:
            notes.append(
                f"storage_offset={storage_offset} is not a multiple of "
                f"{_TWO_BYTE_STICK_ELEMENTS} elements (the DL16/f16/BF16 "
                "stick boundary)"
            )

    return DescriptorAlignmentReport(
        effective_data_ptr=data_ptr,
        element_size=element_size,
        storage_offset=storage_offset,
        storage_offset_bytes=storage_offset_bytes,
        stick_bytes=stick_bytes,
        pointer_aligned=pointer_aligned,
        storage_offset_aligned=storage_offset_aligned,
        notes=tuple(notes),
    )


def check_descriptor_pointer_alignment(
    tensor: Any,
    *,
    stick_bytes: int = STICK_BYTES,
    context: str = "",
    raise_on_misalignment: bool = True,
) -> DescriptorAlignmentReport:
    """Check ``tensor`` against the 128-byte descriptor ABI contract.

    Always computes and logs the report. When the report is not aligned and
    ``raise_on_misalignment`` is true (the default), raises
    ``SpyreDescriptorAlignmentError`` -- fail-loud, never silent-wrong.

    Callers integrating this into a path where ``data_ptr()`` is known not to
    reflect a trustworthy effective address (e.g. today's Spyre codegen-only
    backend -- see the module docstring) should pass
    ``raise_on_misalignment=False`` and treat the returned report as an
    observability signal rather than a gate.
    """
    report = describe_descriptor_pointer_alignment(tensor, stick_bytes=stick_bytes)
    label = f" ({context})" if context else ""
    if report.aligned:
        _log.debug(
            "Spyre descriptor ABI alignment OK%s: data_ptr=0x%x storage_offset=%d",
            label,
            report.effective_data_ptr,
            report.storage_offset,
        )
        return report

    message = (
        f"Spyre descriptor ABI alignment violation{label}: "
        + "; ".join(report.notes)
    )
    if raise_on_misalignment:
        _log.error(message)
        raise SpyreDescriptorAlignmentError(message)
    _log.debug(message)
    return report


def check_descriptor_pointer_alignments(
    tensors: Sequence[Any],
    *,
    stick_bytes: int = STICK_BYTES,
    context: str = "",
    raise_on_misalignment: bool = False,
) -> list[DescriptorAlignmentReport]:
    """Convenience helper: check a sequence of tensors, returning a report per
    tensor. Defaults to ``raise_on_misalignment=False`` because the natural
    caller for a sequence is a shadow-only diagnostic walk over compile-time
    example inputs (see ``torch_spyre._inductor.patches``), not a single
    trusted argument check.
    """
    return [
        check_descriptor_pointer_alignment(
            tensor,
            stick_bytes=stick_bytes,
            context=context,
            raise_on_misalignment=raise_on_misalignment,
        )
        for tensor in tensors
    ]
