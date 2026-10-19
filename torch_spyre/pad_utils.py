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

"""Generic load-time shape-padding primitives for Spyre stick geometry.

T-TB-23 C2. These are the GENERIC tensor-shape padding primitives the
hf-adapters model patches reduce to: end-padding (an ``F.pad``-style
constant tail along one dim) and head-dim padding where the semantic is
pure shape padding (V/O projections; see hf-adapters ``hf_common.py``
``pad_attention_heads``). The SEMANTIC padding rules -- the RoPE Q/K
interleaved pad (each ``[2, D/2]`` half-group padded separately so
``apply_rope_matmul`` identity-rotates the padded positions) and
position_ids construction for sequence padding -- STAY in hf-adapters;
this module never encodes them.

Layer placement / divergence from upstream: these primitives belong to
the load-time model-materialization layer. In the upstream torch-spyre
checkout that layer is ``torch_spyre/model_utils.py``
(``load_model_to_spyre``, ``_dma_to_spyre_indirect_access`` around
model_utils.py:212) -- the convergence point: when this lineage
reconverges with upstream, these functions slot in beside the DMA
layout helpers there. This ``new-runtime`` lineage does not carry
``model_utils.py`` yet, so they live here as a standalone module; no
upstream file is duplicated.

Boundary facts (stick geometry, EAR span) are the KB-side C1 accessors
in spirit (``GenerationCapacities.elements_per_stick`` /
``ear_addressable_bytes``, spyre-knowledgebase commit ``2c72d393`` --
the authority). No cross-repo import: the same derivations are
expressed here as module constants:

* ``STICK_BYTES`` -- one Spyre stick is 128 bytes (1p0 HardwareDescription).
* ``EAR_WIDTH_BITS`` -- the L3 EAR register holds an unsigned stick
  index, 21 bits wide on 1p0 (L3LU RegisterSpec).
* ``EAR_ADDRESSABLE_BYTES`` = ``2**EAR_WIDTH_BITS * STICK_BYTES`` --
  the closed form behind the 256 MiB vocab/smoothness bound (corroborated
  by deeptools dsm/sen_data_ops.cpp:6363, senulator memoryElement.cpp:407).

Every refusal here is LOUD and named: a dtype off the load-time
padding whitelist (bool, float64, non-dtypes), non-tensor input, a
pad target below the original length, a non-int target length, a
non-positive length or vocab count, an out-of-range dimension or
0-dim tensor, and an EAR span no padding can satisfy each raise
their own exception class -- never a silent no-op or a bare
TypeError/IndexError.

Intended consumption seam (lead-ruled): model-level wiring lives on
the CONSUMER side -- e.g. the spyre-knowledgebase stock-run probe's
``--pad`` flag calls these primitives directly against a model's
configs/weights. This module deliberately offers no model-level
wrapper; torch-spyre grows only the primitive layer.
"""

from __future__ import annotations

import torch

# --- Stick geometry / EAR span (KB C1 authority: 2c72d393) ----------------

STICK_BYTES = 128
EAR_WIDTH_BITS = 21
EAR_ADDRESSABLE_BYTES = (1 << EAR_WIDTH_BITS) * STICK_BYTES  # 256 MiB on 1p0


# --- Named refusals --------------------------------------------------------


class PadPrimitiveError(Exception):
    """Base class for load-time padding refusals (always loud)."""


class UnsupportedPadDtypeError(PadPrimitiveError):
    """The dtype has no integral element capacity in a 128-byte stick."""


class NotATensorError(PadPrimitiveError):
    """The value to pad is not a ``torch.Tensor``."""


class PadTargetBelowOriginalError(PadPrimitiveError):
    """Padding may only grow a dimension, never shrink it."""


class EARSpanUncomputableError(PadPrimitiveError):
    """No padded length brings the per-core span under the EAR bound."""


class InvalidVocabSizeError(PadPrimitiveError):
    """A vocab size that is not a positive count of rows to pad."""


# --- Stick-geometry arithmetic ---------------------------------------------

# Dtype whitelist: exactly the dtypes with a known Spyre device
# representation at load-time padding granularity. Anything else --
# bool (128 "elements" of flag is the trap class), float64, or a
# non-dtype -- refuses loudly rather than computing a stick count the
# device cannot honor.
_PAD_SUPPORTED_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.int8}
)


def elements_per_stick(dtype: torch.dtype) -> int:
    """Elements per 128-byte stick for ``dtype``.

    Derived from the stick size and the dtype's itemsize (the same
    derivation as the KB C1 accessor), never a hardcoded 64: fp16/bf16
    -> 64, fp32 -> 32, int8 -> 128. Refuses loudly when the dtype is
    not on the load-time padding whitelist (e.g. bool, float64 -- no
    known device representation at this granularity) or is not a
    ``torch.dtype`` instance.
    """
    if not isinstance(dtype, torch.dtype):
        raise UnsupportedPadDtypeError(
            f"elements_per_stick: expected a torch.dtype, got {dtype!r}."
        )
    if dtype not in _PAD_SUPPORTED_DTYPES:
        raise UnsupportedPadDtypeError(
            f"elements_per_stick: dtype {dtype} is not one of the "
            f"load-time padding dtypes "
            f"{sorted(str(d) for d in _PAD_SUPPORTED_DTYPES)}."
        )
    itemsize = dtype.itemsize
    if itemsize <= 0 or STICK_BYTES % itemsize != 0:
        raise UnsupportedPadDtypeError(
            f"elements_per_stick: dtype {dtype} (itemsize {itemsize}B) "
            f"does not pack integrally into a {STICK_BYTES}B stick."
        )
    return STICK_BYTES // itemsize


def pad_length_to_multiple(length: int, multiple: int) -> int:
    """Round ``length`` up to a multiple of ``multiple`` (identity if it
    already is). Refuses a non-positive multiple and a negative
    length: a negative length is a config defect, not a pad target."""
    if multiple <= 0:
        raise PadPrimitiveError(
            f"pad_length_to_multiple: multiple must be positive, got {multiple!r}."
        )
    if length < 0:
        raise PadPrimitiveError(
            f"pad_length_to_multiple: length must be non-negative, got {length!r}."
        )
    return -(-length // multiple) * multiple


# --- Generic end-pad primitive ---------------------------------------------


def end_pad(
    tensor: torch.Tensor,
    dim: int,
    target_length: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Constant-tail pad of ``tensor`` along ``dim`` up to
    ``target_length`` (an ``F.pad``-style end-pad; identity when the dim
    already meets the target).

    This is the primitive behind the V/O head-dim pad (hf-adapters
    ``pad_attention_heads``) and any pad-to-stick-multiple rule: the
    original data keeps its positions, the tail is constant. Anything
    positional (the RoPE Q/K interleave) is adapter-side by design.

    Refusals (loud, named): non-tensor input, a target below the
    original length, a negative or out-of-range dimension index, a
    0-dim tensor, or a non-int target length.
    """
    if not isinstance(tensor, torch.Tensor):
        raise NotATensorError(
            f"end_pad: expected a torch.Tensor, got {type(tensor).__name__}."
        )
    if not isinstance(dim, int) or dim < 0:
        raise PadPrimitiveError(
            f"end_pad: dim must be a non-negative int, got {dim!r}."
        )
    if tensor.ndim == 0:
        raise PadPrimitiveError(
            f"end_pad: cannot pad a 0-dim tensor (shape {tuple(tensor.shape)})."
        )
    if dim >= tensor.ndim:
        raise PadPrimitiveError(
            f"end_pad: dim {dim} is out of range for a "
            f"{tensor.ndim}-dim tensor (shape {tuple(tensor.shape)})."
        )
    if not isinstance(target_length, int):
        raise PadPrimitiveError(
            f"end_pad: target_length must be an int, got "
            f"{target_length!r} ({type(target_length).__name__})."
        )
    original = tensor.shape[dim]
    if target_length < original:
        raise PadTargetBelowOriginalError(
            f"end_pad: target length {target_length} is below the "
            f"original length {original} along dim {dim}; padding may "
            f"only grow a dimension."
        )
    if target_length == original:
        return tensor
    pad_amount = target_length - original
    # F.pad wants a pair per dim, innermost first; only dim is nonzero.
    pad = [0] * (2 * tensor.ndim)
    pad[(tensor.ndim - 1 - dim) * 2 + 1] = pad_amount
    return torch.nn.functional.pad(tensor, pad, mode="constant", value=fill_value)


# --- Head-dim rule ---------------------------------------------------------


def padded_head_dim(head_dim: int, dtype: torch.dtype) -> int:
    """Head-dim padded up to a multiple of ``2 * elements_per_stick``.

    The hf-adapters rule (``hf_common.py:3006-3017``): the RoPE
    ``[2, D/2]`` reshape needs each half-group to be at least one
    stick, so head_dim rounds up to a multiple of twice the
    elements-per-stick (2 x 64 = 128 for fp16/bf16). Expressed here as
    the GENERIC multiple so torch-spyre callers get the same target
    without importing adapter code.
    """
    eps = elements_per_stick(dtype)
    return pad_length_to_multiple(head_dim, 2 * eps)


# --- Vocab smoothness / EAR span -------------------------------------------


def _largest_prime_factor(n: int) -> int:
    largest = 1
    while n % 2 == 0:
        largest = 2
        n //= 2
    f = 3
    while f * f <= n:
        while n % f == 0:
            largest = f
            n //= f
        f += 2
    if n > 1:
        largest = n
    return largest


def vocab_stick_span_bytes(
    vocab_size: int, hidden_size: int, dtype: torch.dtype
) -> int:
    """Per-core EAR byte span of an embedding table with the given
    vocab/hidden: the bundler's per-core weight span,
    ``lpf(vocab_sticks) * hidden * elements_per_stick * itemsize``
    (the arithmetic behind the dxp_standalone abort class; KB C1 /
    hf-adapters ARCHITECTURE.md:386-400).

    Refuses non-positive vocab loudly (:class:`InvalidVocabSizeError`):
    vocab 0 is a config defect, not a pad target -- and ``lpf(0)``
    would not terminate, so the refusal also guards the loop.
    """
    if vocab_size <= 0:
        raise InvalidVocabSizeError(
            f"vocab_stick_span_bytes: vocab_size must be a positive "
            f"row count, got {vocab_size!r}."
        )
    eps = elements_per_stick(dtype)
    vocab_sticks = -(-vocab_size // eps)  # ceil
    lpf = _largest_prime_factor(vocab_sticks)
    return lpf * hidden_size * eps * dtype.itemsize


def padded_vocab_size(vocab_size: int, hidden_size: int, dtype: torch.dtype) -> int:
    """Smallest padded vocab (a multiple of elements_per_stick, at least
    ``vocab_size``) whose per-core EAR span stays under
    ``EAR_ADDRESSABLE_BYTES``.

    Refuses loudly with :class:`EARSpanUncomputableError` when no vocab
    can help -- i.e. the span is over the EAR bound even at the smallest
    possible largest-prime-factor (2, for any stick count >= 2): that is
    a hidden-size/dtype problem, not a padding problem.

    Refuses non-positive vocab loudly (:class:`InvalidVocabSizeError`)
    for the same reason as :func:`vocab_stick_span_bytes`.
    """
    if vocab_size <= 0:
        raise InvalidVocabSizeError(
            f"padded_vocab_size: vocab_size must be a positive row "
            f"count, got {vocab_size!r}."
        )
    eps = elements_per_stick(dtype)
    floor_span = 2 * hidden_size * eps * dtype.itemsize
    if floor_span > EAR_ADDRESSABLE_BYTES:
        raise EARSpanUncomputableError(
            f"padded_vocab_size: no vocab padding can bring the per-core "
            f"span under the EAR bound -- even lpf=2 gives "
            f"{floor_span} bytes for hidden={hidden_size}, "
            f"dtype={dtype} against EAR_ADDRESSABLE_BYTES="
            f"{EAR_ADDRESSABLE_BYTES}."
        )
    candidate = pad_length_to_multiple(vocab_size, eps)
    while vocab_stick_span_bytes(candidate, hidden_size, dtype) > EAR_ADDRESSABLE_BYTES:
        candidate += eps
    return candidate
