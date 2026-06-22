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

"""spyre::embedding custom op — on-device token gather via sentient_codegen.

Reads indices on the host, builds per-launch init-packet binaries with the
torch-free _embedding_host helper, allocates the output on device, launches each
binary, and reshapes to PyTorch's (*idx_shape, d_model). No silent CPU fallback.

PATH SELECTION:
  d_model <= 512 (spt <= 8): EBR plane-major path (hardware-verified PM-T6).
    launch args: [weight(f16, vocab×d_model), buf(f16, tokens×d_model)]
    weight stickified plane-major (existing behavior, unchanged).

  d_model > 512 (spt > 8): IBR indirect gather path (hardware-verified IBR-T7).
    launch args: [weight_rc(f16, vocab*spt×64), ibr_table(int32, 1×32), buf(f16)]
    XLAT[0]=weight_rc (seg0), XLAT[1]=ibr_table (seg1), XLAT[2]=buf (seg2).
    weight_rc = weight.reshape(vocab*spt, 64) — row-contiguous layout (see below).
    ibr_table = int32 IBR address table (see _embedding_host.build_ibr_address_table).
    IBR[t] = (0<<seg_bits) + idx*spt = idx*spt (weight at segment 0).
    buf shape: (tokens_per_launch*spt, 64) row-contiguous; reshape after D2H to
    (tokens_per_launch, spt, 64).reshape(tokens_per_launch, d_model).

ROW-CONTIGUOUS WEIGHT RESHAPE (IBR path):
  A (vocab, d_model) weight is stickified PLANE-MAJOR on device: plane s of all
  tokens comes before plane s+1.  LDIM gather needs ROW-CONTIGUOUS layout: token
  t's spt sticks at device positions [t*spt, (t+1)*spt).

  Achieved by: weight_rc = weight.cpu().reshape(vocab*spt, 64)
  where spt = d_model // 64 (sticks per row at 16-bit).
  SpyreTensorLayout([vocab*spt, 64], float16) → stride_map=[64, 1] — one stick
  per row, sequential rows = row-contiguous.  Moving this reshaped tensor to device
  gives the correct layout for LDIM.

  Constraint: d_model must be a multiple of 64 (elements_per_stick at 16-bit).
  Most LLM d_model values (768, 1024, 2048, 4096, 8192) are multiples of 64.
  Non-multiples are rejected at planning time (build_embedding_launches raises
  ValueError); add padding in the caller before this op if needed.
"""
from __future__ import annotations

import torch
import numpy as np

from torch_spyre._C import launch_kernel_from_bytes
from ._embedding_host import (
    build_embedding_launches,
    build_ibr_address_table,
    EmbeddingHostError,  # re-exported so callers can catch the op's host-side failure mode
    _EBR_D_MODEL_MAX,
    _IBR_FILE_SIZE,
)

__all__ = ["embedding", "EmbeddingHostError"]

_DTYPE_BITS = {torch.float16: 16}  # DL16 on 1p0; extend per generation
# elements per stick at each element width (128 bytes / element_bytes).
_ELEMENTS_PER_STICK = {16: 64}  # float16: 128 / 2 = 64

# Module-level cache for row-contiguous reshaped weights (IBR path).
#
# Key: (weight_id: int, weight_version: int)
#   weight_id = id(weight) — the Python object id of the weight tensor.
#   weight_version = weight._version — PyTorch version counter, incremented
#     on in-place operations.  Combining id + version avoids stale cache hits
#     when a weight tensor is freed and a new tensor reuses the same address.
# Value: weight_rc (device tensor, float16, shape (vocab*spt, 64))
#
# The reshape (vocab, d_model) → (vocab*spt, 64) + D2H + H2D costs ~seconds
# for large vocabularies.  Caching it by (id, version) means the IBR path only
# pays this cost ONCE per unique weight tensor (i.e. once per model load), not
# once per forward-pass call.
_ibr_weight_cache: dict = {}


@torch.library.custom_op("spyre::embedding", mutates_args=(), device_types="spyre")
def embedding(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if weight.device.type != "spyre":
        raise ValueError(f"embedding weight must be on spyre device, got {weight.device}")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"embedding indices must be integer, got {indices.dtype}")
    if weight.dtype not in _DTYPE_BITS:
        raise ValueError(f"embedding weight dtype {weight.dtype} unsupported (1p0: float16)")

    vocab, d_model = int(weight.shape[0]), int(weight.shape[1])
    element_bits = _DTYPE_BITS[weight.dtype]
    flat_idx = indices.reshape(-1).cpu().tolist()

    # IBR path: d_model > 512 → row-contiguous weight + int32 IBR table.
    # EBR path: d_model <= 512 → plane-major weight (existing behavior, unchanged).
    if d_model > _EBR_D_MODEL_MAX:
        return _embedding_ibr(weight, flat_idx, indices, vocab, d_model, element_bits)

    # EBR path (d_model <= 512, unchanged).
    launches, tokens_per_launch = build_embedding_launches(vocab, d_model, element_bits, flat_idx)

    # Assemble output on CPU then move to device once.
    # Device-side strided slice-assign (out[start:end] = ...) ignores the start
    # offset on spyre hardware — launch1's rows land at 0..43 instead of 256..299,
    # corrupting multi-launch output.  CPU staging avoids this entirely: each
    # launch's buffer is copied down and placed into the correct CPU rows; a single
    # .to(device) at the end moves the fully-assembled tensor to the device.
    # Performance: this costs one D2H copy per launch plus one H2D at the end.
    # Acceptable for embedding (output is small, <=1 MB/launch). Do NOT revert to a
    # device-side out[start:end] = ... slice-assign to "save" the copies — that
    # silently reintroduces the offset-corruption bug above (hardware-confirmed).
    out_cpu = torch.empty((len(flat_idx), d_model), dtype=weight.dtype)
    for (start, end), binary in launches:
        # Each binary writes exactly tokens_per_launch output rows regardless of
        # how many real tokens are in this launch (the final launch may be partial).
        # Allocate a full-sized device buffer so the binary never overruns a shorter
        # slice, then copy only the real rows to the CPU staging tensor.
        buf = torch.empty((tokens_per_launch, d_model), dtype=weight.dtype, device=weight.device)
        launch_kernel_from_bytes(binary, [weight, buf])
        out_cpu[start:end] = buf[: end - start].cpu()
    return out_cpu.to(weight.device).reshape(*indices.shape, d_model)


def _embedding_ibr(
    weight: torch.Tensor,
    flat_idx: list[int],
    indices: torch.Tensor,
    vocab: int,
    d_model: int,
    element_bits: int,
) -> torch.Tensor:
    """IBR-path embedding for d_model > 512.

    SILICON-PROVEN RECIPE (IBR-T7: both tokens max_diff=0):

    WEIGHT LAYOUT:
      weight (on device, plane-major) is brought back to CPU, reshaped to
      (vocab*spt, 64) — row-contiguous — and re-moved to device.  This costs
      one D2H + one H2D but is necessary to give LDIM the correct layout.

      spt = d_model // elements_per_stick.
      elements_per_stick = 64 at 16-bit (128 bytes / 2 bytes-per-element).

      Constraint: d_model must be a multiple of elements_per_stick.  The planning
      step (build_embedding_launches) validates this and raises ValueError if not.

    3-TENSOR LAUNCH LAYOUT (XLAT segment assignment = input tensor list order):
      XLAT[0] = weight_rc  (f16,   vocab*spt×64, row-contiguous) — segment 0
      XLAT[1] = ibr_table  (int32, 1×32)                         — segment 1
      XLAT[2] = buf        (f16,   tokens_per_launch*spt×64,
                                   row-contiguous)                — segment 2

    IBR TABLE:
      ibr_table[0, t] = (0 << seg_bits) + flat_idx_slice[t] * spt = flat_idx[t]*spt
      weight is at segment 0, so the segment term is 0.
      Built per-launch from the slice of flat_idx for that launch.
      Uses torch.from_numpy on the numpy int32 array returned by
      build_ibr_address_table(slice_indices, spt, segment=0, seg_bits).

    OUTPUT BUFFER + ASSEMBLY:
      buf is row-contiguous (tokens_per_launch*spt, 64): L3SU deposits spt
      sticks per token sequentially (no plane-major transposition).
      After D2H: buf.cpu().reshape(n_launch, spt, 64).reshape(n_launch, d_model)
      gives token-major (n_launch, d_model) — token t at rows [t*spt..(t+1)*spt).
      Only real rows [:n_real] are copied to out_cpu.
      Same CPU-staging pattern as the EBR path (one H2D at end).
    """
    eps = _ELEMENTS_PER_STICK[element_bits]   # 64 at 16-bit
    spt = d_model // eps                       # sticks per token row

    # Plan the launches (validates d_model % eps == 0).
    # _build_embedding_launches_ibr caches the binary per (d_model, element_bits)
    # and patches only the JCR_cnt IMM field per launch — no recompile per call.
    launches, tokens_per_launch = build_embedding_launches(
        vocab, d_model, element_bits, flat_idx,
    )

    # Row-contiguous weight: reshape (vocab, d_model) → (vocab*spt, 64).
    # CACHE: the reshape (D2H + reshape + H2D) costs ~seconds for large vocabs.
    # Cache by (id(weight), weight._version) to pay this cost only once per unique
    # weight tensor (i.e. once per model load).  weight._version increments on
    # in-place mutations, so stale hits from freed/reused addresses are avoided.
    weight_key = (id(weight), weight._version)
    if weight_key not in _ibr_weight_cache:
        # Pull to CPU, reshape to (vocab*spt, 64), move back to device.
        # SpyreTensorLayout([vocab*spt, 64], float16) → stride_map=[64, 1]:
        # one stick per row, sequential rows = row-contiguous layout for LDIM.
        weight_cpu = weight.cpu().reshape(vocab * spt, eps)
        _ibr_weight_cache[weight_key] = weight_cpu.to(weight.device)
    weight_rc = _ibr_weight_cache[weight_key]

    # IBR seg_bits from the generation: always 27 for rcudd1a.
    # Import from the gen registry rather than hard-coding 27.
    try:
        from sentient_codegen.gen.registry import get_generation
    except ImportError as e:
        raise EmbeddingHostError(
            "sentient_codegen not importable — source env.sh so codegen is "
            "on PYTHONPATH (required for IBR embedding launch)."
        ) from e
    gen = get_generation("rcudd1a")   # TODO: make target configurable per generation
    seg_bits = gen.hw.segment_size_bits

    # Weight is at XLAT[0] (segment 0) — silicon-proven IBR-T7 layout.
    # ibr_table is at XLAT[1] (segment 1), so IBR[t] = 0<<seg_bits + idx*spt = idx*spt.
    weight_segment = 0

    # n_launch = tokens_per_launch = CACHED_TOKENS = 32 (fixed geometry from
    # the cached binary; the hardware writes 32 rows per launch regardless of
    # n_real — only the L3LU gather loop count is patched to n_real, so the
    # hardware gathers n_real rows then stops; output rows [n_real..31] are
    # written but ignored by the caller).
    n_launch = tokens_per_launch

    out_cpu = torch.empty((len(flat_idx), d_model), dtype=weight.dtype)
    for (start, end), binary in launches:
        n_real = end - start

        # Build per-launch IBR table for real indices [start, end).
        # Entries [n_real..31] are 0 (don't-care: L3LU loop exits after n_real
        # iterations; these sticks are never read by the hardware).
        slice_indices = list(flat_idx[start:end])

        # segment=0: weight at XLAT[0], so IBR[t] = (0<<seg_bits) + idx*spt = idx*spt.
        ibr_np = build_ibr_address_table(
            slice_indices, spt, segment=weight_segment, seg_bits=seg_bits,
        )
        ibr_t = torch.from_numpy(ibr_np)  # int32, shape (1, 32)
        ibr_dev = ibr_t.to(weight.device)

        # Output buffer: row-contiguous (n_launch*spt, 64).
        # L3SU stores spt sticks per token contiguously; no plane-major transposition.
        # After D2H: reshape (n_launch*spt, 64) → (n_launch, spt, 64) → (n_launch, d_model).
        buf = torch.empty((n_launch * spt, 64), dtype=weight.dtype, device=weight.device)

        # Launch: [weight_rc, ibr_table, output_buf]
        # XLAT[0]=weight_rc (seg0), XLAT[1]=ibr_dev (seg1), XLAT[2]=buf (seg2).
        launch_kernel_from_bytes(binary, [weight_rc, ibr_dev, buf])

        # Extract real token rows from the row-contiguous buf.
        # buf row (t*spt+s) = token t plane s; reshape to (n_launch, spt, 64) then
        # flatten last two dims to get (n_launch, d_model).
        buf_rows = buf.cpu().reshape(n_launch, spt, 64).reshape(n_launch, d_model)
        out_cpu[start:end] = buf_rows[:n_real]

    return out_cpu.to(weight.device).reshape(*indices.shape, d_model)


@embedding.register_fake
def _(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return weight.new_empty((*indices.shape, weight.shape[1]))


# Route aten.embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False,
# sparse=False) on spyre to spyre::embedding. padding_idx is forward-irrelevant
# (only affects backward) so it is ignored; the training-only flags are rejected.
@torch.library.register_kernel("aten::embedding", ["spyre"])
def _aten_embedding_spyre(weight, indices, padding_idx=-1,
                           scale_grad_by_freq=False, sparse=False):
    if scale_grad_by_freq:
        raise NotImplementedError("spyre embedding: scale_grad_by_freq unsupported")
    if sparse:
        raise NotImplementedError("spyre embedding: sparse unsupported")
    return torch.ops.spyre.embedding(weight, indices)
