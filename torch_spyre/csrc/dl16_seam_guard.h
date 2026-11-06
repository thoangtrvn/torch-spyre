/*
 * Copyright 2025 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <util/sendefs.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "spyre_tensor_impl.h"

namespace spyre {

/*
 * The DL16 H2D converter-in seam ceiling (T-DC-17, silicon-witnessed
 * 2026-11-05, claim bf16pivot-sixitem-20261105-002 item 2).
 *
 * The flex DMA converter that turns host 16-bit floats into device DL16
 * words SATURATES SILENTLY at word 0x7DFF for inputs above it — including
 * values that are in-range for the DL16 format itself (the largest
 * in-range grid value 8522825728, module word 0x7FF8, comes back
 * seam-saturated). Silent-wrong defect class; this guard converts it to a
 * typed loud refusal on the host, before the transfer.
 *
 * 0x7DFF = sign 0 | exponent field 62 | fraction 0x1FF, which decodes in
 * the DL16 1-6-9 format (exponent bias 31) to
 *
 *   (2 - 2^-9) * 2^31 = 2^32 - 2^22 = 4290772992
 *
 * — one binade below the format max (2 - 2^-9) * 2^32 = 8581545984.
 *
 * TWO NUMBERS, TWO SITES: the format max is the compute/decode-path
 * boundary and is refused loudly by the fp32 pivot
 * (torch_spyre/_inductor/dtype_convert.py, Bf16Dl16RangeError); the seam
 * ceiling below is the H2D converter-in's usable limit. Neither is
 * corrected into the other. Values in (seam ceiling, format max] are
 * format-representable but unreachable through this seam; refusing them
 * is the honest narrowing this guard implements.
 */
constexpr double kDl16H2DSeamMax = 4290772992.0;  // decode(0x7DFF)

enum class Dl16H2DSeamBoundary {
  kOk = 0,
  kOverSeamCeiling = 1,
  kNonFinite = 2,
};

/* Classify one host-side float value against the H2D seam ceiling.
 * Exposed via torch_spyre._C.dl16_h2d_seam_boundary for oracle tests. */
inline Dl16H2DSeamBoundary dl16_h2d_seam_boundary(double value) {
  if (std::isnan(value) || std::isinf(value)) {
    return Dl16H2DSeamBoundary::kNonFinite;
  }
  if (value > kDl16H2DSeamMax || value < -kDl16H2DSeamMax) {
    return Dl16H2DSeamBoundary::kOverSeamCeiling;
  }
  return Dl16H2DSeamBoundary::kOk;
}

inline const char* dl16_h2d_seam_boundary_name(Dl16H2DSeamBoundary b) {
  switch (b) {
    case Dl16H2DSeamBoundary::kOk:
      return "ok";
    case Dl16H2DSeamBoundary::kOverSeamCeiling:
      return "over-seam-ceiling";
    case Dl16H2DSeamBoundary::kNonFinite:
      return "non-finite";
  }
  return "ok";
}

/*
 * Typed loud refusal for a host-to-device copy whose destination is a
 * SEN169_FP16 (DL16) device tensor: scan the host tensor's elements and
 * TORCH_CHECK-refuse any value the DMA converter-in would silently
 * saturate or corrupt (above the seam ceiling, or non-finite — the
 * converter has no DL16 inf/nan word and its field-63 marker decodes as a
 * large finite value).
 *
 * No-op unless the destination device dtype is SEN169_FP16 and the host
 * scalar type is a 16-bit float: fp32 H2D is a pure bit-move
 * (silicon-witnessed, claim item 3) and integer formats have no float
 * seam. fp16's format max (65504) is eight binades below the ceiling, so
 * only bf16 can cross it in practice, but both 16-bit floats are scanned
 * so the guard is derived from the seam's dtype class, not a guess about
 * which dtype reaches it.
 *
 * The scan walks the tensor's logical view (sizes/strides/storage offset
 * respected) — the same elements the stride-aware DMA converts.
 */
template <typename DecodeElem>
inline void check_dl16_h2d_seam_scan(const at::Tensor& cpu_tensor,
                                     DecodeElem decode_elem) {
  const auto sizes = cpu_tensor.sizes();
  const auto strides = cpu_tensor.strides();
  const int64_t numel = cpu_tensor.numel();
  if (numel == 0) {
    return;
  }
  const uint16_t* data =
      static_cast<const uint16_t*>(cpu_tensor.const_data_ptr());

  // Odometer over the logical view: offset tracks the element offset of
  // the current index vector, so strided views (broadcasts, slices) are
  // scanned exactly as the DMA sees them.
  std::vector<int64_t> idx(sizes.size(), 0);
  int64_t offset = 0;
  for (int64_t linear = 0; linear < numel; ++linear) {
    const double value = decode_elem(data[offset]);
    const auto boundary = dl16_h2d_seam_boundary(value);
    if (boundary != Dl16H2DSeamBoundary::kOk) {
      TORCH_CHECK(
          false, "Spyre H2D copy refused at the DL16 converter-in seam: ",
          dl16_h2d_seam_boundary_name(boundary), " value ", value,
          " at element ", linear, " of a ", cpu_tensor.scalar_type(),
          " host tensor bound for a SEN169_FP16 device tensor. The DMA "
          "converter-in silently saturates above ",
          kDl16H2DSeamMax, " (word 0x7DFF) — including format-in-range "
          "values up to the DL16 max 8581545984. Rescale the tensor or "
          "convert through the fp32 pivot "
          "(torch_spyre._inductor.dtype_convert) before upload.");
    }
    for (int64_t d = static_cast<int64_t>(sizes.size()) - 1; d >= 0; --d) {
      ++idx[d];
      offset += strides[d];
      if (idx[d] < sizes[d]) {
        break;
      }
      idx[d] = 0;
      offset -= strides[d] * sizes[d];
    }
  }
}

inline void check_dl16_h2d_seam(const at::Tensor& cpu_tensor,
                                const SpyreTensorLayout& stl) {
  if (stl.device_dtype != DataFormats::SEN169_FP16) {
    return;
  }
  const auto scalar_type = cpu_tensor.scalar_type();
  if (scalar_type == c10::kBFloat16) {
    // bf16 is truncated fp32: the upper-half upcast is exact.
    check_dl16_h2d_seam_scan(cpu_tensor, [](uint16_t word) -> double {
      uint32_t bits = static_cast<uint32_t>(word) << 16;
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      return static_cast<double>(f);
    });
  } else if (scalar_type == c10::kHalf) {
    check_dl16_h2d_seam_scan(cpu_tensor, [](uint16_t word) -> double {
      return static_cast<double>(static_cast<float>(
          c10::Half(word, c10::Half::from_bits())));
    });
  }
}

}  // namespace spyre
