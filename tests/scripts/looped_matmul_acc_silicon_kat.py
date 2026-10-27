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

"""Frontend-routed looped-matmul accumulator silicon KAT (pre-registered).

Measures whether a ``torch.compile``-routed (Inductor -> triton-spyre ->
sentient_codegen) fp16 looped matmul with K > 64 produces numerically
correct output ON DEVICE under RANDOM inputs. This is the device-numerics
closure of the loop-carried reduction-accumulator upcast override
(``enable_spyre_context`` -> ``upcast_acc_dtype``, merge ff74490): the
patch's landed evidence is emission-assert only (TORCH_SPYRE_CODEGEN_ONLY=1
output-code ``tl.full`` dtype text, tests/test_looped_matmul_acc_dtype.py);
no run has ever witnessed the NUMERICS of the compiled artifact on silicon.

HOME-JUSTIFICATION (torch-spyre, not codegen/scripts): the KAT's subject is
the FRONTEND route — torch.compile -> Inductor -> triton-spyre backend ->
sentient_codegen — so it must import torch_spyre, build spyre-device
operands, and launch through the torch path. The codegen/scripts KATs
(reduction_silicon_measurement_kat.py, softmax_silicon_measurement_kat.py)
are DIRECT-codegen launches (emit + launch_kernel_from_bytes) which is
exactly the route this KAT must NOT take.

PRE-REGISTRATION (committed before any hardware launch; the criteria below
are FROZEN — no post-hoc adjudication. A mismatch is a finding, not an
embarrassment to be patched into green):

  ADDENDUM 2026-10-27 (harness hardening post-commit 8c51d42; NO scoring
  criterion constant moved): (1) score() now fails non-finite outputs
  explicitly (r1 M0 — a naive `err > bound` laundered NaN to PASS; the
  frozen criterion "EVERY element must satisfy |got-ref| <= bound" always
  implied this; the repair enforces it and adds R-SCORE-NAN red-first);
  (2) the NO-SUBNORMAL-BAND and R-SCORE-HONEST registrations were
  rewritten to state their true scope (reference-lattice-only band
  argument; same-model circularity disclosed) per the r1 derivation
  review. The frozen bound constants, shapes, distributions and seeds
  are unchanged.

  ROUTE — ``torch.compile(lambda a, b: a @ b, dynamic=False)`` on fp16
  spyre tensors. Route authenticity is ASSERTED, not assumed: the emitted
  Inductor output code must carry the fp32 loop-carried accumulator
  (``tl.full(..., tl.float32)``, no fp16 init) and ``tl.dot``; a run whose
  artifact fails that assert is a KAT failure (wrong route), never scored.

  SHAPES (K > 64 looped matmul; the accumulator seam only exists there —
  K <= 64 has no loop-carried accumulator):
    LM1 (M=32,  K=256, N=64)  — the SP3K HW-witnessed envelope row (ks=4,
                                C=1); the most likely first-admitted shape
                                once the structured route opens.
    LM2 (M=8,   K=384, N=64)  — clean-K envelope boundary (ks=6; ks=8 is
                                the #311 TILE_SIZE fail-loud seam).
    LM3 (M=128, K=512, N=128) — the accumulator lane's red-first repro
                                shape (the emission the patch fixed). Also
                                beyond the #311 ks<=6 clean-K envelope, so
                                it may fail-loud at the resolver even after
                                structured admission lands; recorded, never
                                laundered.

  CURRENT ADMISSION STATE (2026-10-26, offline, codegen current): EVERY
  K>64 matmul module is refused at compile time by the structured-route
  admission gate — ``StructuredProductionSubsetRefusedError`` (T-SR-29
  typed refusal; corroborated by codegen/docs/structured_acceptance_census.json:
  0 of 244 tt.dot-carrying corpus matmul cases with K>64 are structured-
  admitted, all 90 admitted are K=64). The KAT records the typed refusal
  per row in offline mode and REFUSES TO FABRICATE a launch. The launch
  path is wired and frozen so the day admission opens, the HW claim is a
  coordinator message, not a redesign.

  INPUT DISTRIBUTIONS (seed 20261026 + M*1000003 + K*10007 + N*101 + h,
  randint(-h, h+1) -> fp16 / 8, the established KAT convention): every
  operand is a multiple of 1/8 with |x| <= h/8, hence DL16-exact — the
  input side contributes ZERO rounding:
    A (clean, all rows)  h=32  => |x| <= 4.
    B (wide,  all rows)  h=64  => |x| <= 8; |ref| <= K*64 <= 32768 < 65504
                           (fp16 max, 2x margin) — exercises the deep DL16
                           ulp ladder + cancellation while every partial
                           sum stays in the fp16 normal band.

  NO-SUBNORMAL-BAND (recorded, not assumed — and scoped to what it proves):
  products of 1/8-multiples are multiples of 2^-6, so every nonzero
  product, EXACT partial sum and reference is >= 2^-6 — far above the
  fp16 min normal 2^-14 and the DL16 min normal 2^-30 (witnessed per
  seed by test_no_subnormal_band_holds). SCOPE LIMIT (r1 m2, recorded):
  this lattice argument covers the REFERENCE side only. The DEVICE
  accumulator leaves the 2^-6 lattice after its first DL16 grid round
  (DL16 spacing in binade e is 2^(e-9)), so a strongly cancelling
  column can carry a sub-2^-14 residual — inside the fp16 subnormal
  readback band, where DL16->fp16 readback is NOT exact (at binade -16
  and below, DL16 spacing is finer than the fp16 subnormal quantum
  2^-24).
  The conclusion survives by a DIFFERENT argument: for a cancelling
  column the bound is 2^-10 * (sum|s_k| + sum|p_k|), which at these
  magnitudes is orders above any subnormal-readback error; the
  subnormal readback defect class is dominated, not excluded.

  MATCH DEFINITION (per shape x distribution; EVERY output element must
  satisfy it for PASS — backward-error form vs the float64 reference):
      |got - ref| <= 2^-10 * (sum_k |s_k| + sum_k |p_k|) + 2^-8 * |ref|
    where p_k = a[m,k]*b[k,n] (exact in float64: products of 2^-3-multiples
    are 2^-6-multiples), s_k = the running partial sum after the k-th
    accumulate step (float64, from the seeded input — deterministic), and
    ref = s_K.

    DERIVATION (cited, not tuned — same constants and mechanism as
    codegen/scripts/reduction_silicon_measurement_kat.py):
      - The AIU 1.0 datapath has NO accumulator wider than the DL16 store
        grid on the K-reduction path (tests/test_op_matmul.py law: "the PT
        FP accumulator IS DL16 (no wider accumulator; 1p5 gets FP24)").
        DL16 is a 1-6-9 float: 10-bit significand, round-half-even
        (sentient_codegen/emulator/dl16_round.py, W1d-pinned).
      - ACCUMULATE TERM 2^-10 * sum_k |s_k|: each of the K sequential
        accumulate steps rounds once to the DL16 grid at <= 0.5 ulp of the
        value written; ulp(x) <= |x| * 2^-9 for a 10-bit significand, so
        0.5 ulp <= |s_k| * 2^-10. If the true internal accumulator is
        WIDER (the dl16_round accumulator-width sweep models 10..13
        mantissa bits truncated on write-out), the per-step error is only
        smaller — the bound is an upper bound either way. FIRST-ORDER
        NOTE (recorded): the term bounds each round against the EXACT
        s_k; error feedback across steps compounds the worst case by at
        most (1+2^-10)^K - 1 <= 0.65 at K=512 — the same first-order form
        the reduction KAT uses, and absorbed by the >= 6x margin the
        honest closed forms exhibit against the criterion offline.
        DISCLOSED POWER LIMIT (recorded): this criterion cannot by itself
        distinguish an fp32 accumulator from a hypothetical fp16 one —
        an fp16 accumulate round is <= 0.5 ulp_fp16 <= |s_k| * 2^-11,
        inside the 2^-10 budget. The accumulator-dtype seam is witnessed
        by the route-authenticity emission assert (fp32 loop-carried
        init + chain, no fp16 init), not by the numerics; the numerics
        close the DL16-dominated datapath.
      - PRODUCT TERM 2^-10 * sum_k |p_k|: whether the PE FMA rounds the
        product to the DL16 grid before the add is not modeled by the
        closed form; each such round would be <= 0.5 ulp <= |p_k| * 2^-10
        (p_k >= 2^-6 when nonzero, normal band). Budgeted, not assumed
        absent — the fail-safe direction.
      - OUTPUT TERM 2^-8 * |ref|: the output side (PE forward -> SFP
        staging -> store) is budgeted 2 ulp at output magnitude, and
        2 ulp <= 2 * |ref| * 2^-9 = |ref| * 2^-8 (the reduction KAT's
        output-side constant, same derivation). The DL16->fp16 readback is
        exact in this band (fp16's 11-bit significand strictly contains
        DL16's 10, and every value is fp16-normal).
      - CANCELLATION: the backward form scales with the partial-sum
        magnitudes sum_k |s_k|, never with |ref| alone — a cancelling
        column tightens nothing (the row-sum-criterion law).
      The bound is COMPOSED from per-mechanism terms; no constant was
      fitted to any observed output.

  CLOSED-FORM LEDGER (recorded, never gated — the maturity measure): the
  golden-free DL16-sequential closed form (acc = round_to_dl16(acc + p_k)
  per step, float64 between rounds, exact products) is computed per
  element; the count of device outputs differing from it and the max
  deviation are recorded. A disagreement is a datapath-model finding, not
  a KAT failure.

  RED-FIRST (offline, no device; pinned in
  tests/test_looped_matmul_acc_silicon_kat.py):
    R-EMISSION — with the accumulator override neutralized (pre-patch
      simulation: ``upcast_acc_dtype`` identity on fp16), the compile of
      every row dies at triton-spyre's loop-carried type assert. This is
      the defect the patch fixed, reproduced from current source.
    R-SCORE-HONEST — the DL16 closed form itself PASSES the criterion
      (same-model circularity, disclosed: surrogate, bound and rounder
      share one mechanism model, so this validates arithmetic dominance
      and scorer plumbing — GROSS SENSITIVITY ONLY, not mechanism
      coverage; its honest-variant value is witnessing the offline
      margin, recorded as a number in every run log).
    R-SCORE-TRUNCATED — a surrogate that accumulates only the first K//2
      products FAILS the criterion (dropped-magnitude class).
    R-SCORE-PERMUTED — a surrogate scored against k-tile-swapped operands
      FAILS the criterion (mis-addressing class).
    R-SCORE-NAN — a single NaN element in an otherwise-honest output
      FAILS the criterion and is counted non-finite (r1 M0: a naive
      `err > bound` gate launders NaN to PASS; silicon-NaN is a known
      law family, dl16-ffff PE FMINMAX).

ISOLATION LAW: every row's COMPILE runs in its OWN subprocess with a
fresh Python (the Triton in-memory cache survives torch._dynamo.reset();
the FLAT-band cache-hygiene law) and, offline, TORCH_SPYRE_CODEGEN_ONLY=1
so the compile is fully exercised (admission verdict included) but never
launched. The red-first scorer self-checks are pure numerics (no compile,
no Triton-cache contact) and run in the parent. PROVENANCE NOTE: the
worker subprocess imports the INSTALLED torch_spyre/triton-spyre (the
box's rsync'd ff74490 trees) — the record names the install target, and
the run log is the record of what actually executed.

Usage (on the box, under a coordinated HW claim — HW Probe Claim
Coordination is MANDATORY; this script never edits box state):

    source <env> && rm -rf ~/.sentient_cache/* && \
    timeout 2000 python3 tests/scripts/looped_matmul_acc_silicon_kat.py \
        --launch --json-out <log>.json

(The outer timeout must exceed 6 rows x the 300 s per-row budget.)

Offline (no device, no launch): omit --launch. Builds every row's inputs,
reference, bound and closed form, attempts the compile in a codegen-only
subprocess, records the admission verdict (currently the T-SR-29 typed
refusal per row), and runs the red-first scorer self-checks. Not shippable
routing; not a test; the record of this lane is the run log + verdict
table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Pre-registered constants (FROZEN — see the module docstring derivations).
# ---------------------------------------------------------------------------
_SEED_BASE = 20261026

#: (key, M, K, N) — the pre-registered shape set. K > 64 on every row: the
#: loop-carried accumulator (the seam under test) only exists past one
#: K-stick (64 fp16 elements).
SHAPE_ROWS: tuple[tuple[str, int, int, int], ...] = (
    ("LM1", 32, 256, 64),
    ("LM2", 8, 384, 64),
    ("LM3", 128, 512, 128),
)

#: (key, randint half-range h) — pre-registered distributions. A = clean;
#: B = wide (deep ulp ladder + cancellation, |ref| <= K*64 < 65504).
DISTRIBUTIONS: tuple[tuple[str, int], ...] = (
    ("A", 32),
    ("B", 64),
)

#: Frozen match-criterion scales (derived, not tuned; see docstring).
_ACCUM_TERM_SCALE = 2.0 ** -10   # per-step DL16 store round: <= 0.5 ulp(s_k)
_PRODUCT_TERM_SCALE = 2.0 ** -10  # per-product DL16 round, if the FMA rounds
_OUTPUT_TERM_SCALE = 2.0 ** -8   # 2 ulp at output magnitude (forward+stage+store)

#: DL16 is a 1-6-9 float: 9 explicit mantissa bits (10-bit significand),
#: bias-31 exponent, round-half-even (sentient_codegen/emulator/dl16_round.py).
_DL16_MANTISSA_BITS = 9

#: Elements per stick at 16-bit — the K-chunk granularity of the loop.
_ELEMS_PER_STICK = 64

_ROW_KEYS = {key: (m, k, n) for key, m, k, n in SHAPE_ROWS}
_DIST_KEYS = {key: h for key, h in DISTRIBUTIONS}


def seed_for(row_key: str, m: int, k: int, n: int, h: int) -> int:
    """Deterministic per-(row, distribution) seed (documented, stable)."""
    del row_key  # (M, K, N) already names the row; the key is documentation.
    return _SEED_BASE + m * 1_000_003 + k * 10_007 + n * 101 + h


def row_shape(row_key: str) -> tuple[int, int, int]:
    """(M, K, N) for a pre-registered row key (fail-loud on unknown keys)."""
    try:
        return _ROW_KEYS[row_key]
    except KeyError:
        raise ValueError(
            f"unknown row key {row_key!r}: pre-registered rows are "
            f"{sorted(_ROW_KEYS)}"
        ) from None


def dist_half_range(dist_key: str) -> int:
    """randint half-range h for a pre-registered distribution key."""
    try:
        return _DIST_KEYS[dist_key]
    except KeyError:
        raise ValueError(
            f"unknown distribution key {dist_key!r}: pre-registered "
            f"distributions are {sorted(_DIST_KEYS)}"
        ) from None


# ---------------------------------------------------------------------------
# Inputs, reference, bound, closed form — pure numerics, no torch_spyre.
# ---------------------------------------------------------------------------
def build_inputs(m: int, k: int, n: int, h: int, seed: int):
    """Seeded DL16-exact operands: randint(-h, h+1) -> fp16 / 8.

    Every element is a multiple of 1/8 with |x| <= h/8 <= 8, hence exactly
    representable in DL16 (and fp16) — the input side rounds nothing.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    a = (torch.randint(-h, h + 1, (m, k), generator=gen)
         .to(torch.float16) / 8.0)
    b = (torch.randint(-h, h + 1, (k, n), generator=gen)
         .to(torch.float16) / 8.0)
    return a, b


def round_to_dl16(x):
    """Round float64 to the DL16 grid (10-bit significand, half-even).

    Mirrors sentient_codegen/emulator/dl16_round.py at W=9 (frexp/rint
    decomposition); zero passes through and the exponent is NOT clamped —
    every value in this KAT's band (>= 2^-6) is DL16-normal, so clamping
    is unreachable by construction. Kept local so the scorer runs without
    the codegen stack; tests/test_looped_matmul_acc_silicon_kat.py
    cross-checks equivalence against the reference when it is importable.
    """
    import numpy as np

    arr = np.asarray(x, dtype=np.float64)
    out = np.array(arr, dtype=np.float64, copy=True)
    nonzero = arr != 0.0
    if not np.any(nonzero):
        return out
    vals = arr[nonzero]
    m, e = np.frexp(vals)            # vals = m * 2**e, m in [0.5, 1)
    m2 = m * 2.0                     # m2 in [1, 2)
    e2 = e - 1                       # vals = m2 * 2**e2
    scale = float(1 << _DL16_MANTISSA_BITS)
    q = np.rint((m2 - 1.0) * scale) / scale
    rounded = (1.0 + q) * np.ldexp(np.ones_like(e2, dtype=np.float64), e2)
    out[nonzero] = np.sign(vals) * np.abs(rounded)
    return out


def reference_and_bound(a, b):
    """Float64 reference and the frozen per-element backward-error bound.

    Returns (ref, bound) as [M, N] float64 tensors. p_k and the running
    s_k are exact in float64 (operands are 2^-3-multiples, products
    2^-6-multiples, K <= 512 — no float64 rounding at these magnitudes).
    """
    import torch

    a64 = a.double()
    b64 = b.double()
    # products[m, k, n] — exact
    p = a64.unsqueeze(2) * b64.unsqueeze(0)
    # running partial sums s_k along k — exact
    s = torch.cumsum(p, dim=1)
    ref = s[:, -1, :]
    bound = (
        _ACCUM_TERM_SCALE * s.abs().sum(dim=1)
        + _PRODUCT_TERM_SCALE * p.abs().sum(dim=1)
        + _OUTPUT_TERM_SCALE * ref.abs()
    )
    return ref, bound


def dl16_sequential_closed_form(a, b, *, pre_round_products: bool = False):
    """Golden-free DL16-sequential accumulate closed form (ledger only).

    acc = round_to_dl16(acc + p_k) per k, float64 between rounds, exact
    products (the FMA product round, if the datapath makes one, is NOT
    modeled — that is what the bound's product term budgets). Never gated.

    ``pre_round_products=True`` additionally rounds each product to the
    DL16 grid before the add — the honest variant used by the red-first
    self-checks to show the criterion also admits the product-round
    mechanism, not just the accumulate-round one.
    """
    import numpy as np
    import torch

    a64 = a.double().numpy()
    b64 = b.double().numpy()
    m, k, n = a64.shape[0], a64.shape[1], b64.shape[1]
    acc = np.zeros((m, n), dtype=np.float64)
    for i in range(k):
        p = a64[:, i, None] * b64[None, i, :]
        if pre_round_products:
            p = round_to_dl16(p)
        acc = round_to_dl16(acc + p)
    return torch.from_numpy(acc)


def score(got, ref, bound):
    """Score an output against the frozen criterion. ALL elements must hold.

    ``got`` is fp16 device output (any torch tensor); ref/bound are the
    float64 reference pair. Returns a verdict dict — never raises on a
    numeric mismatch (a mismatch is a finding, recorded in the table).
    """
    import torch

    got64 = got.detach().to("cpu").double()
    err = (got64 - ref).abs()
    # NaN laundering guard (r1 M0): NaN comparisons are False, so
    # `err > bound` alone PASSES a NaN element. The criterion says EVERY
    # element must satisfy |got - ref| <= bound — a non-finite got cannot
    # satisfy it. Silicon-NaN is a known law family here (dl16-ffff PE
    # FMINMAX), so this is a reachable false-green, not a hypothetical.
    nonfinite = ~torch.isfinite(got64)
    fails = ~(err <= bound) | nonfinite
    # Headroom err/bound (PASS iff <= 1); guard the divide where bound == 0
    # (an all-zero column: err is 0 there too, so it never fails).
    slack = torch.where(
        bound > 0, err / bound.clamp_min(2.0 ** -1074), torch.zeros_like(err)
    )
    return {
        "n_elements": int(err.numel()),
        "n_fail": int(fails.sum().item()),
        "n_nonfinite": int(nonfinite.sum().item()),
        "max_abs_err": float(err.max().item()),
        "max_slack": float(slack.max().item()),
        "pass": bool(not fails.any().item()),
    }


def closed_form_ledger_stats(got, ledger):
    """Recorded, never gated: deviation of the output from the closed form."""
    got64 = got.detach().to("cpu").double()
    diff = (got64 - ledger).abs()
    return {
        "n_differ": int((got64 != ledger).sum().item()),
        "max_deviation": float(diff.max().item()),
    }


# ---------------------------------------------------------------------------
# Route: torch.compile through the spyre backend (the FRONTEND route).
# ---------------------------------------------------------------------------
_ACC_INIT_RE_TEMPLATE = (
    r"(\w+) = tl\.full\(\[[A-Za-z_]\w*, [A-Za-z_]\w*\], 0, tl\.{dtype}\)"
)

#: The emitted loop-carried accumulator initialization: a TWO-DIMENSIONAL
#: fp32 zero-init (the [YBLOCK, XBLOCK] output tile — the 1-D ``tl.int1``
#: mask fulls must not match), e.g.
#: ``_tmp6 = tl.full([YBLOCK, XBLOCK], 0, tl.float32)``. Captures the
#: variable name so the full accumulate CHAIN can be verified, not just
#: the dtype (same seam as tests/test_looped_matmul_acc_dtype.py, made
#: def-site-strict: an unrelated fp32 full elsewhere in the kernel must
#: not authenticate the route).
FP32_ACC_INIT_RE = _ACC_INIT_RE_TEMPLATE.format(dtype="float32")
FP16_ACC_INIT_RE = _ACC_INIT_RE_TEMPLATE.format(dtype="float16")


def route_authenticity(output_text: str) -> dict:
    """Assert the compiled artifact is Inductor's looped matmul, fp32-acc.

    The whole point of this KAT is the FRONTEND route; an artifact without
    ``tl.dot`` or without the fp32 loop-carried accumulator is a wrong-route
    failure and is never scored. The accumulator check is CHAIN-shaped, not
    dtype-shaped: the 2-D fp32 zero-init variable must be re-assigned from
    an expression that reads itself (the K-loop accumulate emission is
    ``tmp7 = _tmp6 + tmp5`` / ``_tmp6 = tmp7``).
    """
    fp32_match = re.search(FP32_ACC_INIT_RE, output_text)
    fp32_chain = False
    if fp32_match is not None:
        var = re.escape(fp32_match.group(1))
        fp32_chain = (
            # the accumulator feeds an add (self-read)...
            re.search(rf"= {var} \+", output_text) is not None
            # ...and is re-assigned at least twice (init + loop back-edge)
            and len(re.findall(rf"{var} = ", output_text)) >= 2
        )
    return {
        "tl_dot": "tl.dot" in output_text,
        "fp32_loop_carried_acc": fp32_chain,
        "fp16_loop_carried_acc": re.search(FP16_ACC_INIT_RE, output_text) is not None,
    }


def run_row(row_key: str, dist_key: str, *, launch: bool, result_out: str) -> None:
    """Single-row worker: compile (and launch if asked), score, write JSON.

    Runs in its own subprocess (spawned by :func:`run`); the compile is
    the real torch.compile route — offline under
    TORCH_SPYRE_CODEGEN_ONLY=1 (full compile incl. the admission verdict,
    no launch), launch mode without it. Route authenticity is asserted
    from the Inductor output code this process prints under
    TORCH_LOGS=output_code.
    """
    m, k, n = row_shape(row_key)
    h = dist_half_range(dist_key)
    seed = seed_for(row_key, m, k, n, h)

    import torch
    import torch_spyre

    torch_spyre._autoload()

    record: dict = {
        "row": row_key,
        "dist": dist_key,
        "M": m,
        "K": k,
        "N": n,
        "h": h,
        "seed": seed,
        "mode": "launch" if launch else "offline",
    }

    a, b = build_inputs(m, k, n, h, seed)
    ref, bound = reference_and_bound(a, b)
    ledger = dl16_sequential_closed_form(a, b)
    record["reference_max_abs"] = float(ref.abs().max().item())
    record["bound_max"] = float(bound.max().item())

    def fn(a, b):
        return a @ b

    from torch_spyre._runtime import SpyreRuntimeUnavailableError

    # Allocate FIRST, outside the compile try: a codegen-only environment
    # that refuses allocation must be recorded as an environment failure,
    # never misread as "compiled".
    try:
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
    except SpyreRuntimeUnavailableError:
        record["status"] = "ENV_REFUSED_ALLOCATION"
        _write_record(record, result_out)
        return

    try:
        compiled = torch.compile(fn, dynamic=False)
        got = compiled(a_dev, b_dev)
    except SpyreRuntimeUnavailableError:
        # Codegen-only mode: allocation succeeded and the compiled call ran
        # to the runtime's launch refusal — the compile (admission verdict
        # included) executed, the kernel did not.
        record["status"] = "compiled_not_launched"
        record["route_authenticity"] = route_authenticity(_output_code_text())
        _write_record(record, result_out)
        return
    except Exception as exc:  # typed refusals are the recorded outcome
        record["status"] = "refused"
        record["refusal_type"] = type(exc).__name__
        message = str(exc)
        record["tsr29_typed"] = "structured production subset refused" in message
        if record["tsr29_typed"]:
            inside = message.split("refusal codes: ", 1)
            if len(inside) == 2:
                record["refusal_codes"] = [
                    c.strip() for c in inside[1].split(")", 1)[0].split(",")
                ]
            else:
                # Typed marker present but no parseable code list — record
                # the raw text; never fabricate an empty/partial code list.
                record["refusal_detail"] = message[:400]
        else:
            record["refusal_detail"] = message[:400]
        # The emission seam is still witnessable on a refused compile: the
        # output code is printed before the downstream admission runs.
        record["route_authenticity"] = route_authenticity(_output_code_text())
        _write_record(record, result_out)
        return

    if not launch:
        raise RuntimeError(
            "offline worker reached a launch: TORCH_SPYRE_CODEGEN_ONLY did "
            "not stop the route (env must be set by the parent)"
        )
    record["status"] = "compiled_and_launched"
    record["route_authenticity"] = route_authenticity(_output_code_text())
    if not all(
        [
            record["route_authenticity"]["tl_dot"],
            record["route_authenticity"]["fp32_loop_carried_acc"],
            not record["route_authenticity"]["fp16_loop_carried_acc"],
        ]
    ):
        # Wrong route is a KAT failure, never scored.
        record["status"] = "ROUTE_AUTHENTICITY_FAILED"
        _write_record(record, result_out)
        return

    record["score"] = score(got, ref, bound)
    record["closed_form_ledger"] = closed_form_ledger_stats(got, ledger)
    _write_record(record, result_out)


def _write_record(record: dict, result_out: str) -> None:
    with open(result_out, "w") as fh:
        json.dump(record, fh, indent=1)


def _output_code_text() -> str:
    """The Inductor output code this process printed (TORCH_LOGS=output_code).

    torch's logging sinks to stderr; the parent captures this process's
    stderr and hands it back via ``_LM_KAT_STDERR_CAPTURE``.
    """
    path = os.environ.get("_LM_KAT_STDERR_CAPTURE")
    if not path or not os.path.exists(path):
        return ""
    # This process's own stderr IS the capture (inherited fd): flush the
    # Python-level buffer before reading back so a partially-buffered
    # output-code log cannot produce a spurious all-False record.
    sys.stderr.flush()
    with open(path) as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Red-first scorer self-checks (offline, no device, no compile).
# ---------------------------------------------------------------------------
def red_first_self_checks():
    """Validate the instrument: honest model passes, defects fail.

    Returns a list of (name, passed) pairs. Everything here is pure
    numerics on the seeded inputs — no compile, no device.
    """
    import torch

    results = []
    for row_key, m, k, n in SHAPE_ROWS:
        for dist_key, h in DISTRIBUTIONS:
            seed = seed_for(row_key, m, k, n, h)
            a, b = build_inputs(m, k, n, h, seed)
            ref, bound = reference_and_bound(a, b)
            ledger = dl16_sequential_closed_form(a, b)
            tag = f"{row_key}/{dist_key}"

            # R-SCORE-HONEST: the DL16 closed form passes (same-model
            # circularity, disclosed in the registration — arithmetic
            # dominance and scorer plumbing only; the margin it exhibits
            # is recorded as a number in the run log).
            honest = score(ledger, ref, bound)
            results.append((f"{tag}:honest-closed-form-passes", honest["pass"]))

            # R-SCORE-HONEST-PRE-ROUND: the product-round variant must also
            # pass — the criterion admits BOTH budgeted mechanisms, not just
            # the one the ledger models.
            pre_round = score(
                dl16_sequential_closed_form(a, b, pre_round_products=True),
                ref, bound,
            )
            results.append((f"{tag}:honest-pre-round-passes", pre_round["pass"]))

            # R-SCORE-TRUNCATED: accumulate only the first K//2 products —
            # the dropped-magnitude class must FAIL. (b's K is axis 0,
            # a's K is axis 1.)
            half = dl16_sequential_closed_form(a[:, : k // 2], b[: k // 2])
            truncated = score(half, ref, bound)
            results.append((f"{tag}:truncated-k-fails", not truncated["pass"]))

            # R-SCORE-PERMUTED: the honest closed form of k-tile-swapped
            # operands vs the true reference — the mis-addressing class
            # must FAIL (swapping two whole K-sticks changes every column
            # sum at O(1), far past the 2^-10-scale bound).
            a_swapped = a.clone()
            a_swapped[:, :_ELEMS_PER_STICK] = a[:, _ELEMS_PER_STICK:2 * _ELEMS_PER_STICK]
            a_swapped[:, _ELEMS_PER_STICK:2 * _ELEMS_PER_STICK] = a[:, :_ELEMS_PER_STICK]
            permuted = score(dl16_sequential_closed_form(a_swapped, b), ref, bound)
            results.append((f"{tag}:k-tile-permuted-fails", not permuted["pass"]))

            # R-SCORE-NAN: a single NaN element in the device output must
            # FAIL the criterion (r1 M0 red-first: NaN comparisons are
            # False, so a naive `err > bound` gate laundered NaN to PASS;
            # silicon-NaN is a known law family, dl16-ffff PE FMINMAX).
            nan_seed = ledger.clone()
            nan_seed.view(-1)[nan_seed.numel() // 2] = float("nan")
            nan_verdict = score(nan_seed, ref, bound)
            results.append((f"{tag}:nan-element-fails", not nan_verdict["pass"]))
            results.append(
                (f"{tag}:nan-counted-nonfinite", nan_verdict["n_nonfinite"] == 1)
            )
    return results


# ---------------------------------------------------------------------------
# Parent runner: one fresh subprocess per row (cache-hygiene isolation law).
# ---------------------------------------------------------------------------
def _spawn_row(row_key: str, dist_key: str, *, launch: bool, tmp: str) -> dict:
    m, k, n = row_shape(row_key)
    result_out = os.path.join(
        tmp, f"result_{row_key}_{dist_key}.json"
    )
    stderr_capture = os.path.join(tmp, f"stderr_{row_key}_{dist_key}.txt")
    env = os.environ.copy()
    env["TORCH_LOGS"] = "output_code"
    env["_LM_KAT_STDERR_CAPTURE"] = stderr_capture
    # Env hygiene (same neutralization as the pytest harness): stray
    # autoload/descriptor flags must not silently change the compiled route.
    env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    env.pop("TORCH_SPYRE_EXPERIMENTAL_DESCRIPTOR_POINTWISE", None)
    if not launch:
        # Full compile, admission verdict included, never launched.
        env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    else:
        env.pop("TORCH_SPYRE_CODEGEN_ONLY", None)

    crash_record = {
        "row": row_key,
        "dist": dist_key,
        "M": m,
        "K": k,
        "N": n,
        "h": dist_half_range(dist_key),
        "seed": seed_for(row_key, m, k, n, dist_half_range(dist_key)),
        "mode": "launch" if launch else "offline",
    }

    with open(stderr_capture, "w") as stderr_fh:
        try:
            proc = subprocess.run(
                [
                    sys.executable, os.path.abspath(__file__),
                    "--worker", row_key, dist_key,
                    *(["--launch"] if launch else []),
                    "--result-out", result_out,
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=stderr_fh,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            # A hung row (the documented on-box deadlock class) is a
            # RECORD, never a parent crash — the verdict table must survive.
            crash_record.update(status="WORKER_TIMEOUT", timeout_s=300)
            return crash_record

    if proc.returncode != 0 or not os.path.exists(result_out):
        crash_record.update(
            status="WORKER_CRASHED",
            returncode=proc.returncode,
            stdout_tail=proc.stdout[-800:],
        )
        return crash_record
    with open(result_out) as fh:
        return json.load(fh)


#: The route-authenticity triple EVERY row must carry (refused rows
#: included — the output code is emitted before the admission gate runs).
_EXPECTED_ROUTE_AUTHENTICITY = {
    "tl_dot": True,
    "fp32_loop_carried_acc": True,
    "fp16_loop_carried_acc": False,
}

#: Verdicts that are the recorded contract, not a KAT failure.
_CLEAN_STATUSES = frozenset(
    {"refused", "compiled_not_launched", "compiled_and_launched"}
)


def apply_route_authenticity_gate(verdicts: list) -> None:
    """Flag (never overwrite) inauthentic rows — in place, pure on input.

    A row whose recorded authenticity triple differs from the expected one
    gets ``route_authenticity_failed = True``; its ``status`` is NOT
    touched, so crash/timeout/environment records keep their diagnosis
    (the run log is the record — the primary field must not lie about the
    failure class). Rows without an authenticity record (crashed, timed
    out, env-refused) are left to their own status.
    """
    for rec in verdicts:
        auth = rec.get("route_authenticity")
        if auth is not None and auth != _EXPECTED_ROUTE_AUTHENTICITY:
            rec["route_authenticity_failed"] = True


def exit_code(verdicts: list, failed_checks: list, launch: bool) -> int:
    """The KAT's exit-code contract (pure; unit-tested).

    0 = recorded contract held (offline: typed refusals / not-launched
    compiles; launch: every row launched AND passed the frozen criterion).
    1 = red-first self-checks failed (the instrument itself is broken).
    2 = KAT failure: wrong route, broken environment, crashed/timed-out
    worker, or (launch mode) a row that could not launch.
    3 = launch mode: device numerics violated the frozen criterion — THE
    finding this KAT exists to surface.
    """
    if failed_checks:
        return 1
    broken = [
        rec
        for rec in verdicts
        if rec.get("status") not in _CLEAN_STATUSES
        or rec.get("route_authenticity_failed")
        # A `refused` row is only the recorded contract when it is the
        # TYPED T-SR-29 structured-route refusal. An arbitrary compile
        # death (e.g. a missing `sentient_codegen` after a staled rsync)
        # records status="refused" with tsr29_typed falsy — that is an
        # environment failure, not the contract (r3 MAJOR-1r3).
        or (rec.get("status") == "refused" and not rec.get("tsr29_typed"))
    ]
    if broken:
        return 2
    if launch:
        if any(rec["status"] != "compiled_and_launched" for rec in verdicts):
            # Recorded, but a launch run that could not launch every row
            # is not a PASS.
            return 2
        if any(not rec["score"]["pass"] for rec in verdicts):
            return 3
    return 0


def honest_closed_form_margins():
    """Witness (not prose) the offline margin the docstring leans on.

    Per row x distribution, the max err/bound slack of BOTH honest closed
    forms (with and without the product pre-round) against the frozen
    criterion. The docstring's compounding note leans on this margin
    absorbing the first-order feedback factor; recording it makes that
    claim a number in the run log, not prose.
    """
    margins = {}
    for row_key, m, k, n in SHAPE_ROWS:
        for dist_key, h in DISTRIBUTIONS:
            seed = seed_for(row_key, m, k, n, h)
            a, b = build_inputs(m, k, n, h, seed)
            ref, bound = reference_and_bound(a, b)
            tag = f"{row_key}/{dist_key}"
            margins[tag] = {
                form: score(
                    dl16_sequential_closed_form(a, b, pre_round_products=pre),
                    ref, bound,
                )["max_slack"]
                for form, pre in (("closed_form", False), ("pre_round", True))
            }
    return margins


def run(launch: bool, json_out: str | None) -> int:
    import shutil

    # NOT a TemporaryDirectory: the stderr captures (output code, worker
    # tracebacks) are the diagnosis for any non-clean row and must survive
    # the run — deleted only when everything is clean.
    tmp = tempfile.mkdtemp(prefix="lm_kat_capture_")
    verdicts = []
    for row_key, _, _, _ in SHAPE_ROWS:
        for dist_key, _ in DISTRIBUTIONS:
            rec = _spawn_row(row_key, dist_key, launch=launch, tmp=tmp)
            rec["stderr_capture"] = os.path.join(
                tmp, f"stderr_{row_key}_{dist_key}.txt"
            )
            verdicts.append(rec)

    # Route authenticity gates EVERY row, offline included: a broken stderr
    # capture or an unauthentic artifact is a KAT failure, never a silent
    # all-False record (the pre-registration's ASSERTED-not-assumed law).
    apply_route_authenticity_gate(verdicts)

    checks = red_first_self_checks()
    failed_checks = [name for name, ok in checks if not ok]
    code = exit_code(verdicts, failed_checks, launch)

    if code == 0:
        # Everything clean: the captures are reproducible from the seeds.
        shutil.rmtree(tmp, ignore_errors=True)
        capture_note = "captures removed (clean run)"
    else:
        capture_note = f"captures PRESERVED at {tmp}"

    summary = {
        "launch": launch,
        "kat": "looped_matmul_acc_silicon_kat",
        "pre_registration": "FROZEN (module docstring, committed pre-launch)",
        "verdicts": verdicts,
        "red_first_self_checks": {name: ok for name, ok in checks},
        "red_first_all_passed": not failed_checks,
        "honest_closed_form_margins": honest_closed_form_margins(),
        "capture_note": capture_note,
        "exit_code": code,
    }
    if json_out:
        with open(json_out, "w") as fh:
            json.dump(summary, fh, indent=1)

    print(f"looped_matmul_acc_silicon_kat — mode: {'LAUNCH' if launch else 'OFFLINE'}")
    for rec in verdicts:
        line = (
            f"  {rec['row']}/{rec['dist']} [{rec['M']}x{rec['K']}x{rec['N']}]: "
            f"{rec['status']}"
        )
        if rec.get("route_authenticity_failed"):
            line += " (+ROUTE_AUTHENTICITY_FAILED)"
        if rec["status"] == "refused":
            line += (
                f" ({rec.get('refusal_type')}: "
                f"{rec.get('refusal_codes') or rec.get('refusal_detail', '')[:120]})"
            )
        if "score" in rec:
            line += (
                f" pass={rec['score']['pass']} n_fail={rec['score']['n_fail']}"
                f" max_slack={rec['score']['max_slack']:.3g}"
                f" ledger_n_differ={rec['closed_form_ledger']['n_differ']}"
                f" ledger_max_dev={rec['closed_form_ledger']['max_deviation']:.3g}"
            )
        print(line)
    print(
        "  red-first self-checks: "
        + ("ALL PASS" if not failed_checks else "FAILED: " + ", ".join(failed_checks))
    )
    print(f"  {capture_note}")
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Frontend-routed looped-matmul accumulator silicon KAT "
        "(pre-registered; see module docstring)"
    )
    parser.add_argument(
        "--launch", action="store_true",
        help="launch on device (requires hardware + a coordinated HW claim)",
    )
    parser.add_argument("--json-out", default=None, help="write the verdict JSON here")
    parser.add_argument(
        "--worker", nargs=2, metavar=("ROW", "DIST"),
        help=argparse.SUPPRESS,  # internal: single-row worker mode
    )
    parser.add_argument(
        "--result-out", default=None, help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.worker:
        row_key, dist_key = args.worker
        if not args.result_out:
            parser.error("--worker requires --result-out")
        run_row(row_key, dist_key, launch=args.launch, result_out=args.result_out)
        return 0
    return run(launch=args.launch, json_out=args.json_out)


if __name__ == "__main__":
    sys.exit(main())
