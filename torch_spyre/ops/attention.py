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
"""spyre::attention — SDPA on the spyre device through the attention
program (the PyTorch last mile, design 61 §10 item 2).

THE ARCHITECTURE (the embedding/softmax precedent): an opaque custom op
that owns its own buffers and launches precompiled init-packet binaries
via ``torch_spyre._C.launch_kernel_from_bytes``. The generic
Inductor→Triton→TTIR path cannot bind this kernel's multi-segment ABI
(the softmax #86/#113 lesson), so — like softmax — attention is a fusion
barrier at this seam. That is the DECLARED v1 form.

DELEGATION LAW (plans/PLAN_attention_aten_exposure.md): this module owns
ONLY the ATen seam. Admission is codegen's ``attention_aten_plan``
(which delegates to the attention predicate and the multiblock owner —
every refusal arrives with the predicate's own clause name); the launch
sequence is the plan's typed steps; the multi-block fold is
``sentient_codegen.scheduler.attention_merge`` (the one executable
spelling). No refusal reason is invented here beyond the two ATen-flag
refusals (dropout, debug mask) that are not envelope questions.

EXECUTION FORMS (both silicon-witnessed; the plan DECLARES which):
  * CONTAINER (kv_len == block_kv): ONE launch of the fused 13-stage
    container with the 7-segment ABI [kv_src, p_out, q_or_out, tables,
    k_or_scratch, v_out, s_out] — q rides seg2 and the output overwrites
    it after q_scale consumed it. The transposed-B host contract is
    satisfied NATIVELY (the on-device k_transpose stage) — asserted, not
    assumed.
  * HOST_SERIAL (kv_len > block_kv): the multiblock owner's per-phase
    sequence — one hoisted q_scale, then per block: K/V ingress gathers
    (32 rows each, slot tables through codegen's build_kv_slot_table —
    the ONE owner), scores (B staged host-side with the contraction dim
    on the stick axis — the scores_b_layout contract, ASSERTED), the
    softmax bundle, and PV; per-block partials folded host-side in fp64
    through attention_merge.

HOST STAGING (the KATs' verbatim form): the K/V sources and the
matmul B operands are staged on the host and moved per launch. This
costs a D2H+H2D roundtrip per call — the DECLARED v1 cost (the
embedding lane accepted the same for correctness-first exposure; a
device-resident staging path is a follow-up, never a silent guess at an
unwitnessed device-side layout equivalence).

ENVELOPE (v1 — the witnessed set only): decode-path SDPA, rows <= 32,
head_dim a whole-stick multiple <= 512, DL16 (torch.float16), causal at
the decode tail (the host contract: every staged KV position is visible
to every q row — the K/V tensor holds only positions strictly prior to
the query block), any kv_len that is a whole number of 64-position
blocks. Everything else refuses typed with the predicate's clause names.
There is deliberately NO CPU fallback: a silent host attention is worse
than a hard error (the #70 law).

BLOCK TABLES: ``block_table=None`` is the CONTIGUOUS form (K rows then
V rows, one row-contiguous combined source). A non-None int tensor of
shape (num_ingress_steps, 32) is the PAGED form: row i holds the 32
physical slot rows of ingress step i, in the PLAN'S OWN step order (the
same order the steps list carries — no second ordering to get wrong).
The a3/a5p provenance: same binaries, gapped layout, gaps live entirely
in the table. On the CONTAINER path the paged K and V sources are
scattered into ONE combined kv_src here (host-side, the a5p KAT form);
on HOST_SERIAL the two sources stay separate (the a4p KAT form).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from torch_spyre._runtime import require_runtime

__all__ = ["attention", "sdpa_attention_mapping", "AttentionHostError"]

#: Element width per torch dtype. 1p0 is DL16 (1-6-9) — never "FP16";
#: bfloat16 is 1p5/BF16 and refuses [dtype] via the predicate.
_DTYPE_BITS = {torch.float16: 16}
_ELEMENTS_PER_STICK = 64  # 128 B / 2 B at DL16 (1p0 registry row)


class AttentionHostError(RuntimeError):
    """Raised when sentient_codegen is unreachable, so no binary can be built."""


def _codegen_attention_plan(
    rows, head_dim, kv_len, mask_mode, paged, gqa_group, dtype_bits, scale
):
    """Guarded-lazy codegen import (the _embedding_host law): a missing
    PYTHONPATH is a clear message, never an ImportError mid-trace."""
    try:
        from sentient_codegen.encoder.attention_aten_dispatch import (
            attention_aten_plan,
        )
    except ImportError as e:
        raise AttentionHostError(
            "sentient_codegen not importable — source env.sh so codegen is "
            "on PYTHONPATH (required to build the attention binary)."
        ) from e
    return attention_aten_plan(
        rows,
        head_dim,
        kv_len,
        mask_mode=mask_mode,
        paged=paged,
        gqa_group=gqa_group,
        dtype_bits=dtype_bits,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# The ATen mapping — pure, device-free, off-box testable
# ---------------------------------------------------------------------------


def sdpa_attention_mapping(
    query, key, value, attn_bias, dropout_p, is_causal, return_debug_mask, scale
):
    """Map one ATen SDPA call to per-(b, h) attention launch specs.

    Returns a list of dicts (one per (b, h) pair, row-major) carrying
    the typed launch facts plus the codegen plan (admission DELEGATED —
    this function refuses only on ATen flags, never on geometry).

    Refusal vocabulary: every geometric refusal is the attention
    predicate's own clause name, raised by codegen. ``attn_bias`` maps
    to the predicate's [mask-mode] clause (the closed none|causal
    vocabulary); GQA to [gqa-group] (any H != H_kv, integral or not);
    dtype to [dtype].
    """
    if return_debug_mask:
        raise NotImplementedError(
            "spyre attention: return_debug_mask is unsupported — the v1 "
            "route returns no debug artifacts"
        )
    if not (0.0 <= dropout_p < 1.0):
        # A DELIBERATE TIGHTENING owned here, not ATen parity: probed on
        # torch 2.13.0, the ATen composite has NO composite-level range
        # gate — with the math backend forced, dropout_p=-0.5 and 1.0
        # are accepted (only p>1 errors, via the inner dropout op's own
        # check, with a backend-specific message). Those backend-specific
        # checks are bypassed on this route (the compiled path never runs
        # the composite; the Spyre seam has no dropout op to catch it),
        # so the range is enforced at the mapping: outside [0, 1) is a
        # typed error, never a silently-ignored value.
        raise NotImplementedError(
            f"spyre attention: dropout_p must be in [0, 1), got {dropout_p} "
            f"(dropout itself is unsupported on the v1 route)"
        )
    if dropout_p > 0.0:
        raise NotImplementedError(
            "spyre attention: dropout is unsupported on the v1 route"
        )

    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise NotImplementedError(
            f"spyre attention: only 4-D (B, H, S, E) SDPA tensors are "
            f"mapped, got q{tuple(query.shape)} k{tuple(key.shape)}"
        )
    batch, heads, rows, head_dim = (
        int(query.shape[0]),
        int(query.shape[1]),
        int(query.shape[2]),
        int(query.shape[3]),
    )
    kv_heads = int(key.shape[1])
    kv_len = int(key.shape[2])
    if int(value.shape[2]) != kv_len or int(value.shape[3]) != head_dim:
        raise NotImplementedError(
            f"spyre attention: K/V geometry mismatch — "
            f"k{tuple(key.shape)} v{tuple(value.shape)}"
        )

    # GQA/MQA: any head sharing refuses through the predicate's own
    # [gqa-group] clause. A NON-INTEGRAL expansion (H=3, Hkv=2) must not
    # floor to 1 and silently admit — floor(3/2) == 1 would pair q head 2
    # with a nonexistent kv head and fault or read the wrong slice; the
    # ragged case passes 0 so the predicate refuses the same clause.
    if heads == kv_heads:
        gqa_group = 1
    elif heads > kv_heads and heads % kv_heads == 0:
        gqa_group = heads // kv_heads
    else:
        gqa_group = 0

    # dtype: DL16 only. Any other width is refused by the predicate's
    # [dtype] clause (pass 32 — the clause's own "serves 16 only" text).
    dtype_bits = _DTYPE_BITS.get(query.dtype, 32)
    if key.dtype != query.dtype or value.dtype != query.dtype:
        dtype_bits = 32  # mixed dtypes: refuse through the same clause

    # mask: is_causal maps to "causal" UNDER THE DECODE-TAIL HOST
    # CONTRACT (the K/V tensor holds only positions strictly prior to
    # the query block — every staged position visible to every q row);
    # no mask maps to "none" (identical device construction). An
    # explicit bias is neither and refuses the predicate's [mask-mode]
    # clause via its closed vocabulary.
    if attn_bias is not None:
        mask_mode = "explicit_bias"  # never admitted; the owner refuses
    elif is_causal:
        # ATen's is_causal is the bottom-right-aligned tril: at rows > 1
        # an early q row must NOT see the trailing kv positions, and this
        # route's construction is full-visibility (faithful only at
        # rows == 1, pure decode). The decode-tail-with-row-block form
        # (q chunk after the whole KV context) is a CALLER convention the
        # ATen seam cannot verify — it stays reachable through the
        # custom op's declared mask_mode, never through is_causal.
        # (Gauntlet R1 M4: previously admitted then computed wrong.)
        if rows > 1:
            raise NotImplementedError(
                f"attention clause [mask-mode]: is_causal=True at "
                f"rows={rows} > 1 — ATen's causal mask is the "
                f"bottom-right-aligned tril and this route computes "
                f"full visibility (faithful only at rows == 1, decode); "
                f"the row-block decode-tail form is a caller convention "
                f"(spyre::attention's declared mask_mode), not ATen "
                f"semantics. Refusing typed rather than computing the "
                f"wrong mask."
            )
        mask_mode = "causal"
    else:
        mask_mode = "none"

    calls = []
    for _b in range(batch):
        for _h in range(heads):
            plan = _codegen_attention_plan(
                rows, head_dim, kv_len, mask_mode, False, gqa_group, dtype_bits, scale
            )
            calls.append(
                {
                    "rows": rows,
                    "head_dim": head_dim,
                    "kv_len": kv_len,
                    "mask_mode": mask_mode,
                    "gqa_group": gqa_group,
                    "dtype_bits": dtype_bits,
                    "scale": plan.scale,
                    "strategy": plan.strategy.value,
                    "plan": plan,
                }
            )
    return calls


# ---------------------------------------------------------------------------
# The device executor
# ---------------------------------------------------------------------------


def _launch(binary, tensors):
    from torch_spyre._C import launch_kernel_from_bytes

    launch_kernel_from_bytes(binary, tensors)


#: The container ABI's per-symbol declared roles — READ from the codegen
#: container dispatch's own segment classification (``_segment_source_
#: kinds``), never restated here. argN ↔ segment N (the plan's symbol
#: convention). kv_src (arg0) is workspace-class: read at launch, then
#: overwritten mid-chain by the k_transpose stage's K^T output.
_CONTAINER_ROLE_BY_ARG = None  # derived lazily (guarded codegen import)


def _container_roles():
    global _CONTAINER_ROLE_BY_ARG
    if _CONTAINER_ROLE_BY_ARG is None:
        from sentient_codegen.encoder.attention_container_dispatch import (
            _segment_source_kinds,
        )

        _CONTAINER_ROLE_BY_ARG = {
            f"arg{seg}": kind for seg, kind in _segment_source_kinds().items()
        }
    return _CONTAINER_ROLE_BY_ARG


def _launch_container_validated(container_plan, tensors):
    """The CONTAINER launch, gated through the plan's own binding contract
    BEFORE the C++ call (the softmax T-LR-2B pattern): named specs derived
    from the ACTUAL runtime allocations (never the plan's requirement
    echoed back), roles read from the container dispatch's own
    classification. A refusal propagates — no fallback launch."""
    try:
        from sentient_codegen.backend.launch_abi import (
            validate_positional_launch,
        )
    except ImportError as e:
        raise AttentionHostError(
            "sentient_codegen not importable — source env.sh so codegen is "
            "on PYTHONPATH (required to validate the attention launch "
            "bindings)."
        ) from e
    specs = {
        f"arg{i}": (t.numel() * t.element_size(), _container_roles()[f"arg{i}"])
        for i, t in enumerate(tensors)
    }
    validate_positional_launch(container_plan.xlat_plan, container_plan.binary, specs)
    _launch(container_plan.binary, tensors)


def _host_rows(t: torch.Tensor) -> torch.Tensor:
    """The (n, d) CPU fp16 rows of a 2-D input (already CPU — the op's
    batched entry does the ONE whole-tensor D2H before slicing; a D2H of
    a device-slice VIEW is the segfault the old decomposition's
    clone-first preamble existed to prevent)."""
    return t.detach().reshape(t.shape[0], -1).contiguous()


def _to_dev_rows(rows_cpu: torch.Tensor, spt: int, dev) -> torch.Tensor:
    """Row-contiguous (n*spt, 64) device sticks from CPU rows (the KATs'
    staging form: the CPU reshape then H2D gives row-contiguous layout —
    a device-side reshape of a plane-major (n, d) tensor would NOT)."""
    return rows_cpu.reshape(-1, _ELEMENTS_PER_STICK).to(dev)


def _ingress_table(slot_rows, spt: int, cache_rows: int, seg_bits: int):
    """One (1, 32) int32 IBR table through codegen's ONE owner
    (build_kv_slot_table: distinct-slots/slot-extent/ibr-file validated,
    ZERO padding)."""
    from sentient_codegen.encoder.kv_dispatch import build_kv_slot_table

    slot = build_kv_slot_table(
        [int(s) for s in slot_rows],
        sticks_per_row=spt,
        cache_rows=cache_rows,
        segment=0,
        seg_bits=seg_bits,
        route="kv_paged_read",
    )
    return torch.from_numpy(slot.entries)


def _run_container(plan, q2d, k2d, v2d, block_table, dev):
    """The fused-container execution (the a5/a5p KAT form, verbatim)."""
    from sentient_codegen.encoder.attention_container_dispatch import (
        build_gather_tables,
    )

    segmap = plan.container.segmap
    rows, head_dim, kv_len = plan.rows, plan.head_dim, plan.kv_len
    spt = head_dim // _ELEMENTS_PER_STICK

    k_rows = _host_rows(k2d)
    v_rows = _host_rows(v2d)
    q_rows = _host_rows(q2d)

    if block_table is None:
        # contiguous: K rows then V rows in ONE combined source
        kv_rows_cpu = torch.cat([k_rows, v_rows], dim=0)

        def phys(kind, row):
            base = 0 if kind == "k" else kv_len
            return (base + row) * spt
    else:
        # paged: scatter K and V into the gapped physical layout the
        # table addresses (the a5p KAT form — gaps live in the table)
        tbl = block_table.tolist()
        k_gathers = kv_len // 32
        phys_rows = max(max(r) for r in tbl) + 1
        # SENTINEL FILL 7.0 (the a5p convention): an unaddressed row
        # reads back VISIBLE, never silently zero
        kv_rows_cpu = torch.full((phys_rows, head_dim), 7.0, dtype=q2d.dtype)
        for r in range(kv_len):
            g, off = divmod(r, 32)
            kv_rows_cpu[tbl[g][off]] = k_rows[r]
            kv_rows_cpu[tbl[k_gathers + g][off]] = v_rows[r]

        def phys(kind, row):
            kind_idx = 0 if kind == "k" else 1
            g, off = divmod(row, 32)
            return int(tbl[kind_idx * (kv_len // 32) + g][off]) * spt

    kv_src = _to_dev_rows(kv_rows_cpu, spt, dev)
    tables_np = np.array(build_gather_tables(segmap, phys), dtype=np.int32)
    tables = torch.from_numpy(tables_np).to(dev)

    def sticks(n: int) -> tuple:
        return (n, _ELEMENTS_PER_STICK)

    p_out = torch.zeros(sticks(segmap.qp_p_sticks), dtype=q2d.dtype, device=dev)
    q_or_out = _to_dev_rows(q_rows, spt, dev)
    k_scratch = torch.zeros(
        sticks(segmap.k_scratch_sticks), dtype=q2d.dtype, device=dev
    )
    v_seg = torch.zeros(sticks(segmap.v_sticks), dtype=q2d.dtype, device=dev)
    s_seg = torch.zeros(sticks(segmap.s_sticks), dtype=q2d.dtype, device=dev)

    _launch_container_validated(
        plan.container, [kv_src, p_out, q_or_out, tables, k_scratch, v_seg, s_seg]
    )
    return q_or_out.cpu().reshape(rows, head_dim).to(dev)


def _run_host_serial(plan, q2d, k2d, v2d, block_table, dev):
    """The multiblock HOST_SERIAL execution (the a4 KAT form, verbatim):
    per-phase launches, host-side partials, fp64 fold through
    attention_merge (the one owner)."""
    from sentient_codegen.scheduler.attention_merge import (
        block_partial_from_scores,
        merge_attention_state,
        neutral_attention_state,
        finalize_attention_state,
        AttentionMergeState,
    )

    # The scores B operand is staged HOST-SIDE on this path — the
    # scores_b_layout host-contract axis is ASSERTED, never assumed (the
    # a5 silicon finding: the gather's row-major K is the exact
    # transposed-product failure). Typed raise, never a bare assert
    # (which dies under python -O).
    if plan.scores_b_layout != "k_on_stick_axis":
        raise AttentionHostError(
            f"attention host contract [scores-b-layout]: this executor "
            f"stages the scores B operand with the contraction dim on the "
            f"stick axis; the plan declares "
            f"{plan.scores_b_layout!r}"
        )

    rows, head_dim, kv_len = plan.rows, plan.head_dim, plan.kv_len
    block_kv = plan.block_kv
    spt = head_dim // _ELEMENTS_PER_STICK

    from sentient_codegen.gen.registry import get_generation

    seg_bits = get_generation("rcudd1a").hw.segment_size_bits

    k_rows = _host_rows(k2d)
    v_rows = _host_rows(v2d)
    q_rows = _host_rows(q2d)

    # contiguous sources: logical row r at physical row r.
    #
    # PAGED — THE ONE ORDER CONTRACT (gauntlet R1 M2): the block_table's
    # rows are in the PLAN'S OWN INGRESS STEP ORDER (block{b}:k_ingress_0,
    # k_ingress_1, v_ingress_0, v_ingress_1, block{b+1}:...). The staging
    # scatter below conforms to that SAME order, and the executor
    # consumes the rows in that SAME order — one declared order, three
    # conforming sites, pinned by the executor-execution round-trip test
    # (the first cut scattered kind-major while consuming step-order,
    # silently feeding K slots to V gathers).
    if block_table is None:
        k_phys = k_rows
        v_phys = v_rows
        k_cache_rows = kv_len
        v_cache_rows = kv_len
    else:
        tbl = block_table.tolist()
        # split the step-ordered rows back into per-kind lists (for the
        # cache extents and the scatter), WITHOUT re-ordering the table:
        # per block, the first (kv_len//32) rows are K gathers, the next
        # (kv_len//32) are V gathers — the plan's own phase order
        # gathers per tensor PER BLOCK (the IBR-file split of ONE
        # block — never the whole context)
        gathers_per_tensor = block_kv // 32
        gathers_per_block = gathers_per_tensor * 2
        k_step_rows, v_step_rows = [], []
        for b in range(kv_len // block_kv):
            base = b * gathers_per_block
            k_step_rows.extend(tbl[base : base + gathers_per_tensor])
            v_step_rows.extend(
                tbl[base + gathers_per_tensor : base + gathers_per_block]
            )
        if len(tbl) != (kv_len // block_kv) * gathers_per_block:
            raise AttentionHostError(
                f"attention executor wiring drift: block_table has "
                f"{len(tbl)} rows but the plan's HOST_SERIAL form has "
                f"{(kv_len // block_kv) * gathers_per_block} ingress "
                f"steps — the table and the plan have drifted"
            )
        k_cache_rows = max(max(r) for r in k_step_rows) + 1
        v_cache_rows = max(max(r) for r in v_step_rows) + 1
        # SENTINEL FILL 7.0 (a5p convention): an unaddressed row reads
        # back VISIBLE, never silently zero
        k_phys = torch.full((k_cache_rows, head_dim), 7.0, dtype=q2d.dtype)
        v_phys = torch.full((v_cache_rows, head_dim), 7.0, dtype=q2d.dtype)
        for b in range(kv_len // block_kv):
            for t in range(block_kv):
                gather, off = divmod(t, 32)
                k_phys[k_step_rows[b * gathers_per_tensor + gather][off]] = k_rows[
                    b * block_kv + t
                ]
                v_phys[v_step_rows[b * gathers_per_tensor + gather][off]] = v_rows[
                    b * block_kv + t
                ]

    k_src = _to_dev_rows(k_phys, spt, dev)
    v_src = _to_dev_rows(v_phys, spt, dev)

    q_dev = _to_dev_rows(q_rows, spt, dev)
    qp_dev = torch.zeros((rows * spt, _ELEMENTS_PER_STICK), dtype=q2d.dtype, device=dev)

    state = neutral_attention_state(rows, head_dim)
    vf = v_rows.float().numpy().astype("float64")

    table_rows = block_table.tolist() if block_table is not None else None
    table_idx = 0
    k_block = None
    v_block = None
    s_dev = p_dev = None

    for step in plan.steps:
        name = step.name
        if name == "q_scale":
            _launch(step.binary, [q_dev, qp_dev])
            continue
        _block, phase = name.split(":")
        b = int(_block[len("block") :])
        lo = b * block_kv
        if phase.startswith("k_ingress") or phase.startswith("v_ingress"):
            is_k = phase.startswith("k")
            src = k_src if is_k else v_src
            cache_rows = k_cache_rows if is_k else v_cache_rows
            if table_rows is None:
                g = int(phase.rsplit("_", 1)[1])
                slot_rows = [lo + g * 32 + t for t in range(32)]
            else:
                slot_rows = table_rows[table_idx]
                table_idx += 1
            table = _ingress_table(slot_rows, spt, cache_rows, seg_bits).to(dev)
            out = torch.zeros(
                (32 * spt, _ELEMENTS_PER_STICK), dtype=q2d.dtype, device=dev
            )
            _launch(step.binary, [src, table, out])
            block_rows = out.cpu().reshape(32, head_dim)
            if is_k:
                k_block = k_block if k_block is not None else torch.empty(0)
                k_block = (
                    torch.cat([k_block, block_rows]) if k_block.numel() else block_rows
                )
            else:
                v_block = v_block if v_block is not None else torch.empty(0)
                v_block = (
                    torch.cat([v_block, block_rows]) if v_block.numel() else block_rows
                )
            continue
        if phase == "scores":
            # B staged host-side: K^T (contraction dim on the stick axis)
            kt = k_block.t().contiguous()
            s_dev = torch.full((rows, block_kv), -777.0, dtype=q2d.dtype, device=dev)
            _launch(step.binary, [qp_dev, kt.to(dev), s_dev])
            continue
        if phase == "softmax":
            # FRESH zeroed scratch per block — the a4 KAT's witnessed
            # construction (gauntlet R1 M5: reuse was unwitnessed; match
            # the KAT rather than prove write-before-read)
            scratch = torch.zeros(plan.softmax_scratch, dtype=q2d.dtype, device=dev)
            p_dev = torch.zeros((rows, block_kv), dtype=q2d.dtype, device=dev)
            _launch(step.binary, [scratch, s_dev, p_dev])
            continue
        if phase == "pv":
            o_dev = torch.full((rows, head_dim), -777.0, dtype=q2d.dtype, device=dev)
            _launch(step.binary, [p_dev, v_block.contiguous().to(dev), o_dev])
            # ---- partials + fp64 host fold (the merge owner) ----
            got_s = s_dev.cpu().float().numpy().astype("float64")
            got_pv = o_dev.cpu().float().numpy().astype("float64")
            part = block_partial_from_scores(got_s, vf[lo : lo + block_kv])
            part = AttentionMergeState(m=part.m, l=part.l, acc=part.l[:, None] * got_pv)
            state = merge_attention_state(state, part)
            k_block = v_block = None
            continue
        raise AttentionHostError(
            f"attention executor wiring drift: unknown HOST_SERIAL phase "
            f"{phase!r} — the plan's step vocabulary and this executor "
            f"have drifted"
        )

    out64 = finalize_attention_state(state)
    return torch.from_numpy(out64).to(q2d.dtype).to(dev)


# ---------------------------------------------------------------------------
# The custom op
# ---------------------------------------------------------------------------


def _dynamo_offline(fn):
    """Keep the codegen plan build OUT of dynamo graphs: the plan builder
    is an lru_cache-wrapped host computation (binary emission, numpy);
    tracing it into the graph duplicates megabytes of IR per call for
    zero benefit. A graph break here is correct and cheap."""
    try:
        return torch._dynamo.disable(fn)
    except Exception:
        return fn


@_dynamo_offline
def _build_plan_offline(
    rows, head_dim, kv_len, mask_mode, paged, gqa_group, dtype_bits, scale
):
    return _codegen_attention_plan(
        rows, head_dim, kv_len, mask_mode, paged, gqa_group, dtype_bits, scale
    )


def _to_cpu_contiguous(t: torch.Tensor) -> torch.Tensor:
    """ONE whole-tensor D2H (the supported form). A non-contiguous spyre
    tensor is cloned contiguous ON DEVICE first — the old decomposition's
    clone-first preamble, which exists because a strided/offset D2H is
    the _C segfault class (silicon finding 2026-11-23, cell e1)."""
    if t.device.type == "spyre" and not t.is_contiguous():
        t = t.clone(memory_format=torch.contiguous_format)
    return t.detach().cpu()


def _attention_batched(q, k, v, block_table, dev, scale=None):
    """The op body: whole-tensor D2H once, host-side (b, h) slicing, per-
    call 2-D execution, CPU assembly, ONE H2D. NO device-side slicing or
    strided D2H anywhere (the e1 silicon finding)."""
    # mask_mode is "causal" by the SEAM'S OWN REFUSALS: the ATen mapping
    # has already refused every mask the construction cannot honor
    # (explicit bias, is_causal at rows>1) — the op-level default is the
    # decode-tail equivalence the predicate declares.
    if q.dim() == 2:
        plan = _build_plan_offline(
            int(q.shape[0]),
            int(q.shape[1]),
            int(k.shape[0]),
            "causal",
            block_table is not None,
            1,
            _DTYPE_BITS[q.dtype],
            scale,
        )
        return _attention_2d(plan, q, k, v, block_table, dev)

    batch, heads, rows, head_dim = (
        int(q.shape[0]),
        int(q.shape[1]),
        int(q.shape[2]),
        int(q.shape[3]),
    )
    kv_len = int(k.shape[2])
    if block_table is not None and batch * heads != 1:
        raise NotImplementedError(
            f"spyre attention: a paged block_table serves ONE (b, h) "
            f"call (got B={batch} H={heads}) — batched paged tables are a "
            f"caller composition this primitive does not model"
        )
    kv_heads = int(k.shape[1])
    gqa_group = 1
    if heads != kv_heads:
        # the mapping already refused non-uniform head sharing; the
        # primitive re-derives the same fact defensively
        gqa_group = (
            heads // kv_heads if (heads > kv_heads and heads % kv_heads == 0) else 0
        )

    q_cpu = _to_cpu_contiguous(q)
    k_cpu = _to_cpu_contiguous(k)
    v_cpu = _to_cpu_contiguous(v)

    plan = _build_plan_offline(
        rows,
        head_dim,
        kv_len,
        "causal",
        block_table is not None,
        gqa_group,
        _DTYPE_BITS[q.dtype],
        scale,
    )

    out_cpu = torch.empty((batch, heads, rows, head_dim), dtype=q.dtype)
    for b in range(batch):
        for h in range(heads):
            # h // gqa_group: the kv head SHARED by q head h (correct the
            # day GQA is admitted; today gqa_group == 1 reduces it to h
            # and every other gqa_group refuses upstream)
            h_kv = h // gqa_group
            out_cpu[b, h] = _attention_2d(
                plan, q_cpu[b, h], k_cpu[b, h_kv], v_cpu[b, h_kv], block_table, dev
            )
    return out_cpu.to(dev)


def _attention_2d(plan, q2d, k2d, v2d, block_table, dev):
    """One (rows, head_dim) call on CPU 2-D inputs: strategy dispatch to
    the container or HOST_SERIAL executor (both take CPU rows and do
    their own staged H2D/D2H — the KATs' witnessed forms)."""
    if plan.strategy.value == "container":
        return _run_container(plan, q2d, k2d, v2d, block_table, dev)
    return _run_host_serial(plan, q2d, k2d, v2d, block_table, dev)


@torch.library.custom_op("spyre::attention", mutates_args=(), device_types="spyre")
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Attention on the spyre device through the attention program.

    Accepts the ATen 4-D (B, H, S, E) form (whole-tensor D2H once, host
    (b, h) slicing, CPU assembly, one H2D — no device-side slicing) or a
    2-D (rows, head_dim) single-block form. ``block_table`` (int32/int64,
    one 32-slot row per ingress step, the plan's step order) selects the
    PAGED layout and serves exactly one (b, h) call. Returns q-shaped
    fp16 on the device. Admission is codegen's (clause-named refusals).
    """
    require_runtime("spyre attention")

    if q.device.type != "spyre":
        raise ValueError(f"spyre attention: q must be on spyre, got {q.device}")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype not in _DTYPE_BITS:
            raise NotImplementedError(
                f"spyre attention: {name} dtype {t.dtype} unsupported "
                f"(1p0 is DL16, i.e. torch.float16; bfloat16 is 1p5)"
            )
    if q.dim() not in (2, 4) or k.dim() != q.dim() or v.dim() != q.dim():
        raise NotImplementedError(
            f"spyre attention: 2-D (rows, head_dim)/(kv_len, head_dim) or "
            f"4-D (B, H, S, E) only — got q{tuple(q.shape)}"
        )
    if block_table is not None and block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"spyre attention: block_table must be int32/int64, got {block_table.dtype}"
        )
    return _attention_batched(q, k, v, block_table, q.device, scale)


@attention.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return torch.empty_like(q)
