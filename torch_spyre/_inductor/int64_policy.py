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

"""AIU 1.0 graph dtype policy: int64 metadata is declared as int32 on device.

WHY THIS EXISTS
---------------
On Sentient 1.0 there is no i64 *data* dtype: ``csrc/types_mapping.h`` maps the
H2D boundary ``int64`` -> ``IEEE_INT64`` CPU-side but ``IEEE_INT32`` device-side,
so an ``int64`` tensor moved to Spyre already lives as 4-byte int32 in device
memory (``generate_dci`` upcasts int32->int64 again on D2H). The PyTorch tensor
*metadata* still says ``torch.int64``.

That metadata/​storage disagreement has two visible consequences:

1. ``sentient_codegen`` reads the graph's declared dtype, emits a
   ``!tt.ptr<i64>`` kernel argument, and refuses loudly (``I64PointerDataError``
   — "i64 is metadata-only on 1p0"). Every compiled graph that carries an int64
   tensor as a kernel argument dies here.
2. The device buffer is sized/interpreted through 8-byte metadata while only 4
   bytes-per-element exist on device — the class behind the ``setStorage``
   undersized-allocation failure.

The target's actual law is: *int64 is metadata-only*. Every int64 tensor a
decoder graph carries is positional/index metadata — ``cumsum`` of an attention
mask turned into position_ids (``modeling_opt.py:353-355``), position_ids /
cache_position buffers, and int64 comparison masks. None is genuine 64-bit data.

This module makes the graph's *declarations* agree with the device: device
integer-metadata tensors are declared ``int32``. It rewrites at the DECLARATION
level — placeholder dtypes, ``dtype=`` of the ops that produce int64, and the
int64-widening casts — and deliberately does NOT insert a device ``int64->int32``
cast kernel: such a cast would itself declare an ``int64`` input pointer and be
refused, and would read the 4-byte storage as 8-byte elements. Reinterpretation,
not conversion, is correct because the device already stores int32.

SAFETY (no silent truncation)
-----------------------------
int32 covers every observed carrier by construction: position_ids, cache
positions, cumulative attention masks, and the models' max-position constants
(e.g. 258880) are bounded by sequence/token counts, which are many orders of
magnitude below 2**31. No device int64 tensor can hold genuine data (there is no
i64 data dtype), so demotion cannot truncate a value the hardware could hold.

The pass is fail-loud, not best-effort: a graph *output* that would be int64 on
device is REFUSED (``SpyreInt64PolicyError``) rather than silently returned as
int32 — a user-visible ABI change must be a deliberate decision, not a rewrite.

SEAM NOTE
---------
This runs inside the ``compile_fx`` wrapper, where the Dynamo-captured graph's
nodes carry ``meta['example_value']`` (a FakeTensor with the Spyre device) and
targets are raw builtins/methods (``torch.cumsum``, ``operator.mul``, the
``long`` method) — NOT ``aten.*`` overloads and NOT ``meta['val']``. The helpers
below read whichever meta key is populated and match both spellings.

ABI / VERIFICATION
------------------
AOTAutograd re-derives the functional graph from ``example_inputs``, so a pre-AOT
meta rewrite alone is discarded; the pass therefore also reinterprets each int64
device example input as an int32 tensor of the same shape/stride (a metadata
reinterpretation, not a ``.to`` copy, which would read int64 and be refused). The
rewritten graph declares int32 inputs; the runtime call still passes the caller's
int64-typed tensor, whose device storage is already int32, so its data pointer
reads correctly as int32. Whether the launch wrapper reinterprets the pointer
(offline signatures say yes: the emitted kernel is ``*i32``, verified codegen-
only) or re-asserts the tensor's int64 dtype at call time is a hardware-verified
question, disclosed in the lane stop report.

OFFLINE EVIDENCE: with this policy the compiled Triton signature for the OPT
position-id kernel is ``in_ptr0:*i32, in_ptr1:*i32, in_ptr2:*fp16`` — no
``*i64`` — versus ``*i64`` without it. Float-only graphs are untouched.
"""

from __future__ import annotations

import torch
import torch.fx

from torch_spyre.constants import DEVICE_NAME

__all__ = [
    "SpyreInt64PolicyError",
    "apply_int64_metadata_policy",
    "is_int64_metadata_op",
]


class SpyreInt64PolicyError(RuntimeError):
    """A graph's int64 use cannot be soundly demoted to device int32 metadata."""


# Canonical op identity -> the builtins/methods that spell it. Dynamo graphs
# carry raw builtins (``<built-in method cumsum ...>`` / ``operator.mul``), not
# ``aten.*`` overloads.
_CUMSUM = {torch.cumsum, torch.ops.aten.cumsum.default}
_SUM = {torch.sum, torch.ops.aten.sum.default}
_PROD = {torch.prod, torch.ops.aten.prod.default}
_ARANGE = {
    torch.arange,
    torch.ops.aten.arange.start,
    torch.ops.aten.arange.start_step,
    torch.ops.aten.arange.default,
}
_TO_CAST = {torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype}

# Integer-metadata producers that promote/default to int64 regardless of input
# dtype; demote these by forcing ``dtype=torch.int32``.
_INT64_PROMOTING = _CUMSUM | _SUM | _PROD | _ARANGE | _TO_CAST

# method-name widening casts (``.long()`` / ``.to(torch.int64)``) that
# reintroduce int64 downstream of an already-int32 metadata source.
_WIDENING_METHODS = {"long"}


def is_int64_metadata_op(target) -> bool:
    """True when ``target`` is a known integer-metadata int64 producer."""
    if target in _INT64_PROMOTING:
        return True
    packet = getattr(target, "overloadpacket", None)
    return packet in {torch.ops.aten.cumsum, torch.ops.aten.sum, torch.ops.aten.prod}


def _val_of(node):
    """The node's example FakeTensor (``example_value`` first, then ``val``)."""
    v = node.meta.get("example_value", None)
    if not isinstance(v, torch.Tensor):
        v = node.meta.get("val", None)
    return v if isinstance(v, torch.Tensor) else None


def _dtype_of(node):
    v = _val_of(node)
    return None if v is None else v.dtype


def _devtype_of(node):
    v = _val_of(node)
    if v is None:
        return None
    return getattr(v.device, "type", None)


def _is_int64_on_device(node, device_types) -> bool:
    if _dtype_of(node) != torch.int64:
        return False
    dev = _devtype_of(node)
    # Unknown device is treated as eligible: this pass only runs on graphs
    # already established as Spyre graphs.
    return dev in device_types or dev is None


def _set_dtype(node, dtype):
    """Re-declare ``node``'s output dtype across every populated meta key."""
    for key in ("example_value", "val"):
        v = node.meta.get(key, None)
        if isinstance(v, torch.Tensor) and v.dtype != dtype:
            node.meta[key] = v.to(dtype)


def apply_int64_metadata_policy(
    gm,
    example_inputs=None,
    *,
    device_types=frozenset({DEVICE_NAME}),
):
    """Rewrite int64 device-metadata declarations in ``gm`` to int32, in place.

    Args:
        gm: the FX GraphModule about to be handed to Inductor.
        example_inputs: the graph's example inputs. Int64 device inputs are
            re-declared int32 so Inductor's input buffers match the graph. A new
            list is returned when any input is rewritten; else the original.
        device_types: device types whose int64 tensors are subject to the policy.
            Defaults to Spyre; injectable so the transform is unit-testable
            without a Spyre device.

    Returns:
        The (possibly rewritten) ``example_inputs``.

    Raises:
        SpyreInt64PolicyError: the graph would RETURN an int64 device tensor.
    """
    graph = gm.graph
    changed = False

    # 0. Refuse first, on the ORIGINAL declarations, if the graph would RETURN an
    #    int64 device tensor. Once producers are demoted their output reads int32;
    #    demoting a *user-visible* output would silently change the declared dtype.
    out_node = graph.output_node()
    for out in _iter_nodes(out_node.args[0] if out_node.args else ()):
        if _is_int64_on_device(out, device_types):
            raise SpyreInt64PolicyError(
                "compiled Spyre graph would return an int64 tensor on device. "
                "i64 is metadata-only on 1p0 (no i64 data dtype exists); cast "
                "the value to int32 before the graph boundary, or move it to the "
                "host. Refusing rather than silently returning int32 where the "
                "caller declared int64."
            )

    # 1. Demote int64-device placeholders: re-declare int32. No cast node — the
    #    device storage is already int32, so this is a reinterpretation.
    for node in graph.nodes:
        if node.op == "placeholder" and _is_int64_on_device(node, device_types):
            _set_dtype(node, torch.int32)
            changed = True

    # 1b. Reinterpret int64 device example inputs as int32-typed tensors of the
    #     SAME shape/stride. AOTAutograd re-derives the functional graph from
    #     ``example_inputs``, so the pre-AOT meta rewrite alone is discarded. This
    #     is a metadata reinterpretation (device storage is already int32), NOT a
    #     ``.to(int32)`` copy — a dtype-converting copy would itself read i64 and
    #     be refused. The runtime call still passes the caller's tensor; its data
    #     pointer is reinterpreted as the int32 the device actually stores.
    if any(
        isinstance(ex, torch.Tensor)
        and ex.dtype == torch.int64
        and getattr(ex.device, "type", None) in device_types
        for ex in (example_inputs or ())
    ):
        new_inputs = list(example_inputs)
        for i, ex in enumerate(new_inputs):
            if not (
                isinstance(ex, torch.Tensor)
                and ex.dtype == torch.int64
                and getattr(ex.device, "type", None) in device_types
            ):
                continue
            if getattr(ex.device, "type", None) == DEVICE_NAME:
                # Real Spyre tensor: its storage is already int32 (H2D downcast),
                # so reinterpreting shape/stride at int32 is faithful. A ``.to``
                # copy is refused (it would read int64) and is unnecessary.
                new_inputs[i] = torch.empty_strided(
                    ex.shape, ex.stride(), dtype=torch.int32, device=ex.device
                )
            else:
                # Host-side example (unit tests / cpu compile): value-preserving.
                new_inputs[i] = ex.to(torch.int32)
        example_inputs = new_inputs
        changed = True

    # 2. Demote int64 producers. Values inherited from these (mul/sub/add/
    #    comparison/getitem/reshape, ...) follow their input dtype and need no
    #    entry — demoting the source keeps the chain int32.
    for node in list(graph.nodes):
        if _dtype_of(node) != torch.int64:
            continue

        if node.op == "call_function" and is_int64_metadata_op(node.target):
            node.kwargs = {**node.kwargs, "dtype": torch.int32}
            _set_dtype(node, torch.int32)
            changed = True
            continue

        # int64-widening methods: .long() on already-integer metadata is a
        # no-op reinterpretation; .int() keeps the value int32.
        if node.op == "call_method" and node.target in _WIDENING_METHODS:
            node.target = "int"
            _set_dtype(node, torch.int32)
            changed = True
            continue

    # 3. Re-propagate dtypes. Demoting a producer leaves every node *downstream*
    #    of it carrying a stale int64 meta (mul/sub/add/reshape cache the dtype
    #    they were traced with). Inductor reads that meta to size/type the
    #    intermediate buffers, so a stale int64 meta re-materialises an i64
    #    buffer even though the producing ops are now int32. Re-running FakeTensor
    #    propagation through the rewritten graph refreshes every meta to its true
    #    int32 dtype. This is a metadata pass over FakeTensors — no device work.
    if changed:
        _repropagate_dtypes(gm, example_inputs)
        graph.lint()
        gm.recompile()
    return example_inputs


def _repropagate_dtypes(gm, example_inputs) -> None:
    """Refresh ``node.meta`` dtypes by re-running FakeTensor propagation."""
    if not example_inputs:
        return
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    with FakeTensorMode() as mode:
        fake_inputs = []
        for ex in example_inputs:
            if not isinstance(ex, torch.Tensor):
                fake_inputs.append(ex)
                continue
            # A device int64 input is reinterpreted as the int32 the device
            # actually stores; build the fake with that dtype so propagation is
            # consistent with the placeholder meta rewritten in step 1.
            dt = torch.int32 if ex.dtype == torch.int64 else ex.dtype
            fake_inputs.append(
                torch.empty_strided(ex.shape, ex.stride(), dtype=dt, device=ex.device)
            )
        FakeTensorProp(gm, mode=mode).propagate(*fake_inputs)

    # Prefer the fresh ``val`` over any stale ``example_value`` and keep both in
    # sync so downstream readers (Inductor input buffers) see the same dtype.
    for node in gm.graph.nodes:
        val = node.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            node.meta["example_value"] = val


def _iter_nodes(value):
    """Yield FX Nodes nested (arbitrarily) inside containers."""
    if isinstance(value, torch.fx.Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_nodes(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_nodes(item)
