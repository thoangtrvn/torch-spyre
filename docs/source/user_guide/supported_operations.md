# Supported Operations

This page lists the PyTorch operations that Torch-Spyre supports via
`torch.compile`. Operations are grouped by category.

For details on how operations are implemented and how to add new ones,
see [Adding Operations](../compiler/adding_operations.md).

## Operations Table

| Operation | Eager | Compiled | Execution | Notes |
|-----------|:-----:|:--------:|-----------|-------|
| **Matrix Operations** | | | | |
| `torch.mm` | Y | Y | Spyre | |
| `torch.matmul` | | Y | Spyre | |
| `torch.addmm` | Y | Y | Spyre | Decomposed to `mm` + `add` |
| `torch.bmm` | Y | Y | Spyre | |
| `torch.nn.functional.linear` | Y | Y | Spyre | Decomposed to `matmul` + `add` |
| **Convolution** | | | | |
| `torch.nn.functional.conv2d` (1x1) | Y | Y | Spyre | kh=kw=1/s1/p0/d1/groups=1 only: aliased to `matmul` + separate bias `add`. Sub-stick C or non-contiguous NCHW/OIHW refuses loud |
| `torch.nn.functional.conv2d` (general, v0) | Y | Y | Spyre (mm) + host (im2col) | stride=1/dilation=1/groups=1, padding 0 or `kernel//2` per dim. Rewritten to **host im2col (unfold) + `mm`** with K zero-padded to a whole stick; bias is a separate `add`. **Envelope, refuse-loud before emission:** `ks >= 9` (chunked-K not landed — every real 3x3 at C >= 64 refuses here) and `ks*ns > 127` (14-bit L3 LBR fit). **Host-vs-device ledger:** im2col runs on host — the device-side gather is hardware-vetoed (stick-granular L3 EAR strides; dot-operand provenance guard); the K zero-pad (`K_pad - K` structural zeros, exact) is the im2col amplification bill. Strided/dilated/grouped/transposed/conv1d/conv3d/asymmetric-padding variants refuse loud (no CPU fallback) |
| **Activation Functions** | | | | |
| `torch.nn.functional.softmax` | Y | Y | Spyre | **`dim=-1` only, and ONE shape: `[1024, 64]` float16 (DL16).** Routed via the `spyre::softmax` custom op; HW-verified eager + compiled at 3.28 ULP_DL16. Every other shape/dtype/axis raises `NotImplementedError` (no host fallback). Attention shapes (`[seq, seq]` prefill, `[1, seq]` decode) are NOT yet supported. See `codegen/docs/op-milestones/softmax.md`. |
| `torch.nn.functional.layer_norm` | Y | Y | Spyre | Custom decomposition |
| `torch.nn.functional.rms_norm` | Y | Y | Spyre | Custom decomposition |
| `torch.nn.functional.gelu` | | Y | Spyre | Custom op + lowering |
| `torch.nn.functional.silu` | Y | Y | Spyre | |
| `torch.nn.functional.relu` | Y | Y | Spyre | |
| `torch.nn.functional.sigmoid` | Y | Y | Spyre | |
| `torch.nn.functional.softplus` | Y | Y | Spyre | Custom op + lowering |
| `torch.nn.functional.dropout` | Y | Y | Spyre | |
| `torch.nn.functional.scaled_dot_product_attention` | | Y | Spyre | Custom decomposition (math path); compiled only |
| **Pointwise Unary** | | | | |
| `torch.abs` | Y | Y | Spyre | |
| `torch.neg` | Y | Y | Spyre | |
| `torch.exp` | Y | Y | Spyre | |
| `torch.log` | Y | Y | Spyre | |
| `torch.sqrt` | Y | Y | Spyre | |
| `torch.rsqrt` | Y | Y | Spyre | |
| `torch.reciprocal` | Y | Y | Spyre | |
| `torch.tanh` | Y | Y | Spyre | |
| `torch.logical_not` | Y | Y | Spyre | Custom decomposition |
| `torch.bitwise_not` | Y | Y | Spyre | Custom decomposition |
| `torch.clamp` | | Y | Spyre | Custom op + lowering |
| `torch.pow` | Y | Y | Spyre | |
| **Pointwise Binary** | | | | |
| `torch.add` | Y | Y | Spyre | |
| `torch.sub` | Y | Y | Spyre | |
| `torch.mul` | Y | Y | Spyre | |
| `torch.div` | Y | Y | Spyre | |
| `torch.maximum` | Y | Y | Spyre | |
| `torch.bitwise_and` | | Y | Spyre | Custom decomposition |
| `torch.where` | | Y | Spyre | Compiled only |
| **Comparison** | | | | |
| `torch.eq` | Y | Y | Spyre | |
| `torch.ne` | | Y | Spyre | |
| `torch.gt` | | Y | Spyre | |
| `torch.lt` | Y | Y | Spyre | |
| `torch.ge` | Y | Y | Spyre | |
| `torch.le` | | Y | Spyre | |
| **Reduction** | | | | |
| `torch.sum` | Y | Y | Spyre | |
| `torch.mean` | Y | Y | Spyre | |
| `torch.amax` | | Y | Spyre | Compiled only (no eager dispatch) |
| `torch.amin` | | Y | Spyre | Compiled only (no eager dispatch) |
| `torch.max` | | Y | Spyre | Compiled only (no eager dispatch) |
| `torch.linalg.vector_norm` | Y | | Spyre | Eager only; compiled support not validated |
| **View Ops** [^views] | | | | |
| `torch.reshape` / `torch.view` | | Y | Spyre | Includes `_reshape_alias` lowering |
| `torch.transpose` | | Y | Spyre | |
| `torch.t` | Y | Y | Spyre | View op |
| `torch.permute` | Y | Y | Spyre | |
| `torch.clone` | | Y | Spyre | Compiled-tested as `clone().contiguous()`; standalone `clone` is also lowered and used by many decompositions |
| `torch.contiguous` | | Y | Spyre | Compiled only |
| `torch.squeeze` | | Y | Spyre | Partial; some shapes trigger internal recompile |
| `torch.unsqueeze` | | Y | Spyre | Partial; some shapes trigger internal recompile |
| `torch.cat` | Y | Y | Spyre | |
| `torch.stack` | Y | | Spyre | Eager only |
| `torch.repeat` | Y | | Spyre | Eager only |
| `torch.split` | | Y | Spyre | Compiled only (lowers via `aten.slice`) |
| `torch.expand` | | Y | Spyre | Compiled only; supported when followed by a materializing op (e.g. `clone`, `contiguous`). Used internally by `ones`, `pad`, and SDPA decompositions |
| `torch.narrow` / `torch.select` | | Y | Spyre | Compiled only; basic slicing works (see `test_slice` / `test_split`); broader `narrow`/`select` coverage in development |
| **Tensor Creation** | | | | |
| `torch.ones` | Y | Y | Spyre | Custom decomposition |
| `torch.new_ones` | Y | Y | Spyre | Custom decomposition |
| `torch.zeros` | Y | Y | Spyre | Eager via `aten::zero_` (`ops/eager.py`) |
| `torch.full` | Y | Y | Spyre | Custom decomposition |
| `torch.nn.functional.pad` / `torch.constant_pad_nd` | | Y | Spyre | Custom decomposition |
| **Utility** | | | | |
| `torch.item` | Y | Y | Spyre | Copies to CPU, returns Python scalar |
| **CPU Fallback** | | | | |
| `torch.embedding` | Y | Y | CPU fallback | Runs on CPU, result transferred back |
| `torch.arange` | Y | Y | CPU fallback | Runs on CPU, result transferred back |
| `torch.sin` | Y | Y | CPU fallback | Runs on CPU, result transferred back |
| `torch.cos` | Y | Y | CPU fallback | Runs on CPU, result transferred back |
| `torch.tril` | Y | Y | CPU fallback | Runs on CPU, result transferred back |
| `torch.triu` | Y | Y | CPU fallback | Runs on CPU, result transferred back |
| `torch.isin` | Y | Y | CPU fallback | Runs on CPU, result transferred back |
| `torch.bitwise_xor` | Y | Y | CPU fallback | Runs on CPU, result transferred back |
| `torch.bitwise_or` | Y | Y | CPU fallback | Runs on CPU, result transferred back |
| `torch.argmax` | Y | Y | CPU fallback | Runs on CPU, result transferred back |
| `torch.cumsum` | Y | Y | CPU fallback | Runs on CPU, result transferred back |

> **Column key:**
>
> - **Eager** — supported when running operations directly on a Spyre
>   tensor without `torch.compile`. Eager ops are registered via
>   `torch_spyre/ops/eager.py`, `torch_spyre/ops/fallbacks.py` and
>   select decompositions.
> - **Compiled** — supported when using `torch.compile(model, backend="spyre")`.
> - **Execution** — whether the op runs natively on the Spyre accelerator
>   or falls back to CPU. CPU fallback ops are automatically handled by
>   the compiler — a warning is emitted when fallback occurs.
>
> View ops have **partial support**: some shapes and dimension
> combinations may trigger internal recompilation or are not yet
> implemented (e.g., `expand`, `narrow`). This is an active area of
> development.
>
> This table reflects the operations validated in the torch-spyre test
> suite (`tests/inductor/test_inductor_ops.py`). Coverage
> grows continuously — check the
> [test suite](https://github.com/torch-spyre/torch-spyre/tree/main/tests)
> for the latest state.

[^views]: View ops are implemented without cloning whenever the compiler
    can express the new layout as a different read pattern over the same
    storage. The translation happens during layout propagation in the
    pre-scheduling pipeline; the "Views and Index Translation" section
    of the [Inductor Front-End](../compiler/inductor_frontend.md) walks
    through how this works.

## Unsupported Operations

Operations not listed above will either:
- **Fall back to CPU** — if Inductor cannot lower the op to a Spyre
  kernel, it falls back to CPU execution. A warning is emitted.
- **Raise a compile-time error** — if the op produces a tensor layout
  that is incompatible with downstream Spyre ops.

To request support for a new operation or to contribute one yourself,
see [Adding Operations](../compiler/adding_operations.md).
