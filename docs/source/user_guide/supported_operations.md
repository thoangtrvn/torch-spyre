# Supported Operations

This page lists the PyTorch operations that Torch-Spyre supports via
`torch.compile`. Operations are grouped by the model types where they
are exercised and tested.

For details on how operations are implemented and how to add new ones,
see [Adding Operations](../compiler/adding_operations.md).

## Coverage by Model Type

| Operation | GPT-2 | Llama | Hybrid | ResNet-50 | Notes |
|-----------|:-----:|:-----:|:------:|:---------:|-------|
| `torch.mm` | ✓ | ✓ | ✓ | ✓ | |
| `torch.matmul` | ✓ | ✓ | ✓ | ✓ | |
| `torch.addmm` | ✓ | ✓ | ✓ | | |
| `torch.bmm` | ✓ | ✓ | ✓ | | |
| `torch.nn.functional.linear` | ✓ | ✓ | ✓ | | Decomposed to `addmm` |
| `torch.nn.functional.softmax` | ✓ | ✓ | ✓ | | |
| `torch.nn.functional.layer_norm` | ✓ | ✓ | ✓ | | |
| `torch.nn.functional.gelu` | ✓ | ✓ | ✓ | | |
| `torch.nn.functional.silu` | | ✓ | ✓ | | |
| `torch.nn.functional.relu` | | | | ✓ | |
| `torch.nn.functional.batch_norm` | | | | ✓ | |
| `torch.nn.functional.conv2d` | | | | ✓ | |
| `torch.nn.functional.max_pool2d` | | | | ✓ | |
| `torch.nn.functional.avg_pool2d` | | | | ✓ | |
| `torch.cat` | ✓ | ✓ | ✓ | ✓ | |
| `torch.add` | ✓ | ✓ | ✓ | ✓ | |
| `torch.mul` | ✓ | ✓ | ✓ | ✓ | |
| `torch.reshape` / `torch.view` | ✓ | ✓ | ✓ | ✓ | |
| `torch.transpose` | ✓ | ✓ | ✓ | | |
| `torch.permute` | ✓ | ✓ | ✓ | | |
| `torch.clone` | ✓ | ✓ | ✓ | ✓ | |
| `torch.embedding` | ✓ | ✓ | ✓ | | |

> **Note:** This table reflects the operations validated in the
> torch-spyre test suite at the time of writing. Coverage grows
> continuously — check the
> [test suite](https://github.com/torch-spyre/torch-spyre/tree/main/tests)
> for the latest state.

## Unsupported Operations

Operations not listed above will either:
- **Fall back to CPU** — if Inductor cannot lower the op to a Spyre
  kernel, it falls back to CPU execution. A warning is emitted.
- **Raise a compile-time error** — if the op produces a tensor layout
  that is incompatible with downstream Spyre ops.

To request support for a new operation or to contribute one yourself,
see [Adding Operations](../compiler/adding_operations.md).
