# Codegen-only mode

`TORCH_SPYRE_CODEGEN_ONLY=1` makes the Python compiler stack importable on a host
that does not have the Spyre runtime extension or Sentient runtime libraries. This
is an inspection mode for Inductor/Triton/codegen work: use it to import
`torch_spyre`, register the `spyre` backend name, load TorchInductor patches, and
capture compiler artifacts such as TTIR.

It is deliberately not a runtime simulator.

## What is enabled

With the environment variable set before import:

```bash
TORCH_SPYRE_CODEGEN_ONLY=1 python - <<'PY'
import torch
import torch_spyre

print(torch._C._get_privateuse1_backend_name())  # spyre
print(torch.spyre.codegen_only_available())      # True
print(torch.spyre.is_available())                # False
print(torch.spyre.device_count())                # 0

import torch_spyre._inductor.customops
import torch_spyre._inductor.decompositions
import torch_spyre._inductor.lowering
import torch_spyre._inductor.ir
PY
```

The mode builds only a minimal C++ codegen backend extension in `setup.py`, so
editable installs can be used on compiler-only hosts without linking the
Sentient/Flex runtime libraries:

```bash
TORCH_SPYRE_CODEGEN_ONLY=1 python -m pip install -e .
```

A codegen-only install persists this mode in the Python environment's
`torch.backends` entry point. After installing with `TORCH_SPYRE_CODEGEN_ONLY=1`,
that environment can import `torch`/`torch_spyre` without setting the environment
variable on every command. Reinstall without `TORCH_SPYRE_CODEGEN_ONLY=1` when you
want a full runtime build.

The Python `_C` shim aliases `torch_spyre._codegen_backend` when that extension is
available. That extension registers a synthetic `spyre:0` PrivateUse1 backend and
can materialize layout-bearing fake/codegen tensors for compiler capture:

```python
x = torch.empty((512, 128), dtype=torch.float16)
y = x.to("spyre")
assert y.device.type == "spyre"
assert y.device_tensor_layout().device_size == [512, 2, 64]
```

See [Codegen-only fake backend design](codegen_fake_backend.md) for the boundary
and alternatives considered.

## Diagnostic TensorDescriptor pointwise capture

Torch-Spyre autoload makes the Spyre Inductor path descriptor-friendly by
default:

```python
torch._inductor.config.triton.use_tensor_descriptor is True
torch._inductor.config.assume_aligned_inputs is True
```

Those defaults only let Inductor consider TensorDescriptor lowering. They do not
make descriptor pointwise a production route.

For the narrow frontend-capture diagnostic, set:

```bash
TORCH_SPYRE_EXPERIMENTAL_DESCRIPTOR_POINTWISE=1
```

With `TORCH_SPYRE_CODEGEN_ONLY=1`, this opt-in route lets the Spyre Inductor
context admit TensorDescriptor lowering for the validated `512 x 128` pointwise
identity canary. Inside the Spyre Inductor context, `spyre` inputs are treated as
aligned when `assume_aligned_inputs=True`; PyTorch's upstream helper otherwise
only applies that assumption to GPU device types and records PrivateUse1 inputs in
`V.graph.unaligned_buffers`, which blocks TensorDescriptor loads. The same
context disables Triton's implicit fp16/bf16 compute upcast (`triton.codegen_upcast_to_fp32=False`)
so AIU 1.0 DL16 pointwise kernels do not silently widen loads to fp32 and truncate
on store unless the user requested an explicit dtype conversion.

The expected TTIR evidence is:

```text
tt.make_tensor_descriptor
tt.descriptor_load
tt.descriptor_store
```

and raw-pointer fallback evidence remains rejected for this milestone:

```text
tl.load
tl.store
```

The route is intentionally fail-closed:

- the environment flag is required;
- the diagnostic shape is restricted to `512 x 128`;
- `no_x_dim` stores are rejected;
- unaligned buffer names are rejected using Inductor's base-pointer alignment
  guard after the Spyre context has applied its PrivateUse1 alignment policy;
- unresolved constant offsets are rejected instead of assumed aligned.

The Sentient stick ABI is stronger than Triton's `{tt.divisibility = 16}` TTIR
metadata. That attribute is only a compiler-visible lower-bound hint; it is not
proof that a runtime tensor argument is 128-byte stick-aligned. The descriptor
ABI diagnostic in `torch_spyre._inductor.descriptor_abi` checks the effective
tensor pointer (`tensor.data_ptr()`, not merely the allocation base) and the
storage-offset composition against the 128-byte Sentient stick boundary. For
AIU 1.0 DL16/f16 tensors, a view preserves stick alignment only when its
2-byte-element storage offset remains a multiple of 64 elements.

Today's real Spyre allocator exposes a host-side bookkeeping token through
`Tensor.data_ptr()`, while the real on-device address lives below Python in the
Flex composite address. Therefore the opt-in descriptor pointwise path logs the
alignment report as shadow-only evidence and does not gate compilation on that
host token. The storage-offset part of the report is still load-bearing: a view
whose offset is not a whole 128-byte stick must not be treated as a legal
stick-native descriptor argument.

The current boundary is frontend evidence only. Downstream Sentient codegen still
must add descriptor analysis, pointwise scheduler inputs, QGInit/packet parity,
launch ABI proof, and HW-KAT evidence before TensorDescriptor pointwise can be
claimed as a supported compile route.

## What is refused

Any action that needs a device allocator, tensor transfer, stream operation, or
kernel launch fails loudly with `SpyreRuntimeUnavailableError`. Examples include:

- real Spyre device allocation despite `torch.empty(..., device="spyre")` returning a fake/codegen tensor
- real CPU↔Spyre tensor copies despite `x.to("spyre")` returning a layout-bearing fake/codegen tensor
- `torch.spyre.Stream`, `torch.spyre.synchronize`, and runtime initialization
- custom-op launch paths such as `spyre::embedding` and `spyre::softmax`

This boundary is intentional. Codegen-only mode is for frontend/compiler evidence;
it does not prove packet launchability, runtime ABI correctness, or hardware
correctness.
