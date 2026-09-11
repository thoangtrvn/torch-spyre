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

torch_spyre._autoload()
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

The mode also skips C++ extension builds in `setup.py`, so editable installs can be
used on compiler-only hosts:

```bash
TORCH_SPYRE_CODEGEN_ONLY=1 python -m pip install -e .
```

The Python `_C` and `_hooks` shim modules exist only so compiler modules that need
schema/layout symbols can import. They are not substitutes for the C++ runtime.

## What is refused

Any action that needs a device allocator, tensor transfer, stream operation, or
kernel launch fails loudly with `SpyreRuntimeUnavailableError`. Examples include:

- `torch.empty(..., device="spyre")`
- CPU↔Spyre tensor copies
- `torch.spyre.Stream`, `torch.spyre.synchronize`, and runtime initialization
- custom-op launch paths such as `spyre::embedding` and `spyre::softmax`

This boundary is intentional. Codegen-only mode is for frontend/compiler evidence;
it does not prove packet launchability, runtime ABI correctness, or hardware
correctness.
