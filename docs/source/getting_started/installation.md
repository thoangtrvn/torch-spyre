# Installation

## Status

Torch-Spyre is an in-development project. Installation and build
instructions will be provided once the runtime and supporting software
have progressed to a publicly available release.

Watch this repository for updates or open an issue to express interest
in early access.

## Requirements

| Component | Version |
|---|---|
| Python | ≥ 3.11 |
| PyTorch | ~= 2.11.0 |

The exact pins live in [`pyproject.toml`](https://github.com/torch-spyre/torch-spyre/blob/main/pyproject.toml) and are resolved through [`requirements/run.txt`](https://github.com/torch-spyre/torch-spyre/blob/main/requirements/run.txt).

## Verify the Installation

Once installed, verify your setup with:

```python
import torch

x = torch.tensor([1, 2], dtype=torch.float16, device="spyre")
print(x.device)  # device(type='spyre', index=0)
```

## Codegen-only Installation

For compiler and TTIR-capture work on a machine without the Sentient runtime
libraries, install without building the runtime extensions:

```bash
TORCH_SPYRE_CODEGEN_ONLY=1 python -m pip install -e .
```

This mode is intentionally narrow: `import torch_spyre`, backend registration, and
`torch_spyre._inductor` imports work, but device allocation and kernel launch APIs
fail loudly. See [Codegen-only mode](../compiler/codegen_only.md) for the exact
boundary.

## Running the Test Suite

```bash
python -m pytest tests/
```

## Next Steps

- [Quickstart](quickstart.md) — run your first model on Spyre
- [Tensors and Layouts](../user_guide/tensors_and_layouts.md) — understand how tensors work on Spyre
