# Installation

## Prerequisites

- Python >= 3.11
- PyTorch ~= 2.9.1
- IBM Spyre Software Stack (required for hardware execution)

> **Note:** Building Torch-Spyre requires a development build of the IBM
> Spyre Software Stack. If you are within IBM, instructions can be found
> in the internal `#aiu-inductor` Slack channel.

## Install from Source

```bash
git clone https://github.com/torch-spyre/torch-spyre.git
cd torch-spyre
pip install -e ".[build]"
```

This installs Torch-Spyre in editable mode along with all build
dependencies. The C++ extension is compiled automatically via CMake and
Ninja.

## Verify the Installation

```python
import torch
import torch_spyre

x = torch.tensor([1, 2], dtype=torch.float16, device="spyre")
print(x.device)  # device(type='spyre', index=0)
```

## Running the Test Suite

```bash
python -m pytest tests/
```

## Next Steps

- [Quickstart](quickstart.md) — run your first model on Spyre
- [Tensors and Layouts](../user_guide/tensors_and_layouts.md) — understand how tensors work on Spyre
