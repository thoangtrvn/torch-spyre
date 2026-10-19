#!/bin/zsh
# Build torch-spyre on macOS without the Spyre runtime (flex/sendnn/senlib):
# codegen-only mode compiles just the self-contained _codegen_backend
# extension (pure torch/ATen headers, no runtime symbols to stub).
# The flag must match setup.py's TORCH_SPYRE_CODEGEN_ONLY (not
# TORCH_SPYRE_COMPILE_ONLY, which setup.py does not read).
TORCH_SPYRE_CODEGEN_ONLY=1 CMAKE_VERBOSE_MAKEFILE=ON pip install -e . --no-build-isolation --no-deps
