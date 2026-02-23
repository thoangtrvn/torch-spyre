# Profiling Overview

Torch-Spyre provides tooling to measure and diagnose the performance of
PyTorch workloads running on the Spyre device.

## What Can Be Profiled

- **Compilation time** — time spent in the front-end and back-end compilers
- **Kernel execution time** — wall-clock time per kernel on device
- **Memory usage** — peak DDR allocation during a forward pass
- **Work division efficiency** — core utilization for each operation

## Profiling Workflow

> **TODO:** Document the end-to-end profiling workflow once tooling is
> stabilized. The following is a placeholder outline.

1. Instrument your script with the Torch-Spyre profiler context manager
2. Run the model
3. Collect and view the profiling report

## Integration with PyTorch Profiler

Torch-Spyre emits events compatible with `torch.profiler.profile`, so
standard PyTorch profiling tools (TensorBoard, Chrome trace viewer) can
be used to visualize results.

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU]) as prof:
    output = compiled_model(x)

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

## See Also

- [Running Models](running_models.md)
- [Compiler Architecture](../compiler/architecture.md)
