# RFC: Program Execution Pipeline

## Status

Draft

## Summary

This RFC introduces a new `Program` data structure and execution pipeline that provides a lightweight mechanism for torch-spyre to launch kernels on the flex runtime. It replaces the current graph-execution-based approach with a simpler model where programs are constructed from compiler output, loaded onto the device, and launched directly against device tensors.

## Motivation

The current connection between torch-spyre and the flex runtime relies on graph execution, which carries significant overhead for what is fundamentally a kernel launch operation. The graph-based approach requires constructing, optimizing, and traversing a graph structure even for single-kernel execution, adding complexity and latency that is unnecessary for the common case.

This RFC proposes a new `Program` data structure that simplifies this path. A Program is a lightweight container of Instructions (each instruction mapping directly to a Control Block) that can be constructed once from compiler output, cached, and launched directly against device tensors without the intermediary graph machinery. This gives torch-spyre a direct, minimal-overhead mechanism for kernel execution on the flex runtime.

As a secondary benefit, this simpler abstraction makes it straightforward to extend the launch model — for example, `LaunchTiled` enables reuse of a kernel compiled for a smaller tile size (e.g., 1024) against larger tensors (e.g., 4096) by looping at the runtime level, without recompilation.

The introduction of the SpyreAllocator's VF mode also shapes this design: tensors are now carved from pre-allocated segments with block-level offsets rather than independently allocated, and the Program execution pipeline accounts for this addressing model natively.

## Design

### Core Components

#### SpyreAllocator

Manages device memory in one of two modes:

- **PF Mode**: Each tensor is independently allocated on hardware. `vf_offset` is 0 and unused.
- **VF Mode**: Pre-allocates a fixed pool of large segments (default: 8 segments x 12 GB = 96 GB). Individual tensor allocations carve blocks from these segments with 128-byte alignment. Each allocation produces a `vf_offset` representing the block's position within its segment.

Key properties of an allocation:
- **segment_id**: Which segment the tensor lives in
- **vf_offset**: Byte offset of the tensor's block within that segment
- **size**: Size of the allocation in bytes

#### DMAEngine

Responsible for all host-to-device and device-to-host data movement.

Methods:

| Method | Description |
|--------|-------------|
| `Copy(Program, location_in_segment)` | Copy a program binary to a location in a segment on device |
| `Copy(CPUTensor, location_in_segment)` | Copy a host tensor to a location in a segment on device, producing an SpyreTensor |
| `Copy(SpyreTensor, location_on_host)` | Copy a device tensor back to host memory |

The `location_in_segment` is produced by the SpyreAllocator and encodes the segment ID and offset within that segment.

#### SpyreTensor

A tensor residing on-device. Carries metadata required for execution:

- **shape**: Logical dimensions (e.g., `[4096, 1024]`)
- **stride**: Memory stride per dimension (bytes between consecutive elements along each axis)
- **data_type**: Element type (e.g., float32, uint32)
- **segment_id**: Which device memory segment this tensor occupies
- **vf_offset**: Byte offset within the segment (from SpyreAllocator)
- **size_bytes**: Total byte size of the tensor data

#### Instruction

Maps directly to a Control Block (CB) for Spyre execution. An instruction encapsulates a single atomic unit of work on the Spyre.

Constructors:

| Constructor | Description |
|-------------|-------------|
| `Instruction(SpyreCompute)` | Single compute operation |
| `Instruction(SpyreCompute, List<PreProcessCompute>)` | Compute with required preprocessing (e.g., program correction before execution) |

An instruction should be self-contained: if a compute requires program correction, both the correction and the compute must be part of the same instruction.

#### ExecutionPlan

Produced by the backend compiler (deeptools). Describes the ordered sequence of computes and any preprocessing required per compute. Contains:

- Ordered list of SpyreCompute and PreProcessCompute operations
- Program correction metadata
- Init packets
- Segment table describing expected input/output sizes

This is the bridge between compiler output and runtime execution. (Detailed structure TBD by backend compiler.)

#### Program

Responsible for setting up low-level compute structures (control blocks) from compiler output. A Program is an ordered collection of Instructions.

Constructors:

| Constructor | Description |
|-------------|-------------|
| `Program()` | Empty program, instructions added manually |
| `Program(ExecutionPlan)` | Auto-populated from compiler output |

Methods:

| Method | Description |
|--------|-------------|
| `AddInstruction(Instruction)` | Append an instruction to the program |
| `RemoveInstruction(Instruction)` | Remove an instruction from the program |
| `Launch(List<SpyreTensor>)` | Strict launch — tensor shapes must exactly match the compiled kernel |
| `LaunchTiled(List<SpyreTensor>)` | Tiled launch — automatically loops over tensors larger than the compiled tile size |

### Launch vs LaunchTiled

#### Launch(List\<SpyreTensor\>)

Strict execution mode. Validates that each tensor's shape and stride exactly match what the compiled kernel expects (as defined in the segment table). Fails immediately on mismatch.

**Preconditions:**
- All tensor shapes match the kernel's expected shapes exactly
- All tensor strides are compatible with the kernel's memory layout
- Tensors are allocated on-device and accessible

**Behavior:**
1. Validate tensor metadata against the program's segment table
2. Specialize control blocks with tensor device addresses (segment_id + vf_offset)
3. Submit control blocks to the scheduler for execution
4. Wait for completion

#### LaunchTiled(List\<SpyreTensor\>)

Launch a program repeatedly over tensors larger than the compiled tile size, automatically inferring iteration count and per-iteration offsets from the tensor metadata and the compiled kernel's segment table.

**Preconditions:**
- Each tensor dimension is greater than or equal to the corresponding compiled tile dimension
- Each tensor dimension is evenly divisible by the corresponding tile dimension
- Tensor strides are consistent with the tiling dimension

**Behavior:**
1. Compare each tensor's shape against the kernel's expected tile shape (from the segment table)
2. Infer the tiling dimension(s) and compute `num_iterations = tensor_dim / tile_dim`
3. For each iteration `i`:
   a. Compute per-tensor offset: `offset_i = base_vf_offset + (i * tensor_stride_along_tiled_dim * tile_size * element_size)`
   b. Re-specialize control blocks with updated device addresses
   c. Submit to scheduler and wait for completion before next iteration (same device memory is reused)

**Example — Matmul tiling along M:**

Kernel compiled for `A[1024, K] * B[K, N] = C[1024, N]`. Actual tensor `A` is `[4096, K]`, `C` is `[4096, N]`:

```
num_iterations = 4096 / 1024 = 4

Iteration 0: A[0:1024, :]    * B → C[0:1024, :]
Iteration 1: A[1024:2048, :] * B → C[1024:2048, :]
Iteration 2: A[2048:3072, :] * B → C[2048:3072, :]
Iteration 3: A[3072:4096, :] * B → C[3072:4096, :]
```

Tensors whose shapes already match the tile (e.g., `B` above) have stride 0 — they are reused across iterations without offset changes.

### Front-End Interface

**LaunchKernel(Program, List\<SpyreTensor\>)** — The entry point in torch-spyre. Performs any front-end setup/checks and delegates to either `Launch` or `LaunchTiled` on the Program depending on whether the tensors match the compiled tile size exactly or require tiling.

## Workflows

### Workflow 1: Tensor Allocation and Transfer

```
┌─────────────┐     ┌─────────────────┐     ┌───────────┐     ┌───────────┐
│  CPUTensor   │────▶│ SpyreAllocator  │────▶│ DMAEngine │────▶│ SpyreTensor │
│  (host)      │     │ allocate block  │     │ Copy to   │     │ (device)  │
│              │     │ → segment_id    │     │ device    │     │           │
│              │     │ → vf_offset     │     │           │     │           │
└─────────────┘     └─────────────────┘     └───────────┘     └───────────┘
```

1. User creates a `CPUTensor` and calls `.to(device)`
2. SpyreAllocator allocates a block within a device segment, producing `location_in_segment` (segment_id + vf_offset)
3. DMAEngine copies tensor data from host to the allocated device location
4. Result is an `SpyreTensor` carrying the device address metadata

### Workflow 2: Program Compilation and Loading

```
┌──────────┐     ┌───────────┐     ┌─────────────┐     ┌─────────────────┐
│ Inductor │────▶│ Deeptools │────▶│   Program   │────▶│    DMAEngine    │
│ (sdsc)   │     │ (compile) │     │ (from plan) │     │ Copy to device  │
└──────────┘     └───────────┘     └─────────────┘     └─────────────────┘
                       │
                       ▼
                 ExecutionPlan
                 SpyreCompute
                 PreProcessCompute
```

1. `torch.compile` triggers the inductor frontend, producing sdsc inputs for deeptools
2. Deeptools (backend compiler) produces spyre code: `SpyreCompute`, `PreProcessCompute`, `ExecutionPlan`
3. `Program(ExecutionPlan)` constructs the program and its instructions
4. SpyreAllocator allocates space for the program binary → `location_in_segment`
5. DMAEngine copies the program to device at the allocated location
6. Program is cached for reuse

### Workflow 3: Strict Execution (Launch)

```
┌──────────────┐     ┌──────────┐     ┌────────────┐     ┌───────────┐
│ LaunchKernel │────▶│ Validate │────▶│ Specialize │────▶│  Execute  │
│ (torch-spyre)│     │ shapes   │     │ ctrl blocks│     │  on Spyre │
└──────────────┘     └──────────┘     └────────────┘     └───────────┘
```

1. `LaunchKernel(program, tensors)` is called from torch-spyre
2. Tensor shapes are validated against the program's expected tile shapes
3. Shapes match exactly → delegates to `Program.Launch(tensors)`
4. Control blocks are specialized with tensor device addresses (segment_id + vf_offset)
5. Control blocks are submitted to the scheduler
6. Scheduler dispatches to the Spyre and waits for completion

### Workflow 4: Tiled Execution (LaunchTiled)

```
┌──────────────┐     ┌───────────┐     ┌──────────────────────────────────┐
│ LaunchKernel │────▶│ Detect    │────▶│         Tiling Loop              │
│ (torch-spyre)│     │ tile      │     │                                  │
└──────────────┘     │ mismatch  │     │  ┌─────────┐  ┌──────┐  ┌─────┐ │
                     └───────────┘     │  │ Compute │  │Spclz │  │ Exec │ │
                                       │  │ offset  │─▶│ CBs  │─▶│ Spyre│ │
                                       │  │ for i   │  │      │  │    │ │
                                       │  └─────────┘  └──────┘  └─────┘ │
                                       │       ▲                    │     │
                                       │       └────── next i ◀────┘     │
                                       └──────────────────────────────────┘
```

1. `LaunchKernel(program, tensors)` is called from torch-spyre
2. Tensor shapes are compared against the program's expected tile shapes
3. Mismatch detected (tensor larger than tile) → delegates to `Program.LaunchTiled(tensors)`
4. Runtime infers:
   - Which dimension(s) are tiled (shape mismatch between tensor and kernel)
   - `num_iterations` per tiled dimension
   - Per-tensor stride (from tensor metadata; 0 for tensors that match the tile)
5. For each iteration:
   - Compute each tensor's offset for this iteration using its stride and vf_offset
   - Re-specialize control blocks with updated addresses
   - Execute on Spyre
   - Wait for completion before next iteration

### Workflow 5: End-to-End

```
1. tensor_a = torch.randn(4096, 1024)          # Create on CPU
2. spyre_tensor_a = tensor_a.to("spyre")              # Allocate via SpyreAllocator + DMA copy → SpyreTensor
3. tensor_b = torch.randn(1024, 1024).to("spyre")

4. compiled_fn = torch.compile(matmul_fn)       # Triggers inductor → deeptools
                                                 # Deeptools compiles for tile [1024, 1024]
                                                 # Program created from ExecutionPlan, cached
                                                 # Program binary DMA'd to device

5. result = compiled_fn(spyre_tensor_a, tensor_b)     # LaunchKernel invoked
                                                 # Detects tensor_a [4096,1024] > tile [1024,1024]
                                                 # → LaunchTiled: 4 iterations
                                                 # Each iteration: specialize CBs with offset, execute
```

## Open Questions

1. **Reduction tiling**: When tiling along the reduction dimension (K in matmul), partial results must be accumulated. Should the runtime handle accumulation internally, or should this be expressed as a separate instruction in the ExecutionPlan?

2. **Multi-dimensional tiling**: If both M and N exceed the tile size, the runtime needs a nested loop. Should `LaunchTiled` support arbitrary nesting, or should this be constrained to single-dimension tiling with multi-dim handled by the compiler?

3. **Async iteration overlap**: Currently each iteration waits for completion before starting the next (since device memory is reused). Could we double-buffer to overlap iteration N+1's DMA with iteration N's compute?

4. **LaunchKernel routing**: Should `LaunchKernel` automatically decide between `Launch` and `LaunchTiled`, or should the caller explicitly choose? Auto-detection is simpler for users but hides behavior.

5. **Remainder handling**: What happens when the tensor dimension is not evenly divisible by the tile size (e.g., tensor is 4000 with tile 1024)? Options: fail, pad, or compile a separate remainder kernel.
