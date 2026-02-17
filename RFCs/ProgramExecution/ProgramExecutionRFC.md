# Program Execution Pipeline

**Authors:**
* @JRosenkranz

## **Summary**

This RFC introduces a layered stream-based execution model for torch-spyre. `SpyreStream` (in torch-spyre) implements the PyTorch Stream interface and serves as a thin passthrough that translates PyTorch-native data into the types `FlexStream` (in flex) understands. `FlexStream` is the core execution engine — modeled after CUDA streams, it accepts Operations (representing both compute and data transfers), decomposes them into the correct sequence of low-level steps (copies, launches), and forwards those steps to the Scheduler for hardware dispatch. FlexStream's single `Launch` method automatically detects when tensor shapes exceed the compiled tile size and handles tiled execution transparently — no separate tiling API is needed. All FlexStream methods are asynchronous, returning control to the host immediately, with operations executing sequentially within a stream. A `Scheduler` layer underneath dispatches work onto hardware, respecting the intra-stream FIFO ordering guarantee, while the stream abstraction preserves forward compatibility with future concurrent execution support.

## **Motivation**

The current connection between torch-spyre and the flex runtime relies on graph execution, which carries significant overhead for what is fundamentally a kernel launch operation. The graph-based approach requires constructing, optimizing, and traversing a graph structure even for single-kernel execution, adding complexity and latency that is unnecessary for the common case.

This RFC replaces that approach with a layered stream architecture:

- **SpyreStream** (torch-spyre) implements the PyTorch Stream interface and serves as a thin passthrough layer. It translates PyTorch-native data (torch tensors, device metadata) into the types FlexStream understands (Operations, VirtualAddresses) and delegates all execution logic to FlexStream.

- **FlexStream** (flex) is the core execution engine, modeled after CUDA streams. It accepts Operations — each representing a unit of work with an optional Compute and a list of preprocessing steps — and decomposes them into the correct sequence of low-level steps (copies, launches) that are forwarded to the Scheduler. FlexStream understands that a single logical operation like a matmul may require multiple steps (a DMA of offset inputs, a program correction launch, and the actual compute launch) and owns this decomposition. FlexStream's `Launch` method automatically detects when tensor shapes exceed the compiled tile size and handles tiled execution transparently — reusing compilation artifacts by enqueuing multiple rounds of low-level steps with updated tensor offsets — without requiring a separate API call. All methods are asynchronous — operations enqueued on a stream return immediately. Within a single stream, operations execute in FIFO order; across streams, no ordering is guaranteed.

- **Scheduler** (flex) sits underneath FlexStream and dispatches low-level steps onto hardware, respecting the intra-stream FIFO ordering guarantee. The Scheduler's internal design is covered in a separate Scheduler RFC.

ExecutionPlans remain lightweight containers of Operations built from compiler output. They do not own execution logic — instead, each Operation's binaries are individually loaded to device (each via a preprocessing-only Operation containing a CopyToDevice), and then the compute Operations are submitted through SpyreStream → FlexStream, which decomposes them into low-level steps for the Scheduler.

As a secondary benefit, FlexStream's `Launch` automatically detects when tensors exceed the compiled tile size and transparently reuses the kernel by enqueuing multiple rounds of low-level steps with updated offsets — no recompilation or separate API call required.

The introduction of the SpyreAllocator's VF mode also shapes this design: tensors are now carved from pre-allocated memory regions with block-level offsets rather than independently allocated, and the execution pipeline accounts for this addressing model natively.

## **Proposed Implementation**

### Core Components

#### VirtualAddress

A reference to a location in device memory, produced by SpyreAllocator. A VirtualAddress identifies only a location — size is stored separately alongside it.

Properties:

| Property | Description |
|----------|-------------|
| `region_id` | Identifies the memory region. In VF mode, this is an index into a firmware lookup table that maps to the physical address of the region. In PF mode, this is the physical address of the region itself. |
| `offset` | Byte offset of the allocation within the region. |

The same structure is used in both PF and VF modes — only the interpretation of `region_id` differs. All downstream components (SpyreTensor, Operation, FlexStream) work uniformly with VirtualAddress regardless of mode.

#### SpyreAllocator

Manages device memory and produces `VirtualAddress` values in one of two modes:

- **PF Mode**: Manages a pool of up to 8 memory regions. Returns a `VirtualAddress` where `region_id` is the physical address of the region itself.
- **VF Mode**: Manages a pool of up to 8 memory regions. Returns a `VirtualAddress` where `region_id` is an index into a firmware lookup table that maps to the physical address of the region.

In both modes, individual allocations carve blocks from these regions with 128-byte alignment, and the `offset` field of the returned VirtualAddress is the byte offset of the block within the region.

See the SpyreAllocator RFC (TBD) for full details on allocation strategies, memory region management, and memory lifecycle.

#### SpyreTensor

A tensor residing on-device. Carries metadata required for execution:

- **shape**: Logical dimensions (e.g., `[4096, 1024]`)
- **stride**: Memory stride per dimension (bytes between consecutive elements along each axis)
- **data_type**: Element type (e.g., float32, uint32)
- **virtual_address**: A `VirtualAddress` referencing the tensor's location on device (from SpyreAllocator)
- **size_bytes**: Total byte size of the tensor data
- **layout**: A `SpyreTensorLayout` describing the tensor's tiled layout on device — includes `device_size` (tiled dimensions on device), `device_dtype` (on-device data type), and `dim_map` (mapping between logical shape and device dimensions)

#### Operation

An Operation encapsulates a unit of work to be submitted through FlexStream. An Operation is composed of an optional Compute and a list of PreProcessCompute steps. PreProcessCompute steps may include data transfers (CopyToDevice, CopyFromDevice) as well as device-side preprocessing (e.g., program correction). When FlexStream decomposes an Operation, each PreProcessCompute maps to the appropriate low-level step (copy or launch), followed by a launch for the Compute if present.

Each Operation carries its own `virtual_addresses` — one per binary that has been loaded to device (e.g., a program correction binary and a compute binary). These are set when the Operation's binaries are individually copied to device during ExecutionPlan loading.

Constructors:

| Constructor | Description |
|-------------|-------------|
| `Operation(Compute)` | Single compute operation with no preprocessing |
| `Operation(Compute, List<PreProcessCompute>)` | Compute with required preprocessing (e.g., data transfer for program correction inputs, then program correction, then compute) |
| `Operation(List<PreProcessCompute>)` | Preprocessing only — no compute. Used for standalone data transfers such as loading a tensor or program binary to device |

An Operation should be self-contained: if a compute requires program correction, the data transfer for correction inputs, the correction itself, and the compute must all be part of the same Operation. For pure data movement (e.g., tensor `.to(device)` or program loading), an Operation with only PreProcessCompute entries and no Compute is used.

#### PreProcessCompute

A preprocessing step within an Operation. Each PreProcessCompute maps to a single low-level step when FlexStream decomposes an Operation. PreProcessCompute is a tagged variant — it represents either a data transfer or a device-side compute that must run before the Operation's main Compute.

Variants:

| Variant | Properties | Description |
|---------|------------|-------------|
| `CopyToDevice` | `host_address`, `virtual_address`, `size` | Host-to-device data transfer. Used for copying tensor data, program binaries, or correction input metadata to device. `size` is the byte count to transfer. |
| `CopyFromDevice` | `host_address`, `virtual_address`, `size` | Device-to-host data transfer. Used for reading results back from device. |
| `DeviceLaunch` | `virtual_address` | Execute a binary already resident on device. Used for device-side preprocessing such as program correction. |

**Program correction metadata**: When a PreProcessCompute of type `CopyToDevice` is used for program correction inputs, it copies a host-assembled buffer of tensor location metadata (VirtualAddresses of the resident input/output tensors) to the correction program's input area on device. The size of this buffer is determined at compile time — the compiler knows how many inputs/outputs the kernel expects and the format the correction program requires — and is encoded in the ExecutionPlan's program correction metadata, then captured on the PreProcessCompute when the Operation is constructed.

#### Compute

Represents the actual compute to be executed on the Spyre device. A Compute maps to a compiled binary that has been loaded to device. It is the terminal step in an Operation's decomposition — all PreProcessCompute steps run first, then the Compute is launched.

Properties:

| Property | Description |
|----------|-------------|
| `virtual_address` | `VirtualAddress` referencing the compute binary's location on device (set during ExecutionPlan loading) |
| `expected_input_shapes` | Expected tensor shapes for this compute's inputs, as determined by the compiler. Used by FlexStream to detect tiling requirements. |

A Compute is produced by the backend compiler (deeptools) as part of the ExecutionPlan. The binary is loaded to device during ExecutionPlan loading, at which point the `virtual_address` is set. At launch time, FlexStream decomposes the containing Operation and constructs the appropriate control block (CB) for hardware submission.

#### ExecutionPlan

Produced by the backend compiler (deeptools). An ExecutionPlan starts as compiler output describing the ordered sequence of computes and preprocessing required, then becomes the runtime artifact once its Operations are loaded to device. Loading an ExecutionPlan means loading each Operation's binaries individually — each binary gets its own CopyToDevice and its own virtual_address stored on the Operation.

Constructors:

| Constructor | Description |
|-------------|-------------|
| `ExecutionPlan()` | Empty plan, operations added manually |
| `ExecutionPlan(compiler_output)` | Auto-populated from compiler output |

Properties:

| Property | Description |
|----------|-------------|
| `tensor_input_metadata` | Expected input/output tensor metadata from the compiler |
| `operations` | Ordered list of Operations — each Operation carries its own virtual addresses after loading |
| `correction_metadata` | Program correction metadata (including the byte size of the correction input buffer per Operation) |
| `init_packets` | Init packets from the compiler |

ExecutionPlans do not own execution logic. They are passive data structures whose Operations have their binaries individually loaded to device memory (one DMA per binary) and are then submitted through SpyreStream → FlexStream for execution.

#### SpyreStream

A thin wrapper around `FlexStream` that implements the PyTorch Stream interface. SpyreStream is the torch-spyre-facing API — it translates PyTorch-native data (torch tensors, device metadata) into the types FlexStream understands (Operations, VirtualAddresses) and delegates all execution logic to the underlying FlexStream.

SpyreStream contains no decomposition logic, tiling logic, or understanding of Operation internals. It is a passthrough layer whose purpose is to bridge the PyTorch runtime conventions with the flex execution engine.

**Semantics:**
- Implements the PyTorch Stream interface so that torch-spyre integrates with PyTorch's stream management (e.g., `torch.spyre.Stream`, `torch.spyre.current_stream()`)
- All methods are **asynchronous** — control returns to the host immediately (inherited from FlexStream)
- Within a SpyreStream, operations execute **sequentially** (inherited from FlexStream's FIFO ordering)

Methods:

| Method | Description |
|--------|-------------|
| `Launch(Operation, List<SpyreTensor>, allow_tiled_launch=true)` | Translate SpyreTensors to VirtualAddresses and delegate to `FlexStream.Launch(operation, virtual_addresses, allow_tiled_launch)` |
| `Synchronize()` | Delegate to `FlexStream.Synchronize()` |

#### FlexStream

The core execution engine in flex, modeled after CUDA streams. A FlexStream accepts Operations — each representing a unit of work with an optional Compute and preprocessing steps — and decomposes them into the correct sequence of low-level steps (copies, launches) that are forwarded to the Scheduler as control blocks (CBs) for hardware dispatch.

FlexStream understands the internal structure of Operations: it knows that a matmul may require a DMA of correction metadata, a program correction launch, and the actual compute launch, and it generates the correct sequence of low-level steps for each. FlexStream's `Launch` method also automatically detects when tensor shapes exceed the compiled tile size and handles tiled execution transparently — enqueuing multiple rounds of low-level steps with updated tensor offsets — without requiring a separate API call.

**Semantics:**
- All methods are **asynchronous** — control returns to the host immediately after enqueuing
- Within a single FlexStream, operations execute **sequentially** in FIFO order
- Across different FlexStreams, operations have **no ordering guarantees**
- Each FlexStream is identified by a `stream_index`

Methods:

| Method | Description |
|--------|-------------|
| `Launch(Operation, List<virtual_address>, allow_tiled_launch=true)` | Decompose an Operation into low-level steps and enqueue them. If tensor shapes exceed the compiled tile size and `allow_tiled_launch` is true, automatically enqueues multiple tiled iterations. If shapes exceed the tile size and `allow_tiled_launch` is false, raises an exception. |
| `Synchronize()` | Block the host until all previously enqueued operations on this stream have completed |

**Launch(Operation, List\<virtual_address\>, allow_tiled_launch=true)**

Takes an Operation, the virtual addresses for resident input/output tensors, and an optional `allow_tiled_launch` flag (defaults to true).

FlexStream first compares each tensor's shape against the kernel's expected input shapes (from the Operation's Compute metadata). There are three cases:

1. **Shapes match exactly** — FlexStream walks the Operation's PreProcessCompute list, mapping each entry to the appropriate low-level step (CopyToDevice, CopyFromDevice, or device launch), then enqueues a launch for the Compute if present. Each low-level step maps directly to a control block (CB) forwarded to the Scheduler.

2. **Shapes exceed tile size and `allow_tiled_launch` is true** — FlexStream infers the tiling dimension(s) and iteration count, then enqueues multiple rounds of low-level steps — one full Operation decomposition per tile iteration, each with updated tensor offsets. See [Tiled Execution](#tiled-execution) for details.

3. **Shapes exceed tile size and `allow_tiled_launch` is false** — FlexStream raises an exception indicating that the tensor shapes do not match the compiled tile size and tiled launch is not permitted.

**Example 1 — Tensor load (preprocessing only, no compute):**

An Operation with a single CopyToDevice PreProcessCompute and no Compute:

1. **CopyToDevice** — Copy tensor data from host to device

Single low-level step. No launch follows because there is no Compute.

**Example 2 — Matmul with program correction (preprocessing + compute):**

An Operation with PreProcessCompute entries (CopyToDevice for correction inputs, DeviceLaunch for program correction) and a Compute (matmul):

1. **CopyToDevice** — Copy the offset inputs (tensor device addresses used for program correction) from host to the correction program's input location on device
2. **Launch** (program correction) — Execute the program correction compute, which patches the matmul program with the correct tensor addresses
3. **Launch** (matmul compute) — Execute the corrected matmul program

Each of these is enqueued on the stream in order. Because FlexStream is FIFO, the correction completes before the matmul begins, and the matmul begins only after its program has been corrected.

**Synchronize()**

Blocks the calling host thread until all operations previously enqueued on this stream have completed. This is the only blocking method on FlexStream.

#### Scheduler

The Scheduler sits underneath FlexStream and is responsible for dispatching low-level steps onto hardware. From the perspective of this RFC, the Scheduler is an opaque component — FlexStream submits control blocks to it, and the Scheduler ensures they are executed respecting the intra-stream FIFO ordering guarantee. The Scheduler's internal design (scheduling policies, hardware serialization strategy, multi-stream interleaving) is covered in a separate Scheduler RFC.

### Tiled Execution

Tiled execution is handled automatically by `FlexStream.Launch` when `allow_tiled_launch` is true (the default). When a tensor is larger than the compiled tile size, FlexStream reuses the compiled kernel across the full tensor by enqueuing multiple rounds of low-level steps — one full Operation decomposition per iteration, each with updated tensor offsets. If `allow_tiled_launch` is false and shapes exceed the tile size, Launch raises an exception.

**Preconditions:**
- Each tensor dimension is greater than or equal to the corresponding compiled tile dimension
- Each tensor dimension is evenly divisible by the corresponding tile dimension
- Tensor strides are consistent with the tiling dimension

**Behavior:**
1. Compare each tensor's shape against the kernel's expected input shapes (from the Operation's Compute metadata)
2. Infer the tiling dimension(s) and compute `num_iterations = tensor_dim / tile_dim`
3. For each iteration `i`:
   a. Compute per-tensor offset: `offset_i = base_offset + (i * tensor_stride_along_tiled_dim * tile_size * element_size)`
   b. Decompose the Operation into low-level steps with updated device addresses (CopyToDevice for correction inputs, Launch for correction, Launch for compute)
   c. All steps enqueued asynchronously — sequential within the stream

**Example — Matmul tiling along M (with program correction):**

Kernel compiled for `A[1024, K] * B[K, N] = C[1024, N]`. Actual tensor `A` is `[4096, K]`, `C` is `[4096, N]`:

```
num_iterations = 4096 / 1024 = 4

Iteration 0: A[0:1024, :]    * B → C[0:1024, :]
Iteration 1: A[1024:2048, :] * B → C[1024:2048, :]
Iteration 2: A[2048:3072, :] * B → C[2048:3072, :]
Iteration 3: A[3072:4096, :] * B → C[3072:4096, :]
```

Each iteration enqueues onto the FlexStream:
```
CopyToDevice(correction_offsets_iter_i)  →  Launch(program_correction)  →  Launch(matmul_compute)
```

Tensors whose shapes already match the tile (e.g., `B` above) have stride 1 — they are reused across iterations without offset changes.

### Front-End Interface

**LaunchKernel(SpyreStream, ExecutionPlan, List\<SpyreTensor\>)** — The entry point in torch-spyre. For each Operation in the ExecutionPlan, delegates to `SpyreStream.Launch(operation, tensors, allow_tiled_launch)`. The `allow_tiled_launch` value can be controlled by a user environment setting (e.g., `SPYRE_ALLOW_TILED_LAUNCH`), allowing users to disable automatic tiling for debugging or to enforce that tensor shapes exactly match the compiled tile size. SpyreStream translates the SpyreTensors to VirtualAddresses and passes through to `FlexStream.Launch`, which automatically detects whether tensor shapes match the compiled tile size or require tiled execution. Control returns to the host immediately; use `SpyreStream.Synchronize()` to wait for completion.

### Workflows

#### Workflow 1: Tensor Allocation and Transfer

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  CPUTensor   │────▶│ SpyreAllocator  │────▶│   SpyreStream    │────▶│    FlexStream    │
│  (host)      │     │ allocate block  │     │ (translate data, │     │ Launch(Operation │
│              │     │ → VirtualAddress│     │  passthrough)    │     │  [CopyToDevice]) │
│              │     │                 │     │                  │     │ decompose→sched  │
└─────────────┘     └─────────────────┘     └──────────────────┘     └──────────────────┘
```

1. User creates a `CPUTensor` and calls `.to(device)`
2. SpyreAllocator allocates a block, producing a `VirtualAddress` (`region_id` + `offset`; interpretation of `region_id` depends on PF vs VF mode)
3. A preprocessing-only Operation is created with a CopyToDevice PreProcessCompute (host_address, virtual_address, size)
4. `SpyreStream.Launch(op, tensors)` translates SpyreTensors → VirtualAddresses, then delegates to `FlexStream.Launch(op, virtual_addresses)`
5. FlexStream decomposes the Operation into a CopyToDevice low-level step and forwards it to the Scheduler — returns immediately
6. Result is a `SpyreTensor` carrying the device address metadata
7. Host may continue work; data transfer proceeds asynchronously on the stream

#### Workflow 2: Compilation and Loading

```
┌──────────┐     ┌───────────┐     ┌───────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Inductor │────▶│ Deeptools │────▶│ ExecutionPlan │────▶│   SpyreStream    │────▶│    FlexStream    │
│ (sdsc)   │     │ (compile) │     │               │     │ (passthrough)    │     │ Launch(Operation │
└──────────┘     └───────────┘     └───────────────┘     │ × N (per binary) │     │  [CopyToDevice]) │
                                                          └──────────────────┘     │ × N              │
                                                                                   └──────────────────┘
```

1. `torch.compile` triggers the inductor frontend, producing sdsc inputs for deeptools
2. Deeptools (backend compiler) produces an `ExecutionPlan` — an ordered list of Operations, each containing `Compute` and `PreProcessCompute` steps
3. For each Operation in the ExecutionPlan, for each binary that needs to be on device (e.g., program correction binary, compute binary):
   a. SpyreAllocator allocates space for the binary → `virtual_address`
   b. A preprocessing-only Operation is created with a CopyToDevice PreProcessCompute
   c. `SpyreStream.Launch(op, [])` passes through to `FlexStream.Launch(op, [])`, which decomposes into a CopyToDevice step — `virtual_address` is stored on the Operation
4. ExecutionPlan (with all Operation virtual addresses populated) is cached for reuse

#### Workflow 3: Detailed Execution — LaunchKernel to Hardware

This diagram shows the full path from `LaunchKernel` through every layer for a matmul with program correction, illustrating how FlexStream decomposes a single Operation into low-level steps that the Scheduler drains onto hardware.

```
 torch-spyre                          flex
┌─────────────────────────────────┐  ┌──────────────────────────────────────────────────┐
│                                 │  │                                                  │
│  LaunchKernel(spyre_stream,     │  │                                                  │
│        exec_plan, tensors)      │  │                                                  │
│         │                       │  │                                                  │
│         ▼                       │  │                                                  │
│  SpyreStream.Launch(op, tensors,│  │                                                  │
│    allow_tiled_launch)          │  │                                                  │
│         │                       │  │                                                  │
│  (translate tensors →           │  │                                                  │
│   virtual_addresses, delegate)  │  │                                                  │
│         │                       │  │                                                  │
│         └───────────────────────┼──┼─▶ FlexStream.Launch(op, vaddrs,                 │
│                                 │  │     allow_tiled_launch)                          │
│                                 │  │         │                                        │
└─────────────────────────────────┘  │         │                                        │
                                     │    ┌────┘                                        │
                                     │    │  FlexStream compares tensor shapes           │
                                     │    │  against Compute.expected_input_shapes:      │
                                     │    │                                             │
                                     │    │  ┌──────────────────────────────────────┐   │
                                     │    │  │ shapes match exactly?                │   │
                                     │    │  │   YES → decompose single iteration   │   │
                                     │    │  │   NO  → allow_tiled_launch?          │   │
                                     │    │  │         YES → tiled iterations       │   │
                                     │    │  │         NO  → raise exception        │   │
                                     │    │  └──────────────────────────────────────┘   │
                                     │    │                                             │
                                     │    │  (showing exact-match case below;           │
                                     │    │   see Workflow 4 for tiled case)            │
                                     │    │                                             │
                                     │    │  Operation contains:                        │
                                     │    │    Compute (matmul)                         │
                                     │    │    PreProcessCompute (program correction)   │
                                     │    │    tensor virtual addresses: [A, B]         │
                                     │    │                                             │
                                     │    │  Decomposed into low-level steps:           │
                                     │    ▼                                             │
                                     │  ┌───────────────────────────────────────────┐   │
                                     │  │            FlexStream                     │   │
                                     │  │  (FIFO — sequential within stream)        │   │
                                     │  │                                           │   │
                                     │  │  ┌─────────────────────────────────────┐  │   │
                                     │  │  │ 1. CopyToDevice(                    │  │   │
                                     │  │  │      correction_offsets_host,       │  │   │
                                     │  │  │      correction_input_vaddr,        │  │   │
                                     │  │  │      size)                          │  │   │
                                     │  │  │    Copy tensor offset inputs to     │  │   │
                                     │  │  │    device for program correction    │  │   │
                                     │  │  ├─────────────────────────────────────┤  │   │
                                     │  │  │ 2. Launch(                          │  │   │
                                     │  │  │      program_correction_vaddr)      │  │   │
                                     │  │  │    Execute program correction —     │  │   │
                                     │  │  │    patches matmul program with      │  │   │
                                     │  │  │    correct tensor addresses         │  │   │
                                     │  │  ├─────────────────────────────────────┤  │   │
                                     │  │  │ 3. Launch(                          │  │   │
                                     │  │  │      matmul_compute_vaddr)          │  │   │
                                     │  │  │    Execute the corrected matmul     │  │   │
                                     │  │  │    A × B → C                        │  │   │
                                     │  │  └─────────────────────────────────────┘  │   │
                                     │  └──────────────────┬────────────────────────┘   │
                                     │                     │                            │
                                     │                     ▼                            │
                                     │  ┌──────────────────────────────────────────┐    │
                                     │  │              Scheduler                   │    │
                                     │  │                                          │    │
                                     │  │  Receives CBs from FlexStream:           │    │
                                     │  │    CB(copy offsets)                      │    │
                                     │  │    CB(correction launch)                │    │
                                     │  │    CB(matmul launch)                    │    │
                                     │  │                                          │    │
                                     │  │  Drains FIFO onto hardware:              │    │
                                     │  │  ┌────────┐ ┌────────┐ ┌────────┐       │    │
                                     │  │  │ Copy   │→│ Corr   │→│ Matmul │→ done │    │
                                     │  │  │ offsets│ │ launch │ │ launch │       │    │
                                     │  │  └────────┘ └────────┘ └────────┘       │    │
                                     │  └──────────────────────────────────────────┘    │
                                     │                                                  │
                                     └──────────────────────────────────────────────────┘

 Host returns immediately after enqueuing.
 Call SpyreStream.Synchronize() to block until hardware completes.
```

**Step-by-step:**

1. `LaunchKernel(spyre_stream, exec_plan, tensors)` is called from torch-spyre
2. For each Operation, delegates to `SpyreStream.Launch(op, tensors, allow_tiled_launch)`
3. SpyreStream translates SpyreTensors → VirtualAddresses and calls `FlexStream.Launch(op, virtual_addresses, allow_tiled_launch)`
4. FlexStream compares tensor shapes against the Compute's expected input shapes:
   - Shapes match exactly → proceeds with single-iteration decomposition (this workflow)
   - Shapes exceed tile size and `allow_tiled_launch` is true → tiled iterations (see Workflow 4)
   - Shapes exceed tile size and `allow_tiled_launch` is false → raises exception
5. FlexStream inspects the Operation's internal structure:
   - Identifies the PreProcessCompute (program correction) and Compute (matmul)
   - Prepares correction offset inputs (tensor device addresses that the correction program needs)
6. FlexStream decomposes the Operation into low-level steps and forwards each as a control block to the Scheduler:
   - CopyToDevice(correction_offsets_host, correction_input_vaddr, size) — upload offset inputs
   - Launch(program_correction_vaddr) — run program correction
   - Launch(matmul_compute_vaddr) — run the corrected matmul
7. Scheduler drains control blocks onto hardware in FIFO order
8. Host returns immediately; call `SpyreStream.Synchronize()` when results are needed

#### Workflow 4: Tiled Execution

```
 torch-spyre                                    flex
┌──────────────────────────────────┐  ┌──────────────────────────────────────────────┐
│                                  │  │                                              │
│  LaunchKernel(spyre_stream,      │  │                                              │
│    exec_plan, tensors)           │  │                                              │
│         │                        │  │                                              │
│  SpyreStream.Launch(op, tensors, │  │                                              │
│    allow_tiled_launch=true)      │  │                                              │
│         │                        │  │                                              │
│  (translate tensors → vaddrs)    │  │                                              │
│         │                        │  │                                              │
│         └────────────────────────┼──┼─▶ FlexStream.Launch(op, vaddrs,             │
│                                  │  │     allow_tiled_launch=true)                 │
│                                  │  │         │                                    │
└──────────────────────────────────┘  │  Detects shapes exceed tile size             │
                                      │  allow_tiled_launch=true → proceed           │
                                      │  Infer: 4 iterations (4096/1024)             │
                                      │         │                                    │
                                      │    ┌────┴────┐                               │
                                      │    │ iter 0  │─▶ Copy(offsets_0)→Launch(corr)│
                                      │    │         │   →Launch(matmul)             │
                                      │    ├─────────┤                               │
                                      │    │ iter 1  │─▶ Copy(offsets_1)→Launch(corr)│
                                      │    │         │   →Launch(matmul)             │
                                      │    ├─────────┤                               │
                                      │    │ iter 2  │─▶ Copy(offsets_2)→Launch(corr)│
                                      │    │         │   →Launch(matmul)             │
                                      │    ├─────────┤                               │
                                      │    │ iter 3  │─▶ Copy(offsets_3)→Launch(corr)│
                                      │    │         │   →Launch(matmul)             │
                                      │    └─────────┘                               │
                                      │   (12 CBs total, all async, FIFO in stream)  │
                                      │                     │                        │
                                      │                     ▼                        │
                                      │  ┌──────────────────────────────────────┐    │
                                      │  │            Scheduler                 │    │
                                      │  │  Drains all 12 CBs onto HW:         │    │
                                      │  │  Copy₀→Corr₀→Mat₀→Copy₁→Corr₁→Mat₁│    │
                                      │  │  →Copy₂→Corr₂→Mat₂→Copy₃→Corr₃→Mat₃│   │
                                      │  └──────────────────────────────────────┘    │
                                      └──────────────────────────────────────────────┘
```

1. `LaunchKernel(spyre_stream, exec_plan, tensors)` is called from torch-spyre
2. For each Operation, delegates to `SpyreStream.Launch(op, tensors, allow_tiled_launch=true)`
3. SpyreStream translates SpyreTensors → VirtualAddresses and calls `FlexStream.Launch(op, virtual_addresses, allow_tiled_launch=true)`
4. FlexStream compares tensor shapes against the Compute's expected input shapes — detects shapes exceed tile size
5. `allow_tiled_launch` is true → FlexStream proceeds with tiled execution
6. FlexStream infers tiling dimension and `num_iterations = 4096 / 1024 = 4`
7. For each iteration `i`, FlexStream:
   - Computes updated tensor offsets for iteration `i`
   - Decomposes the Operation into low-level steps with updated addresses:
     - `CopyToDevice(correction_offsets_i)` — upload corrected offset inputs for this iteration
     - `Launch(program_correction)` — patch the program with this iteration's addresses
     - `Launch(matmul_compute)` — execute the matmul for this tile slice
8. All 12 control blocks (3 per iteration × 4 iterations) are forwarded to the Scheduler
9. Scheduler drains all control blocks onto hardware in FIFO order
10. Host returns immediately; call `SpyreStream.Synchronize()` to block until all iterations complete

#### Workflow 5: End-to-End

```
                                                    # Default SpyreStream created at runtime start

1. tensor_a = torch.randn(4096, 1024)             # Create on CPU
2. spyre_a = tensor_a.to("spyre")                 # Operation([CopyToDevice]) → FlexStream decomposes → Scheduler
3. spyre_b = torch.randn(1024, 1024).to("spyre")  # Same — async, all through default SpyreStream → FlexStream

4. compiled_fn = torch.compile(matmul_fn)          # Lazy — no compilation yet

5. result = compiled_fn(spyre_a, spyre_b)          # First call triggers inductor → deeptools
                                                    # ExecutionPlan created, binaries loaded to device
                                                    # Cached via PyTorch compile caching
                                                    # Then: LaunchKernel(exec_plan, tensors)
                                                    # Uses default stream (or creates a new one)
                                                    # SpyreStream.Launch → FlexStream.Launch
                                                    # FlexStream detects tensor_a [4096,1024] > tile [1024,1024]
                                                    # allow_tiled_launch=true → enqueues 4 iterations
                                                    # (12 CBs: 3 per iteration)

6. result_cpu = result.to("cpu")                   # Operation([CopyFromDevice]) → FlexStream decomposes → Scheduler
```

#### Workflow 6: Multi-Stream (Future Hardware)

```
stream_a = SpyreStream()                           # Stream for layer 1
stream_b = SpyreStream()                           # Stream for layer 2

# Enqueue independent work on separate streams
LaunchKernel(stream_a, exec_plan_1, tensors_1)     # Async — SpyreStream passes through to FlexStream
LaunchKernel(stream_b, exec_plan_2, tensors_2)     # Async — no ordering w.r.t. stream_a

# With current hardware: Scheduler serializes both streams' operations
# With future hardware: Both streams may execute concurrently

stream_a.Synchronize()
stream_b.Synchronize()
```

## **Metrics**

TBD

## **Drawbacks**

TBD

## **Alternatives**

TBD

## **Prior Art**

### CUDA Streams and FlexStream

FlexStream is modeled after CUDA streams and shares the two fundamental guarantees: **async enqueue** (all methods return control to the host immediately) and **intra-stream FIFO ordering** (operations within a stream execute sequentially, with no ordering guarantees across streams).

The key difference is where work submission lives. A CUDA stream is a **passive handle** — you pass it as a parameter to external API calls like `cudaMemcpyAsync(dst, src, size, kind, stream)` or kernel launches via `<<<grid, block, sharedMem, stream>>>`. FlexStream is an **active object** — operations are submitted as methods on the stream itself (`stream.Launch(op, ...)`, `stream.Synchronize()`).

The following table maps CUDA stream capabilities to their FlexStream equivalents:

| Capability | CUDA Stream | FlexStream |
|------------|-------------|------------|
| Copy | `cudaMemcpyAsync(dst, src, size, kind, stream)` — standalone primitive call | Data transfers are expressed as PreProcessCompute steps within an Operation; FlexStream decomposes them into low-level copy steps internally |
| Launch | `kernel<<<grid, block, sharedMem, stream>>>()` — takes grid/block dims, function pointer, args | `Launch(Operation, virtual_addresses, allow_tiled_launch=true)` — accepts a compound Operation and decomposes it into the correct sequence of low-level steps (no grid/block dims since Spyre is not SIMT). Automatically detects when tensor shapes exceed the compiled tile size and handles tiled execution transparently when `allow_tiled_launch` is true. |
| Tiled launch | User changes grid dims to cover larger tensors | Handled automatically by `Launch` — no separate API. FlexStream iterates over tiles, enqueuing multiple decompositions with updated offsets. |
| Synchronize | `cudaStreamSynchronize(stream)` | `Synchronize()` — identical semantics |
| Events | `cudaEventRecord` / `cudaStreamWaitEvent` for inter-stream dependencies | Not present (see unresolved question #4) |
| Query | `cudaStreamQuery()` — non-blocking completion poll | Not present (see unresolved question #9) |
| Priorities | `cudaStreamCreateWithPriority()` | Not present (see unresolved question #10) |
| Host callbacks | `cudaLaunchHostFunc()` — enqueue a host-side function on the stream | Not present (see unresolved question #11) |
| Default stream | NULL stream with legacy blocking semantics | Not yet specified (see unresolved question #7) |

FlexStream is a deliberately minimal subset — 2 methods vs. CUDA's dozens. The omitted features (events, query, priorities, host callbacks) are captured as unresolved questions for future consideration as hardware and runtime requirements evolve.

### SpyreStream and FlexStream — Compound Operations

In CUDA, a kernel launch is self-contained — the user or library (cuBLAS, cuDNN) manually issues the correct sequence of memcpy and launch calls on a stream. Spyre operations have inherent **compound structure** that requires the runtime to decompose a single logical operation into multiple low-level steps:

- **A CUDA matmul**: 1 kernel launch
- **A Spyre matmul**: DMA correction metadata → run correction binary → run compute binary (3 low-level steps)

FlexStream owns this decomposition as a first-class part of the runtime. The closest CUDA analogs are vendor libraries like cuBLAS and cuDNN, which internally issue multi-step sequences on a user-provided stream — but those are external libraries, not a core stream layer. In the Spyre stack, compound decomposition is built into FlexStream because every Spyre compute requires it, not just optimized library calls.

SpyreStream, by contrast, is a thin PyTorch Stream wrapper with no CUDA analog needed — it simply bridges PyTorch runtime conventions (torch tensors, device metadata) with FlexStream's types (Operations, VirtualAddresses).

FlexStream's automatic tiling within `Launch` also has no CUDA equivalent. CUDA kernels accept arbitrary dimensions as launch parameters (`<<<grid, block>>>`), so the user simply changes grid dims to cover larger tensors. Spyre kernels are compiled for fixed tile shapes, so FlexStream's `Launch` must detect this and iterate over tiles, enqueuing multiple rounds of low-level steps with updated tensor offsets — all transparently within the same `Launch` call.

## **How we teach this**

TBD

## **Unresolved questions**

1. **Reduction tiling**: When tiling along the reduction dimension (K in matmul), partial results must be accumulated. Should the runtime handle accumulation internally, or should this be expressed as a separate operation in the ExecutionPlan?

2. **Multi-dimensional tiling**: If both M and N exceed the tile size, the runtime needs a nested loop. Should tiled execution support arbitrary nesting, or should this be constrained to single-dimension tiling with multi-dim handled by the compiler?

3. **Async iteration overlap**: Currently each tiled iteration within a stream is sequential. Could we use separate streams or double-buffering to overlap iteration N+1's data movement with iteration N's compute?

4. **Stream events / synchronization primitives**: Should FlexStream support inter-stream synchronization (e.g., CUDA-style events) to express dependencies between streams without full synchronization?

5. **Remainder handling**: What happens when the tensor dimension is not evenly divisible by the tile size (e.g., tensor is 4000 with tile 1024)? Options: fail, pad, or compile a separate remainder kernel.

6. **Scheduler policies**: Deferred to the Scheduler RFC — covers scheduling policy, hardware serialization strategy, and multi-stream interleaving.

7. **Default stream**: Should there be a default FlexStream (like CUDA's default stream) that is used when the user does not explicitly create one?

8. **Program correction across PF/VF modes**: The correction input buffer contains tensor locations derived from VirtualAddresses. Since VirtualAddress is now a single type (`region_id` + `offset`) in both modes, FlexStream can assemble the buffer uniformly — but the correction program on device may still need to interpret `region_id` differently (firmware lookup index vs. physical address). How should this mode distinction be communicated to the correction program?

9. **Non-blocking completion query**: Should FlexStream support a `Query()` method (analogous to `cudaStreamQuery()`) that returns whether all enqueued operations have completed without blocking? This would allow polling-based patterns and avoid unnecessary synchronization.

10. **Stream priorities**: Should FlexStream support creating streams with priority levels (analogous to `cudaStreamCreateWithPriority()`)? This is related to but distinct from unresolved question #6 (Scheduler policies) — #6 asks how the Scheduler drains multiple streams, while this asks whether streams themselves carry priority metadata that the Scheduler could use.

11. **Host callbacks**: Should FlexStream support enqueuing host-side functions onto the stream (analogous to `cudaLaunchHostFunc()`)? This would allow host work to be interleaved with device operations in stream order — useful for logging, triggering downstream work, or updating host-side state at specific points in the execution sequence.

## Resolution

TBD

### Level of Support

TBD

#### Additional Context

TBD

### Next Steps

TBD

#### Tracking issue

TBD

#### Exceptions

TBD
