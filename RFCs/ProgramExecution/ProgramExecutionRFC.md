# Program Execution Pipeline

**Authors:**
* @JRosenkranz

## **Summary**

This RFC introduces a layered stream-based execution model for torch-spyre. At the top, `SpyreStream` (in torch-spyre) provides a high-level interface that accepts SpyreOperations — representing both compute and data transfers — and decomposes them into FlexOperations on an underlying `FlexStream` (in flex). All device work flows through SpyreStream via a unified `Launch(SpyreOperation, ...)` interface. FlexStream, modeled after CUDA streams, provides an asynchronous interface for data movement and kernel launch — `CopyToDevice`, `CopyFromDevice`, `Launch`, and `Synchronize` — where all operations return control to the host immediately and are queued for sequential execution on the device. A `Scheduler` layer underneath serializes operations across streams for the current hardware, while the stream abstraction preserves forward compatibility with future concurrent execution support.

## **Motivation**

The current connection between torch-spyre and the flex runtime relies on graph execution, which carries significant overhead for what is fundamentally a kernel launch operation. The graph-based approach requires constructing, optimizing, and traversing a graph structure even for single-kernel execution, adding complexity and latency that is unnecessary for the common case.

This RFC replaces that approach with a layered stream architecture:

- **SpyreStream** (torch-spyre) provides the high-level interface. All device work — compute and data movement alike — is expressed as SpyreOperations and submitted through `SpyreStream.Launch`. A SpyreOperation bundles an optional compute with a list of preprocessing steps (which may include data transfers such as CopyToDevice/CopyFromDevice, as well as device-side operations like program correction). SpyreStream decomposes each SpyreOperation into the correct sequence of FlexOperations via FlexStream calls. SpyreStream also owns the `LaunchTiled` logic for reusing compilation artifacts when tensor shapes exceed the compiled tile size.

- **FlexStream** (flex) provides the low-level, CUDA-stream-like API: `CopyToDevice`, `CopyFromDevice`, `Launch`, and `Synchronize`. All methods are asynchronous — operations enqueued on a stream return immediately. Within a single stream, operations execute in FIFO order; across streams, no ordering is guaranteed.

- **Scheduler** (flex) sits underneath FlexStream and serializes operations from all active streams onto the hardware. Since the current Spyre hardware does not support concurrent execution of multiple streams, the Scheduler ensures all operations are dispatched in a valid order. This design cleanly separates the programming model (streams) from the hardware reality (serialized execution), allowing the stream API to remain stable as hardware capabilities evolve.

The separation between SpyreStream and FlexStream is deliberate: SpyreStream understands that a single logical operation like a matmul may require multiple steps — a DMA of offset inputs, a program correction launch, and the actual compute launch — and knows how to decompose a SpyreOperation into that sequence of FlexOperations. FlexStream has no knowledge of this breakdown; it only sees individual FlexOperations (copy and launch primitives). This keeps flex simple and generic while torch-spyre owns the operation-level semantics.

ExecutionPlans remain lightweight containers of SpyreOperations built from compiler output. They do not own execution logic — instead, each SpyreOperation's binaries are individually loaded to device (each via a preprocessing-only SpyreOperation containing a CopyToDevice), and then the compute SpyreOperations are submitted through SpyreStream, which decomposes them into FlexOperations.

As a secondary benefit, SpyreStream's `LaunchTiled` makes it straightforward to reuse a kernel compiled for a smaller tile size (e.g., 1024) against larger tensors (e.g., 4096) by enqueuing multiple rounds of FlexStream operations, without recompilation.

The introduction of the SpyreAllocator's VF mode also shapes this design: tensors are now carved from pre-allocated memory regions with block-level offsets rather than independently allocated, and the execution pipeline accounts for this addressing model natively.

## **Proposed Implementation**

### Core Components

#### DeviceHandle

An opaque reference to a location in device memory, produced by SpyreAllocator. DeviceHandle abstracts over the two allocation modes so that all downstream components (SpyreTensor, SpyreOperation, FlexStream) can work uniformly regardless of whether the hardware is in PF or VF mode. A DeviceHandle identifies only a location — size is stored separately alongside it.

Implementations:

| Variant | Properties | Description |
|---------|------------|-------------|
| `PFDeviceHandle` | `physical_address` | A direct physical address on the device. Used in PF mode where each allocation is independently mapped. |
| `VFDeviceHandle` | `region_id`, `vf_offset` | A location within a pre-allocated memory region. `region_id` identifies the region, `vf_offset` is the byte offset of the block within that region. Used in VF mode. |

All APIs that accept or return a `DeviceHandle` are agnostic to the variant — the underlying mode is determined by how SpyreAllocator is configured.

#### SpyreAllocator

Manages device memory and produces `DeviceHandle`s in one of two modes:

- **PF Mode**: Each allocation is independently mapped on hardware. Returns a `PFDeviceHandle` containing the physical address.
- **VF Mode**: Manages a pool of up to 8 memory regions. Individual allocations carve blocks from these regions with 128-byte alignment. Returns a `VFDeviceHandle` containing the region_id and vf_offset.

See the SpyreAllocator RFC (TBD) for full details on allocation strategies, memory region management, and memory lifecycle.

#### SpyreTensor

A tensor residing on-device. Carries metadata required for execution:

- **shape**: Logical dimensions (e.g., `[4096, 1024]`)
- **stride**: Memory stride per dimension (bytes between consecutive elements along each axis)
- **data_type**: Element type (e.g., float32, uint32)
- **device_handle**: A `DeviceHandle` referencing the tensor's location on device (from SpyreAllocator)
- **size_bytes**: Total byte size of the tensor data
- **layout**: A `SpyreTensorLayout` describing the tensor's tiled layout on device — includes `device_size` (tiled dimensions on device), `device_dtype` (on-device data type), and `dim_map` (mapping between logical shape and device dimensions)

#### SpyreOperation

A SpyreOperation encapsulates a unit of work to be submitted through SpyreStream. A SpyreOperation is composed of an optional SpyreCompute and a list of PreProcessCompute steps. PreProcessCompute steps may include data transfers (CopyToDevice, CopyFromDevice) as well as device-side preprocessing (e.g., program correction). When SpyreStream decomposes a SpyreOperation, each PreProcessCompute maps to the appropriate FlexStream primitive (CopyToDevice, CopyFromDevice, or Launch), followed by a Launch for the SpyreCompute if present.

Each SpyreOperation carries its own `device_handles` — one per binary that has been loaded to device (e.g., a program correction binary and a compute binary). These handles are set when the SpyreOperation's binaries are individually copied to device during ExecutionPlan loading.

Constructors:

| Constructor | Description |
|-------------|-------------|
| `SpyreOperation(SpyreCompute)` | Single compute operation with no preprocessing |
| `SpyreOperation(SpyreCompute, List<PreProcessCompute>)` | Compute with required preprocessing (e.g., data transfer for program correction inputs, then program correction, then compute) |
| `SpyreOperation(List<PreProcessCompute>)` | Preprocessing only — no compute. Used for standalone data transfers such as loading a tensor or program binary to device |

A SpyreOperation should be self-contained: if a compute requires program correction, the data transfer for correction inputs, the correction itself, and the compute must all be part of the same SpyreOperation. For pure data movement (e.g., tensor `.to(device)` or program loading), a SpyreOperation with only PreProcessCompute entries and no SpyreCompute is used.

#### PreProcessCompute

A preprocessing step within a SpyreOperation. Each PreProcessCompute maps to a single FlexStream primitive when SpyreStream decomposes a SpyreOperation. PreProcessCompute is a tagged variant — it represents either a data transfer or a device-side compute that must run before the SpyreOperation's main SpyreCompute.

Variants:

| Variant | Properties | Description |
|---------|------------|-------------|
| `CopyToDevice` | `host_address`, `device_handle`, `size` | Host-to-device data transfer. Used for copying tensor data, program binaries, or correction input metadata to device. `size` is the byte count to transfer. |
| `CopyFromDevice` | `host_address`, `device_handle`, `size` | Device-to-host data transfer. Used for reading results back from device. |
| `DeviceLaunch` | `device_handle` | Execute a binary already resident on device. Used for device-side preprocessing such as program correction. |

**Program correction metadata**: When a PreProcessCompute of type `CopyToDevice` is used for program correction inputs, it copies a host-assembled buffer of tensor location metadata (DeviceHandles of the resident input/output tensors) to the correction program's input area on device. The size of this buffer is determined at compile time — the compiler knows how many inputs/outputs the kernel expects and the format the correction program requires — and is encoded in the ExecutionPlan's program correction metadata, then captured on the PreProcessCompute when the SpyreOperation is constructed.

#### SpyreCompute

Represents the actual compute to be executed on the Spyre device. A SpyreCompute maps to a compiled binary that has been loaded to device and is launched via `FlexStream.Launch(device_handle)`. It is the terminal step in a SpyreOperation's decomposition — all PreProcessCompute steps run first, then the SpyreCompute is launched.

Properties:

| Property | Description |
|----------|-------------|
| `device_handle` | `DeviceHandle` referencing the compute binary's location on device (set during ExecutionPlan loading) |
| `expected_input_shapes` | Expected tensor shapes for this compute's inputs, as determined by the compiler. Used by SpyreStream to detect tiling requirements. |

A SpyreCompute is produced by the backend compiler (deeptools) as part of the ExecutionPlan. The binary is loaded to device during ExecutionPlan loading, at which point the `device_handle` is set. At launch time, SpyreStream enqueues a `FlexStream.Launch(device_handle)`, and FlexStream constructs the appropriate control block (CB) for hardware submission.

#### ExecutionPlan

Produced by the backend compiler (deeptools). An ExecutionPlan starts as compiler output describing the ordered sequence of computes and preprocessing required, then becomes the runtime artifact once its SpyreOperations are loaded to device. Loading an ExecutionPlan means loading each SpyreOperation's binaries individually — each binary gets its own CopyToDevice and its own device_handle stored on the SpyreOperation.

Constructors:

| Constructor | Description |
|-------------|-------------|
| `ExecutionPlan()` | Empty plan, operations added manually |
| `ExecutionPlan(compiler_output)` | Auto-populated from compiler output |

Properties:

| Property | Description |
|----------|-------------|
| `tensor_input_metadata` | Expected input/output tensor metadata from the compiler |
| `operations` | Ordered list of SpyreOperations — each SpyreOperation carries its own device handles after loading |
| `correction_metadata` | Program correction metadata (including the byte size of the correction input buffer per SpyreOperation) |
| `init_packets` | Init packets from the compiler |

ExecutionPlans do not own execution logic. They are passive data structures whose SpyreOperations have their binaries individually loaded to device memory (one DMA per binary) and are then submitted through SpyreStream for execution.

#### SpyreStream

The high-level stream abstraction in torch-spyre. A SpyreStream internally holds a `FlexStream` and serves as the single entry point for all device work — both data movement and compute. It translates high-level SpyreOperations into the correct sequence of primitive FlexStream calls. This is the layer where torch-spyre's understanding of SpyreOperations, program correction, data transfers, and tiling is converted into the flat copy/launch primitives that flex understands.

**Semantics:**
- SpyreStream is the torch-spyre-facing API; FlexStream is the flex-facing API
- All methods are **asynchronous** — control returns to the host immediately
- Within a SpyreStream, operations execute **sequentially** (inherited from the underlying FlexStream's FIFO ordering)
- SpyreStream understands SpyreOperations and their internal structure (compute, preprocessing, program correction); FlexStream does not

Methods:

| Method | Description |
|--------|-------------|
| `Launch(SpyreOperation, List<device_handle>)` | Decompose a SpyreOperation into FlexStream primitives and enqueue them, given the resident tensor device handles |
| `LaunchTiled(SpyreOperation, List<device_handle>)` | Like Launch, but handles tensors larger than the compiled tile size by enqueuing multiple iterations |
| `Synchronize()` | Block the host until all previously enqueued operations have completed (delegates to FlexStream.Synchronize) |

**Launch(SpyreOperation, List\<device_handle\>)**

Takes a SpyreOperation and the device handles for resident input/output tensors. SpyreStream walks the SpyreOperation's PreProcessCompute list, mapping each entry to the appropriate FlexStream primitive (CopyToDevice, CopyFromDevice, or Launch), then enqueues a Launch for the SpyreCompute if present.

**Example 1 — Tensor load (preprocessing only, no compute):**

A SpyreOperation with a single CopyToDevice PreProcessCompute and no SpyreCompute:

1. **CopyToDevice** — Copy tensor data from host to device

Single FlexStream call. No Launch follows because there is no SpyreCompute.

**Example 2 — Matmul with program correction (preprocessing + compute):**

A SpyreOperation with PreProcessCompute entries (CopyToDevice for correction inputs, Launch for program correction) and a SpyreCompute (matmul):

1. **CopyToDevice** — Copy the offset inputs (tensor device addresses used for program correction) from host to the correction program's input location on device
2. **Launch** (program correction) — Execute the program correction compute, which patches the matmul program with the correct tensor addresses
3. **Launch** (matmul compute) — Execute the corrected matmul program

Each of these is enqueued on the underlying FlexStream in order. Because FlexStream is FIFO, the correction completes before the matmul begins, and the matmul begins only after its program has been corrected.

**LaunchTiled(SpyreOperation, List\<device_handle\>)**

Handles the case where tensor shapes exceed the compiled tile size. SpyreStream compares each tensor's shape against the kernel's expected input shapes (from the SpyreOperation's SpyreCompute metadata), infers the tiling dimension(s) and iteration count, and enqueues multiple rounds of FlexStream operations — one full Launch decomposition per tile iteration, each with updated tensor offsets.

**Synchronize()**

Delegates to `FlexStream.Synchronize()`. Blocks the calling host thread until all operations previously enqueued on this stream have completed.

#### FlexStream

The low-level execution primitive in flex, modeled after CUDA streams. A FlexStream provides an asynchronous, ordered queue of FlexOperations to be executed on the Spyre device. FlexStream has no knowledge of SpyreOperations, program correction, or tiling — it only understands raw copies and program launches by device handle. SpyreStream is the layer that translates higher-level SpyreOperations into FlexStream calls.

Each FlexStream method produces a **FlexOperation** — a single primitive (copy or launch) that maps directly to a control block (CB) for hardware submission. A FlexOperation is the unit of work that the Scheduler accepts and drains onto hardware.

**Semantics:**
- All methods are **asynchronous** — control returns to the host immediately after enqueuing the operation
- Within a single FlexStream, operations execute **sequentially** in FIFO order
- Across different FlexStreams, operations have **no ordering guarantees**
- Each FlexStream is identified by a `stream_index`

Methods:

| Method | Description |
|--------|-------------|
| `CopyToDevice(host_address, device_handle, size)` | Enqueue a host-to-device copy of `size` bytes from `host_address` to `device_handle` |
| `CopyFromDevice(host_address, device_handle, size)` | Enqueue a device-to-host copy of `size` bytes from `device_handle` to `host_address` |
| `Launch(device_handle)` | Enqueue execution of the binary identified by `device_handle` |
| `Synchronize()` | Block the host until all previously enqueued operations on this stream have completed |

**CopyToDevice(host_address, device_handle, size)**

Enqueues a host-to-device data transfer. The `host_address` is a pointer to source data in host memory. The `device_handle` is a `DeviceHandle` identifying the target location on device. `size` specifies the number of bytes to transfer.

**CopyFromDevice(host_address, device_handle, size)**

Enqueues a device-to-host data transfer. The `device_handle` is a `DeviceHandle` identifying the source location on device. The `host_address` is a pointer to the destination in host memory. `size` specifies the number of bytes to transfer.

**Launch(device_handle)**

Enqueues execution of a binary that has been previously loaded to device. The `device_handle` references the binary's location on device (e.g., a program correction binary or a compute binary from a SpyreOperation). For tiled execution, multiple Launch FlexOperations are enqueued for the same binary, each with updated tensor offsets.

**Synchronize()**

Blocks the calling host thread until all operations previously enqueued on this stream have completed. This is the only blocking method on FlexStream.

#### Scheduler

The Scheduler sits underneath FlexStream and is responsible for serializing FlexOperations for execution on hardware. Since the current Spyre hardware does not support concurrent execution of multiple streams, the Scheduler ensures all FlexOperations are dispatched in a valid order.

Methods:

| Method | Description |
|--------|-------------|
| `QueueOperation(stream_index, flex_op)` | Accept a FlexOperation from the given stream and schedule it for execution on hardware |

**Behavior:**
- Each FlexStream method constructs a FlexOperation (which maps to a control block) and calls `Scheduler.QueueOperation(stream_index, flex_op)` to submit work
- The Scheduler maintains per-stream FIFO queues and drains them onto the hardware
- Within a stream, FlexOperations are strictly ordered — a FlexOperation does not begin until the previous one completes
- Across streams, the Scheduler is free to interleave FlexOperations in any order, though with current hardware constraints it serializes all FlexOperations into a single execution sequence
- The Scheduler abstracts the hardware's execution constraints, allowing the FlexStream API to remain unchanged as hardware evolves to support true concurrent stream execution

### Tiled Execution

Tiled execution is owned by `SpyreStream.LaunchTiled`. When a tensor is larger than the compiled tile size, SpyreStream reuses the compiled kernel across the full tensor by enqueuing multiple rounds of FlexOperations — one full SpyreOperation decomposition per iteration, each with updated tensor offsets.

**Preconditions:**
- Each tensor dimension is greater than or equal to the corresponding compiled tile dimension
- Each tensor dimension is evenly divisible by the corresponding tile dimension
- Tensor strides are consistent with the tiling dimension

**Behavior:**
1. Compare each tensor's shape against the kernel's expected input shapes (from the ExecutionPlan's per-SpyreCompute metadata)
2. Infer the tiling dimension(s) and compute `num_iterations = tensor_dim / tile_dim`
3. For each iteration `i`:
   a. Compute per-tensor offset: `offset_i = base_vf_offset + (i * tensor_stride_along_tiled_dim * tile_size * element_size)`
   b. Decompose the SpyreOperation into FlexOperations with updated device addresses (CopyToDevice for correction inputs, Launch for correction, Launch for compute)
   c. All operations enqueued asynchronously on the underlying FlexStream — sequential within the stream

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

**LaunchKernel(SpyreStream, ExecutionPlan, List\<SpyreTensor\>)** — The entry point in torch-spyre. For each SpyreOperation in the ExecutionPlan, determines whether the tensors match the compiled tile size exactly or require tiling, then delegates to either `SpyreStream.Launch` or `SpyreStream.LaunchTiled` accordingly. Control returns to the host immediately; use `SpyreStream.Synchronize()` to wait for completion.

### Workflows

#### Workflow 1: Tensor Allocation and Transfer

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌────────────┐
│  CPUTensor   │────▶│ SpyreAllocator  │────▶│   SpyreStream    │────▶│ FlexStream │
│  (host)      │     │ allocate block  │     │ Launch(SpyreOp   │     │ CopyTo     │
│              │     │ → DeviceHandle  │     │  [CopyToDevice]) │     │ Device()   │
│              │     │      │         │                  │     │ (async)    │
└─────────────┘     └─────────────────┘     └──────────────────┘     └────────────┘
```

1. User creates a `CPUTensor` and calls `.to(device)`
2. SpyreAllocator allocates a block, producing a `DeviceHandle` (PFDeviceHandle or VFDeviceHandle depending on mode)
3. A preprocessing-only SpyreOperation is created with a CopyToDevice PreProcessCompute (host_address, device_handle, size)
4. `SpyreStream.Launch(spyre_op, [])` decomposes into `FlexStream.CopyToDevice(...)` — returns immediately
5. Result is a `SpyreTensor` carrying the device address metadata
6. Host may continue work; data transfer proceeds asynchronously on the stream

#### Workflow 2: Compilation and Loading

```
┌──────────┐     ┌───────────┐     ┌───────────────┐     ┌──────────────────┐     ┌────────────┐
│ Inductor │────▶│ Deeptools │────▶│ ExecutionPlan │────▶│   SpyreStream    │────▶│ FlexStream │
│ (sdsc)   │     │ (compile) │     │               │     │ Launch(SpyreOp   │     │ CopyTo     │
└──────────┘     └───────────┘     └───────────────┘     │  [CopyToDevice]) │     │ Device()   │
                                                          │ × N (per binary) │     │ × N        │
                                                          └──────────────────┘     └────────────┘
```

1. `torch.compile` triggers the inductor frontend, producing sdsc inputs for deeptools
2. Deeptools (backend compiler) produces an `ExecutionPlan` — an ordered list of SpyreOperations, each containing `SpyreCompute` and `PreProcessCompute` steps
3. For each SpyreOperation in the ExecutionPlan, for each binary that needs to be on device (e.g., program correction binary, compute binary):
   a. SpyreAllocator allocates space for the binary → `device_handle`
   b. A preprocessing-only SpyreOperation is created with a CopyToDevice PreProcessCompute
   c. `SpyreStream.Launch(spyre_op, [])` decomposes into `FlexStream.CopyToDevice(...)` — `device_handle` is stored on the SpyreOperation
4. ExecutionPlan (with all SpyreOperation device handles populated) is cached for reuse

#### Workflow 3: Detailed Execution — LaunchKernel to Hardware

This diagram shows the full path from `LaunchKernel` through every layer for a matmul with program correction, illustrating how SpyreStream decomposes a single SpyreOperation into FlexOperations that the Scheduler drains onto hardware.

```
 torch-spyre                          flex
┌─────────────────────────────────┐  ┌──────────────────────────────────────────────────┐
│                                 │  │                                                  │
│  LaunchKernel(spyre_stream,     │  │                                                  │
│        exec_plan, tensors)      │  │                                                  │
│         │                       │  │                                                  │
│         ▼                       │  │                                                  │
│  ┌─────────────┐                │  │                                                  │
│  │  Validate   │                │  │                                                  │
│  │  shapes vs  │                │  │                                                  │
│  │  SpyreComp  │                │  │                                                  │
│  │  expected   │                │  │                                                  │
│  └──────┬──────┘                │  │                                                  │
│         │                       │  │                                                  │
│         ▼                       │  │                                                  │
│  ┌──────────────┐               │  │                                                  │
│  │ shapes match │──── yes ──────┼──┼─▶ SpyreStream.Launch(spyre_op, tensor_handles) │
│  │ tile size?   │               │  │                         │                        │
│  └──────┬───────┘               │  │                         │                        │
│         │ no                    │  │                         │                        │
│         ▼                       │  │                         │                        │
│  SpyreStream.LaunchTiled(       │  │                         │                        │
│    spyre_op, tensor_handles)    │  │                         │                        │
│         │                       │  │                         │                        │
│  (see Workflow 4 for tiling)    │  │                         │                        │
│                                 │  │                         │                        │
└─────────────────────────────────┘  │                         │                        │
                                     │    ┌────────────────────┘                        │
                                     │    │  SpyreStream decomposes SpyreOperation:      │
                                     │    │                                             │
                                     │    │  SpyreOperation contains:                        │
                                     │    │    SpyreCompute (matmul)                    │
                                     │    │    PreProcessCompute (program correction)   │
                                     │    │    tensor device handles: [A_hdl, B_hdl]    │
                                     │    │                                             │
                                     │    │  Decomposed into FlexOperations:            │
                                     │    ▼                                             │
                                     │  ┌───────────────────────────────────────────┐   │
                                     │  │            FlexStream Queue               │   │
                                     │  │  (FIFO — sequential within stream)        │   │
                                     │  │                                           │   │
                                     │  │  ┌─────────────────────────────────────┐  │   │
                                     │  │  │ 1. CopyToDevice(                    │  │   │
                                     │  │  │      correction_offsets_host,       │  │   │
                                     │  │  │      correction_input_device_hdl,   │  │   │
                                     │  │  │      size)                          │  │   │
                                     │  │  │    Copy tensor offset inputs to     │  │   │
                                     │  │  │    device for program correction    │  │   │
                                     │  │  ├─────────────────────────────────────┤  │   │
                                     │  │  │ 2. Launch(                          │  │   │
                                     │  │  │      program_correction_device_hdl) │  │   │
                                     │  │  │    Execute program correction —     │  │   │
                                     │  │  │    patches matmul program with      │  │   │
                                     │  │  │    correct tensor addresses         │  │   │
                                     │  │  ├─────────────────────────────────────┤  │   │
                                     │  │  │ 3. Launch(                          │  │   │
                                     │  │  │      matmul_compute_device_hdl)     │  │   │
                                     │  │  │    Execute the corrected matmul     │  │   │
                                     │  │  │    A_hdl × B_hdl → C_hdl            │  │   │
                                     │  │  └─────────────────────────────────────┘  │   │
                                     │  └──────────────────┬────────────────────────┘   │
                                     │                     │                            │
                                     │                     ▼                            │
                                     │  ┌──────────────────────────────────────────┐    │
                                     │  │              Scheduler                   │    │
                                     │  │                                          │    │
                                     │  │  QueueOperation(stream_idx, copy_cb)     │    │
                                     │  │  QueueOperation(stream_idx, corr_cb)     │    │
                                     │  │  QueueOperation(stream_idx, matmul_cb)   │    │
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
2. Tensor shapes are validated against the ExecutionPlan's expected input shapes
3. Shapes match exactly → delegates to `SpyreStream.Launch(spyre_op, tensor_device_handles)`
4. SpyreStream inspects the SpyreOperation's internal structure:
   - Identifies the PreProcessCompute (program correction) and SpyreCompute (matmul)
   - Prepares correction offset inputs (tensor device addresses that the correction program needs)
5. SpyreStream enqueues onto FlexStream in order:
   - `CopyToDevice(correction_offsets_host, correction_input_device_hdl, size)` — upload offset inputs
   - `Launch(program_correction_device_hdl)` — run program correction
   - `Launch(matmul_compute_device_hdl)` — run the corrected matmul
6. FlexStream constructs a FlexOperation (containing a control block) for each primitive and forwards it to `Scheduler.QueueOperation(stream_index, flex_op)`
7. Scheduler drains FlexOperations onto hardware in FIFO order
8. Host returns immediately; call `SpyreStream.Synchronize()` when results are needed

#### Workflow 4: Tiled Execution

```
 torch-spyre                                    flex
┌──────────────────────────────────┐  ┌──────────────────────────────────────────────┐
│                                  │  │                                              │
│  LaunchKernel detects tile       │  │                                              │
│  mismatch → SpyreStream         │  │                                              │
│     .LaunchTiled(spyre_op,       │  │                                              │
│       tensor_handles)            │  │                                              │
│         │                        │  │                                              │
│  Infer: 4 iterations (4096/1024) │  │                                              │
│         │                        │  │                                              │
│    ┌────┴────┐                   │  │          FlexStream Queue                    │
│    │ iter 0  │ ──────────────────┼──┼─▶ CopyToDevice(offsets_0) → Launch(corr)    │
│    │         │                   │  │   → Launch(matmul)                           │
│    ├─────────┤                   │  │                                              │
│    │ iter 1  │ ──────────────────┼──┼─▶ CopyToDevice(offsets_1) → Launch(corr)    │
│    │         │                   │  │   → Launch(matmul)                           │
│    ├─────────┤                   │  │                                              │
│    │ iter 2  │ ──────────────────┼──┼─▶ CopyToDevice(offsets_2) → Launch(corr)    │
│    │         │                   │  │   → Launch(matmul)                           │
│    ├─────────┤                   │  │                                              │
│    │ iter 3  │ ──────────────────┼──┼─▶ CopyToDevice(offsets_3) → Launch(corr)    │
│    │         │                   │  │   → Launch(matmul)                           │
│    └─────────┘                   │  │                                              │
│                                  │  │   (12 ops total, all async, FIFO in stream)  │
└──────────────────────────────────┘  │                     │                        │
                                      │                     ▼                        │
                                      │  ┌──────────────────────────────────────┐    │
                                      │  │            Scheduler                 │    │
                                      │  │  Serializes all 12 ops onto HW:     │    │
                                      │  │  Copy₀→Corr₀→Mat₀→Copy₁→Corr₁→Mat₁│    │
                                      │  │  →Copy₂→Corr₂→Mat₂→Copy₃→Corr₃→Mat₃│   │
                                      │  └──────────────────────────────────────┘    │
                                      └──────────────────────────────────────────────┘
```

1. `LaunchKernel(spyre_stream, exec_plan, tensors)` is called from torch-spyre
2. Tensor shapes are compared against the ExecutionPlan's expected input shapes
3. Mismatch detected → delegates to `SpyreStream.LaunchTiled(spyre_op, tensor_device_handles)`
4. SpyreStream infers tiling dimension and `num_iterations = 4096 / 1024 = 4`
5. For each iteration `i`:
   - Computes updated tensor offsets for iteration `i`
   - Decomposes the SpyreOperation into FlexOperations with updated addresses:
     - `CopyToDevice(correction_offsets_i)` — upload corrected offset inputs for this iteration
     - `Launch(program_correction)` — patch the program with this iteration's addresses
     - `Launch(matmul_compute)` — execute the matmul for this tile slice
6. All 12 operations (3 per iteration × 4 iterations) are enqueued asynchronously on FlexStream
7. Scheduler drains all operations onto hardware in FIFO order
8. Host returns immediately; call `SpyreStream.Synchronize()` to block until all iterations complete

#### Workflow 5: End-to-End

```
                                                    # Default SpyreStream created at runtime start

1. tensor_a = torch.randn(4096, 1024)             # Create on CPU
2. spyre_a = tensor_a.to("spyre")                 # SpyreOperation([CopyToDevice]) → FlexStream.CopyToDevice
3. spyre_b = torch.randn(1024, 1024).to("spyre")  # Same — async, all through default SpyreStream

4. compiled_fn = torch.compile(matmul_fn)          # Lazy — no compilation yet

5. result = compiled_fn(spyre_a, spyre_b)          # First call triggers inductor → deeptools
                                                    # ExecutionPlan created, binaries loaded to device
                                                    # Cached via PyTorch compile caching
                                                    # Then: LaunchKernel(exec_plan, tensors)
                                                    # Uses default stream (or creates a new one)
                                                    # Detects tensor_a [4096,1024] > tile [1024,1024]
                                                    # SpyreStream.LaunchTiled enqueues 4 iterations
                                                    # (12 FlexStream ops: 3 per iteration)

6. result_cpu = result.to("cpu")                   # SpyreOperation([CopyFromDevice]) → FlexStream.CopyFromDevice
```

#### Workflow 6: Multi-Stream (Future Hardware)

```
stream_a = SpyreStream()                           # Stream for layer 1
stream_b = SpyreStream()                           # Stream for layer 2

# Enqueue independent work on separate streams
LaunchKernel(stream_a, exec_plan_1, tensors_1)     # Async — SpyreStream decomposes into FlexStream ops
LaunchKernel(stream_b, exec_plan_2, tensors_2)     # Async — no ordering w.r.t. stream_a

# With current hardware: Scheduler serializes both streams' FlexStream ops
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

The key difference is where work submission lives. A CUDA stream is a **passive handle** — you pass it as a parameter to external API calls like `cudaMemcpyAsync(dst, src, size, kind, stream)` or kernel launches via `<<<grid, block, sharedMem, stream>>>`. FlexStream is an **active object** — copy and launch are methods on the stream itself (`stream.CopyToDevice(...)`, `stream.Launch(...)`).

The following table maps CUDA stream capabilities to their FlexStream equivalents:

| Capability | CUDA Stream | FlexStream |
|------------|-------------|------------|
| Copy | `cudaMemcpyAsync(dst, src, size, kind, stream)` — direction via enum | Separate `CopyToDevice` / `CopyFromDevice` — direction explicit in method name |
| Launch | `kernel<<<grid, block, sharedMem, stream>>>()` — takes grid/block dims, function pointer, args | `Launch(device_handle)` — just a handle to a pre-loaded binary (no grid/block dims since Spyre is not SIMT) |
| Synchronize | `cudaStreamSynchronize(stream)` | `Synchronize()` — identical semantics |
| Events | `cudaEventRecord` / `cudaStreamWaitEvent` for inter-stream dependencies | Not present (see unresolved question #4) |
| Query | `cudaStreamQuery()` — non-blocking completion poll | Not present (see unresolved question #9) |
| Priorities | `cudaStreamCreateWithPriority()` | Not present (see unresolved question #10) |
| Host callbacks | `cudaLaunchHostFunc()` — enqueue a host-side function on the stream | Not present (see unresolved question #11) |
| Default stream | NULL stream with legacy blocking semantics | Not yet specified (see unresolved question #7) |

FlexStream is a deliberately minimal subset — 4 methods vs. CUDA's dozens. The omitted features (events, query, priorities, host callbacks) are captured as unresolved questions for future consideration as hardware and runtime requirements evolve.

### SpyreStream — No Direct CUDA Equivalent

CUDA has no built-in runtime layer analogous to SpyreStream. In CUDA, a kernel launch is self-contained — the user or library (cuBLAS, cuDNN) manually issues the correct sequence of memcpy and launch calls on a stream. SpyreStream exists because Spyre operations have inherent **compound structure**:

- **A CUDA matmul**: 1 kernel launch
- **A Spyre matmul**: DMA correction metadata → run correction binary → run compute binary (3 FlexOperations)

SpyreStream encapsulates this decomposition so callers never touch FlexStream directly. The closest CUDA analogs are vendor libraries like cuBLAS and cuDNN, which internally issue multi-step sequences on a user-provided stream — but those are external libraries, not a core runtime layer. SpyreStream makes this pattern a first-class part of the runtime because every Spyre compute requires it, not just optimized library calls.

The other unique SpyreStream capability — `LaunchTiled` — also has no CUDA equivalent. CUDA kernels accept arbitrary dimensions as launch parameters (`<<<grid, block>>>`), so the user simply changes grid dims to cover larger tensors. Spyre kernels are compiled for fixed tile shapes, so the runtime must iterate over tiles, enqueuing multiple rounds of FlexOperations with updated tensor offsets.

## **How we teach this**

TBD

## **Unresolved questions**

1. **Reduction tiling**: When tiling along the reduction dimension (K in matmul), partial results must be accumulated. Should the runtime handle accumulation internally, or should this be expressed as a separate operation in the ExecutionPlan?

2. **Multi-dimensional tiling**: If both M and N exceed the tile size, the runtime needs a nested loop. Should tiled execution support arbitrary nesting, or should this be constrained to single-dimension tiling with multi-dim handled by the compiler?

3. **Async iteration overlap**: Currently each tiled iteration within a stream is sequential. Could we use separate streams or double-buffering to overlap iteration N+1's data movement with iteration N's compute?

4. **Stream events / synchronization primitives**: Should FlexStream support inter-stream synchronization (e.g., CUDA-style events) to express dependencies between streams without full synchronization?

5. **Remainder handling**: What happens when the tensor dimension is not evenly divisible by the tile size (e.g., tensor is 4000 with tile 1024)? Options: fail, pad, or compile a separate remainder kernel.

6. **Scheduler policies**: When draining multiple streams onto serialized hardware, what scheduling policy should be used? Options: round-robin, priority-based, drain-one-first, etc.

7. **Default stream**: Should there be a default FlexStream (like CUDA's default stream) that is used when the user does not explicitly create one?

8. **Program correction across PF/VF modes**: The correction input buffer contains tensor locations — offsets in VF mode, addresses in PF mode. How should SpyreStream assemble this buffer uniformly from DeviceHandles when the correction program expects mode-specific values?

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
