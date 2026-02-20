# Program Execution Pipeline

**Authors:**
* @JRosenkranz

## **Summary**

This RFC introduces a layered stream-based execution model for torch-spyre. `SpyreStream` (in torch-spyre) implements the PyTorch Stream interface and serves as a thin passthrough that translates PyTorch-native data into the types `FlexStream` (in flex) understands. `FlexStream` is the core execution engine — modeled after CUDA streams, it accepts Jobs (representing both compute and data transfers), decomposes them into the correct sequence of low-level steps (copies, launches), and forwards those steps to the Scheduler for hardware dispatch. FlexStream's single `Launch` method automatically detects when tensor shapes exceed the compiled tile size and handles tiled execution transparently — no separate tiling API is needed. All FlexStream methods are asynchronous, returning control to the host immediately, with jobs executing sequentially within a stream. A `Scheduler` layer underneath dispatches work onto hardware, respecting the intra-stream FIFO ordering guarantee, while the stream abstraction preserves forward compatibility with future concurrent execution support.

## **Motivation**

The current connection between torch-spyre and the flex runtime relies on graph execution, which carries significant overhead for what is fundamentally a kernel launch job. The graph-based approach requires constructing, optimizing, and traversing a graph structure even for single-kernel execution, adding complexity and latency that is unnecessary for the common case.

This RFC replaces that approach with a layered stream architecture:

* **SpyreStream** (torch-spyre) implements the PyTorch Stream interface and serves as a thin passthrough layer. It translates PyTorch-native data (torch tensors, device metadata) into the types FlexStream understands (Jobs, AllocationIndices) and delegates all execution logic to FlexStream.

* **FlexStream** (flex) is the core execution engine, modeled after CUDA streams. It accepts Jobs — each containing a JobPlan of ordered steps (HostOperation, DMA, DeviceCompute) — and walks the plan to produce the correct sequence of low-level control blocks forwarded to the Scheduler. FlexStream understands that a single logical job like a matmul may require multiple steps (a host operation to convert tensor virtual address metadata, a DMA of the correction tensor to device, and the device compute launch) and owns this decomposition. FlexStream's `Launch` method automatically detects when tensor shapes exceed the compiled tile size and handles tiled execution transparently — reusing compilation artifacts by enqueuing multiple rounds of low-level steps with updated tensor offsets — without requiring a separate API call. All methods are asynchronous — jobs enqueued on a stream return immediately. Within a single stream, jobs execute in FIFO order; across streams, no ordering is guaranteed.

* **Scheduler** (flex) sits underneath FlexStream and dispatches low-level steps onto hardware, respecting the intra-stream FIFO ordering guarantee. The Scheduler's internal design is covered in a separate Scheduler RFC.

ExecutionPlans remain lightweight containers of Jobs built from compiler output. They do not own execution logic — instead, each Job's binary is loaded to device (via a DMA), and then the Jobs are submitted through SpyreStream → FlexStream, which walks each Job's plan to produce low-level control blocks for the Scheduler.

As a secondary benefit, FlexStream's `Launch` automatically detects when tensors exceed the compiled tile size and transparently reuses the kernel by enqueuing multiple rounds of low-level steps with updated offsets — no recompilation or separate API call required.

The introduction of the SpyreAllocator's VF mode also shapes this design: tensors are now carved from pre-allocated memory regions with block-level offsets rather than independently allocated. torch-spyre receives only opaque `AllocationIndex` handles from FlexAllocator; the execution pipeline resolves these to `VirtualAddress` values internally within flex.

## **Proposed Implementation**

### Core Components

#### AllocationIndex

An opaque integer handle returned by `FlexAllocator.allocate()` to identify an allocation. This is the only allocator type that crosses the flex → torch-spyre boundary — torch-spyre stores it in the `at::DataPtr` and passes it back to flex for deallocation and execution. torch-spyre never inspects or interprets its value. FlexStream resolves `AllocationIndex` values to `VirtualAddress` values internally.

#### VirtualAddress

A flex-internal reference to a location in device memory. VirtualAddress is used within flex by FlexStream, the Scheduler, and the ControlBlock segment table — it never crosses the flex → torch-spyre boundary. A VirtualAddress identifies only a location — size is stored separately alongside it.

Properties:

| Property | Description |
|----------|-------------|
| `region_id` | Identifies the memory region. In VF mode, this is an index into a firmware lookup table that maps to the physical address of the region. In PF mode, this is the physical address of the region itself. |
| `offset` | Byte offset of the allocation within the region. Always 128-byte aligned. |

The same structure is used in both PF and VF modes — only the interpretation of `region_id` differs. All flex-internal components (FlexStream, Scheduler) work uniformly with VirtualAddress regardless of mode.

#### SpyreAllocator / FlexAllocator

Manages device memory in two layers: `SpyreAllocator` (torch-spyre) is a thin wrapper implementing PyTorch's `at::Allocator` interface; `FlexAllocator` (flex) is the core memory manager. FlexAllocator manages a pool of up to 8 memory regions in both PF and VF modes, carving individual allocations from them as 128-byte-aligned blocks.

`FlexAllocator.allocate()` returns an opaque `AllocationIndex` to torch-spyre. Internally, FlexAllocator maintains the mapping from each `AllocationIndex` to its `VirtualAddress` (`region_id` + `offset`). `FlexAllocator.resolve()` returns the `VirtualAddress` for a given `AllocationIndex` — used by FlexStream when assembling correction buffers, computing tiled offsets, or dispatching to hardware.

See the SpyreAllocator RFC for full details on allocation strategies, memory region management, and memory lifecycle.

#### SpyreTensor

A tensor residing on-device. Carries metadata required for execution:

* **shape**: Logical dimensions (e.g., `[4096, 1024]`)
* **stride**: Memory stride per dimension (bytes between consecutive elements along each axis)
* **data_type**: Element type (e.g., float32, uint32)
* **allocation_index**: An `AllocationIndex` identifying the tensor's allocation on device (from FlexAllocator)
* **size_bytes**: Total byte size of the tensor data
* **layout**: A `SpyreTensorLayout` describing the tensor's tiled layout on device — includes `device_size` (tiled dimensions on device), `device_dtype` (on-device data type), and `dim_map` (mapping between logical shape and device dimensions)

#### Job

A Job encapsulates a unit of work to be submitted through FlexStream. It bundles everything needed to execute a single kernel: the compiled binary, correction metadata, and a JobPlan describing the steps to run.

Properties:

| Property | Description |
|----------|-------------|
| `binary_path` | Path to the compiled binary produced by the backend compiler (deeptools). This single binary may internally contain both a program correction program and the compute program. Loaded to device during ExecutionPlan loading, at which point `allocation_index` is set. |
| `allocation_index` | `AllocationIndex` identifying the binary's allocation on device (set during ExecutionPlan loading). FlexStream resolves this to a `VirtualAddress` internally when constructing ControlBlocks. |
| `program_correction_metadata` | Metadata used alongside the HostOperation step to produce the correction tensor (e.g., the vdci.json equivalent). May be empty if the kernel has no symbolic inputs. |
| `job_plan` | A `JobPlan` — the ordered sequence of steps to execute this job |

Constructors:

| Constructor | Description |
|-------------|-------------|
| `Job(binary_path, program_correction_metadata, JobPlan)` | Job with a binary, correction metadata, and a plan of steps to execute |
| `Job(JobPlan)` | DMA-only job with no binary or correction metadata. Used for pure data movement (e.g., tensor `.to(device)`, program binary loading, tensor `.to("cpu")`). |

A Job should be self-contained: if a compute requires program correction, the host operation (conversion of tensor virtual address metadata), the DMA of the correction tensor, and the device compute must all be steps in the same Job's plan. For pure data movement (e.g., tensor `.to(device)` or program loading), a Job with only DMA steps in its plan is used.

#### JobPlan

An ordered sequence of steps that describes how to execute a Job. When FlexStream processes a Job, it walks the JobPlan and maps each step to the appropriate low-level action — HostOperations run on the host CPU, DMAs map to DMA control blocks, and DeviceCompute maps to a compute control block forwarded to the Scheduler.

Steps:

| Step | Properties | Description |
|------|------------|-------------|
| `HostOperation` | `function` | A host-side computation that runs on the CPU. Used for conversion of tensor virtual address metadata — a function that takes resolved symbol values (tensor virtual addresses, shape values) and the Job's `program_correction_metadata`, and produces a correction tensor to be DMA'd to device. |
| `DMA` | `host_address`, `allocation_index`, `size`, `direction` | Data transfer between host and device. `direction` is either `ToDevice` or `FromDevice`. Used for copying tensor data, program binaries, or correction tensors to device, and for reading results back. Maps to a DMA control block. `size` is the byte count to transfer. FlexStream resolves `allocation_index` to a `VirtualAddress` when constructing the DMA control block. |
| `DeviceCompute` | `allocation_index`, `expected_input_shapes` | Launch a compute binary on device. The binary — which may internally contain both a program correction program and the actual compute program — is executed as a single compute control block (CB). `allocation_index` identifies the binary's allocation on device (set during ExecutionPlan loading). FlexStream resolves it to a `VirtualAddress` when constructing the compute control block. `expected_input_shapes` are the compiled tensor shapes, used by FlexStream to detect tiling requirements. |

**Program correction flow**: When a kernel uses symbolic addresses or shapes, the backend compiler produces a unified binary containing both a correction program and the compute program, along with correction metadata. At runtime, the HostOperation step takes the resolved symbol values and the Job's `program_correction_metadata` as input and produces a correction tensor. The DMA step transfers this tensor to a reserved location on device (segment 7, address 0). The DeviceCompute step then launches the unified binary as a single compute CB — internally, the correction program reads the correction tensor to patch the compute program, then the compute program executes. The runtime does not distinguish between the correction program and the compute program; they are opaque within the single CB.

#### ExecutionPlan

Produced by the backend compiler (deeptools). An ExecutionPlan starts as compiler output describing the ordered sequence of Jobs required, then becomes the runtime artifact once each Job's binary is loaded to device. Loading an ExecutionPlan means loading each Job's binary via DMA and storing the resulting `allocation_index` on the Job. Each backend compiler input (sdsc) produced by inductor maps to a single Job. A single `torch.compile` call may produce multiple sdscs, which is why an ExecutionPlan contains an ordered list of Jobs. In many cases, however, the ExecutionPlan will contain only a single Job (one SDSC per compile).

Constructors:

| Constructor | Description |
|-------------|-------------|
| `ExecutionPlan()` | Empty plan, jobs added manually |
| `ExecutionPlan(compiler_output)` | Auto-populated from compiler output |

Properties:

| Property | Description |
|----------|-------------|
| `jobs` | Ordered list of Jobs — each Job carries its own `allocation_index` after loading |

ExecutionPlans do not own execution logic. They are passive data structures whose Jobs each have their binary loaded to device memory (one DMA per Job) and are then submitted through SpyreStream → FlexStream for execution.

#### SpyreStream

A thin wrapper around `FlexStream` that implements the PyTorch Stream interface. SpyreStream is the torch-spyre-facing API — it translates PyTorch-native data (torch tensors, device metadata) into the types FlexStream understands (Jobs, AllocationIndices) and delegates all execution logic to the underlying FlexStream.

SpyreStream contains no decomposition logic, tiling logic, or understanding of Job internals. It is a passthrough layer whose purpose is to bridge the PyTorch runtime conventions with the flex execution engine.

**Semantics:**
* Implements the PyTorch Stream interface so that torch-spyre integrates with PyTorch's stream management (e.g., `torch.spyre.Stream`, `torch.spyre.current_stream()`)
* All methods are **asynchronous** — control returns to the host immediately (inherited from FlexStream)
* Within a SpyreStream, jobs execute **sequentially** (inherited from FlexStream's FIFO ordering)

Methods:

| Method | Description |
|--------|-------------|
| `Launch(Job, List<SpyreTensor>, allow_tiled_launch=true)` | Extract AllocationIndices from SpyreTensors and delegate to `FlexStream.Launch(job, allocation_indices, allow_tiled_launch)` |
| `Synchronize()` | Delegate to `FlexStream.Synchronize()` |

#### FlexStream

The core execution engine in flex, modeled after CUDA streams. A FlexStream accepts Jobs — each containing a JobPlan of ordered steps — and walks the plan to produce the correct sequence of low-level control blocks (DMA CBs, compute CBs) forwarded to the Scheduler for hardware dispatch. HostOperation steps in the plan run on the host CPU before the device-bound control blocks are enqueued.

FlexStream understands the internal structure of Jobs: it knows that a matmul may require a host operation (conversion of tensor virtual address metadata), a DMA of the correction tensor to device, and a device compute launch, and it generates the correct sequence of control blocks for each. FlexStream's `Launch` method also automatically detects when tensor shapes exceed the compiled tile size and handles tiled execution transparently — enqueuing multiple rounds of low-level steps with updated tensor offsets — without requiring a separate API call.

**Semantics:**
* All methods are **asynchronous** — control returns to the host immediately after enqueuing
* Within a single FlexStream, jobs execute **sequentially** in FIFO order
* Across different FlexStreams, jobs have **no ordering guarantees**
* Each FlexStream is identified by a `stream_index`

Methods:

| Method | Description |
|--------|-------------|
| `Launch(Job, List<AllocationIndex>, allow_tiled_launch=true)` | Resolve AllocationIndices to VirtualAddresses via `FlexAllocator.resolve()`, then walk the Job's JobPlan, execute HostOperations on the host, and enqueue DMA/DeviceCompute steps as control blocks. If tensor shapes exceed the compiled tile size and `allow_tiled_launch` is true, automatically enqueues multiple tiled iterations. If shapes exceed the tile size and `allow_tiled_launch` is false, raises an exception. |
| `Synchronize()` | Block the host until all previously enqueued jobs on this stream have completed |

**Launch(Job, List\<AllocationIndex\>, allow_tiled_launch=true)**

Takes a Job, the AllocationIndices for resident input/output tensors, and an optional `allow_tiled_launch` flag (defaults to true). FlexStream resolves each `AllocationIndex` to a `VirtualAddress` via `FlexAllocator.resolve()` before processing the Job's plan.

FlexStream first compares each tensor's shape against the kernel's expected input shapes (from the DeviceCompute step in the Job's JobPlan). There are three cases:

1. **Shapes match exactly** — FlexStream walks the Job's JobPlan, executing HostOperations on the host CPU and mapping DMA/DeviceCompute steps to control blocks forwarded to the Scheduler.

2. **Shapes exceed tile size and `allow_tiled_launch` is true** — FlexStream infers the tiling dimension(s) and iteration count, then enqueues multiple rounds of control blocks — one full JobPlan walk per tile iteration, each with updated tensor offsets. See [Tiled Execution](#tiled-execution) for details.

3. **Shapes exceed tile size and `allow_tiled_launch` is false** — FlexStream raises an exception indicating that the tensor shapes do not match the compiled tile size and tiled launch is not permitted.

**Example 1 — Tensor load (DMA only):**

A Job whose JobPlan contains a single DMA step (ToDevice):

1. **DMA** (ToDevice) — Copy tensor data from host to device

Single DMA control block forwarded to the Scheduler. No compute CB follows.

**Example 2 — Matmul with program correction:**

A Job whose JobPlan contains a HostOperation, a DMA, and a DeviceCompute:

1. **HostOperation** — Run on host CPU: converts tensor virtual address metadata using the Job's `program_correction_metadata`, producing a correction tensor
2. **DMA** (ToDevice) — Transfer the correction tensor to device (segment 7, address 0)
3. **DeviceCompute** — Launch the unified binary as a single compute CB. Internally, the correction program reads the correction tensor, patches the compute program, then the matmul executes.

The HostOperation runs on the CPU first. The DMA and DeviceCompute are then enqueued as control blocks on the stream in FIFO order.

**Synchronize()**

Blocks the calling host thread until all jobs previously enqueued on this stream have completed. This is the only blocking method on FlexStream.

#### Scheduler

The Scheduler sits underneath FlexStream and is responsible for dispatching low-level steps onto hardware. From the perspective of this RFC, the Scheduler is an opaque component — FlexStream submits control blocks to it, and the Scheduler ensures they are executed respecting the intra-stream FIFO ordering guarantee. The Scheduler's internal design (scheduling policies, hardware serialization strategy, multi-stream interleaving) is covered in a separate Scheduler RFC.

### Tiled Execution

Tiled execution is handled automatically by `FlexStream.Launch` when `allow_tiled_launch` is true (the default). When a tensor is larger than the compiled tile size, FlexStream reuses the compiled kernel across the full tensor by enqueuing multiple rounds of low-level steps — one full Job decomposition per iteration, each with updated tensor offsets. If `allow_tiled_launch` is false and shapes exceed the tile size, Launch raises an exception.

**Preconditions:**
* Each tensor dimension is greater than or equal to the corresponding compiled tile dimension
* Each tensor dimension is evenly divisible by the corresponding tile dimension
* Tensor strides are consistent with the tiling dimension

**Behavior:**
1. Compare each tensor's shape against the kernel's expected input shapes (from the DeviceCompute step in the Job's JobPlan)
2. Infer the tiling dimension(s) and compute `num_iterations = tensor_dim / tile_dim`
3. For each iteration `i`:
   a. Compute per-tensor offset: `offset_i = base_offset + (i * tensor_stride_along_tiled_dim * tile_size * element_size)`
   b. Walk the Job's JobPlan with updated device addresses (HostOperation for correction, DMA for correction tensor, DeviceCompute)
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
HostOp(tensor_vaddr_metadata_i, correction_metadata) → DMA(correction_tensor_i) → DeviceCompute(binary)
```

Tensors whose shapes already match the tile (e.g., `B` above) have stride 1 — they are reused across iterations without offset changes.

### Front-End Interface

**LaunchKernel(SpyreStream, ExecutionPlan, List\<SpyreTensor\>)** — The entry point in torch-spyre. For each Job in the ExecutionPlan, delegates to `SpyreStream.Launch(job, tensors, allow_tiled_launch)`. The `allow_tiled_launch` value can be controlled by a user environment setting (e.g., `SPYRE_ALLOW_TILED_LAUNCH`), allowing users to disable automatic tiling for debugging or to enforce that tensor shapes exactly match the compiled tile size. SpyreStream extracts AllocationIndices from the SpyreTensors and passes through to `FlexStream.Launch`, which resolves them to VirtualAddresses internally and automatically detects whether tensor shapes match the compiled tile size or require tiled execution. Control returns to the host immediately; use `SpyreStream.Synchronize()` to wait for completion.

### Workflows

#### Workflow 1: Tensor Allocation and Transfer

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  CPUTensor   │────▶│ SpyreAllocator  │────▶│   SpyreStream    │────▶│    FlexStream    │
│  (host)      │     │ allocate block  │     │ (extract indices,│     │ Launch(Job       │
│              │     │ →AllocationIndex│     │  passthrough)    │     │  [DMA ToDevice]) │
│              │     │                 │     │                  │     │ walk plan→sched  │
└─────────────┘     └─────────────────┘     └──────────────────┘     └──────────────────┘
```

1. User creates a `CPUTensor` and calls `.to(device)`
2. FlexAllocator allocates a block, producing an `AllocationIndex` (opaque handle)
3. A Job is created with a single DMA (ToDevice) step in its JobPlan (host_address, allocation_index, size)
4. `SpyreStream.Launch(job, tensors)` extracts AllocationIndices from SpyreTensors, then delegates to `FlexStream.Launch(job, allocation_indices)`. FlexStream resolves each AllocationIndex to a VirtualAddress internally.
5. FlexStream walks the Job's JobPlan, produces a DMA control block, and forwards it to the Scheduler — returns immediately
6. Result is a `SpyreTensor` carrying the device address metadata
7. Host may continue work; data transfer proceeds asynchronously on the stream

#### Workflow 2: Compilation and Loading

```
┌──────────┐     ┌───────────┐     ┌───────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Inductor │────▶│ Deeptools │────▶│ ExecutionPlan │────▶│   SpyreStream    │────▶│    FlexStream    │
│ (sdsc)   │     │ (compile) │     │               │     │ (passthrough)    │     │ Launch(Job       │
└──────────┘     └───────────┘     └───────────────┘     │ × N (per Job)    │     │  [DMA ToDevice]) │
                                                          └──────────────────┘     │ × N              │
                                                                                   └──────────────────┘
```

1. `torch.compile` triggers the inductor frontend, producing sdsc inputs for deeptools
2. Deeptools (backend compiler) produces an `ExecutionPlan` — an ordered list of Jobs, each containing a `binary_path`, `program_correction_metadata`, and a `JobPlan`
3. For each Job in the ExecutionPlan:
   a. FlexAllocator allocates space for the Job's binary → `allocation_index`
   b. A loading Job is created with a single DMA (ToDevice) step in its JobPlan
   c. `SpyreStream.Launch(job, [])` passes through to `FlexStream.Launch(job, [])`, which resolves the `allocation_index` to a `VirtualAddress` and produces a DMA control block — `allocation_index` is stored on the Job
4. ExecutionPlan (with all Job allocation indices populated) is cached for reuse

#### Workflow 3: Detailed Execution — LaunchKernel to Hardware

This diagram shows the full path from `LaunchKernel` through every layer for a matmul with program correction, illustrating how FlexStream walks a Job's JobPlan to produce control blocks that the Scheduler drains onto hardware.

```
 torch-spyre                          flex
┌─────────────────────────────────┐  ┌──────────────────────────────────────────────────┐
│                                 │  │                                                  │
│  LaunchKernel(spyre_stream,     │  │                                                  │
│        exec_plan, tensors)      │  │                                                  │
│         │                       │  │                                                  │
│         ▼                       │  │                                                  │
│  SpyreStream.Launch(job, tensors│  │                                                  │
│    allow_tiled_launch)          │  │                                                  │
│         │                       │  │                                                  │
│  (extract AllocationIndices,    │  │                                                  │
│   delegate)                     │  │                                                  │
│         │                       │  │                                                  │
│         └───────────────────────┼──┼─▶ FlexStream.Launch(job, alloc_indices,          │
│                                 │  │     allow_tiled_launch)                          │
│                                 │  │         │                                        │
└─────────────────────────────────┘  │         │                                        │
                                     │    ┌────┘                                        │
                                     │    │  FlexStream compares tensor shapes           │
                                     │    │  against DeviceCompute.expected_input_shapes:│
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
                                     │    │  Job contains:                              │
                                     │    │    binary_vaddr (unified matmul binary)     │
                                     │    │    program_correction_metadata              │
                                     │    │    JobPlan:                                 │
                                     │    │      1. HostOperation (convert metadata)    │
                                     │    │      2. DMA (correction tensor → device)    │
                                     │    │      3. DeviceCompute (launch binary)       │
                                     │    │    tensor virtual addresses: [A, B]         │
                                     │    │                                             │
                                     │    │  Decomposed into low-level steps:           │
                                     │    ▼                                             │
                                     │  ┌───────────────────────────────────────────┐   │
                                     │  │            FlexStream                     │   │
                                     │  │  (FIFO — sequential within stream)        │   │
                                     │  │                                           │   │
                                     │  │  ┌─────────────────────────────────────┐  │   │
                                     │  │  │ 1. HostOperation (on CPU)           │  │   │
                                     │  │  │    Convert tensor vaddr metadata    │  │   │
                                     │  │  │    + correction_metadata →          │  │   │
                                     │  │  │    correction tensor                │  │   │
                                     │  │  ├─────────────────────────────────────┤  │   │
                                     │  │  │ 2. DMA(ToDevice)                    │  │   │
                                     │  │  │    correction_tensor →              │  │   │
                                     │  │  │    segment 7, address 0             │  │   │
                                     │  │  ├─────────────────────────────────────┤  │   │
                                     │  │  │ 3. DeviceCompute(binary_vaddr)      │  │   │
                                     │  │  │    Launch unified binary —          │  │   │
                                     │  │  │    correction + matmul A × B → C    │  │   │
                                     │  │  └─────────────────────────────────────┘  │   │
                                     │  └──────────────────┬────────────────────────┘   │
                                     │                     │                            │
                                     │                     ▼                            │
                                     │  ┌──────────────────────────────────────────┐    │
                                     │  │              Scheduler                   │    │
                                     │  │                                          │    │
                                     │  │  Receives CBs from FlexStream:           │    │
                                     │  │    CB(DMA correction tensor)             │    │
                                     │  │    CB(DeviceCompute)                     │    │
                                     │  │  (HostOp ran on CPU — no CB)             │    │
                                     │  │                                          │    │
                                     │  │  Drains FIFO onto hardware:              │    │
                                     │  │  ┌────────┐ ┌─────────┐                 │    │
                                     │  │  │  DMA   │→│ Compute │→ done           │    │
                                     │  │  │ corr   │ │  launch │                 │    │
                                     │  │  └────────┘ └─────────┘                 │    │
                                     │  └──────────────────────────────────────────┘    │
                                     │                                                  │
                                     └──────────────────────────────────────────────────┘

 Host returns immediately after enqueuing.
 Call SpyreStream.Synchronize() to block until hardware completes.
```

**Step-by-step:**

1. `LaunchKernel(spyre_stream, exec_plan, tensors)` is called from torch-spyre
2. For each Job, delegates to `SpyreStream.Launch(job, tensors, allow_tiled_launch)`
3. SpyreStream extracts AllocationIndices from SpyreTensors and calls `FlexStream.Launch(job, allocation_indices, allow_tiled_launch)`
4. FlexStream resolves AllocationIndices to VirtualAddresses via `FlexAllocator.resolve()`, then compares tensor shapes against the DeviceCompute's expected input shapes:
   * Shapes match exactly → proceeds with single-iteration walk (this workflow)
   * Shapes exceed tile size and `allow_tiled_launch` is true → tiled iterations (see Workflow 4)
   * Shapes exceed tile size and `allow_tiled_launch` is false → raises exception
5. FlexStream walks the Job's JobPlan:
   * **HostOperation** (on CPU): converts tensor virtual address metadata using the Job's `program_correction_metadata`, producing a correction tensor
   * **DMA** (ToDevice): enqueues a DMA control block to transfer the correction tensor to device (segment 7, address 0)
   * **DeviceCompute**: enqueues a compute control block to launch the unified binary (correction + matmul)
6. Scheduler receives 2 control blocks (DMA CB, then compute CB) and drains them onto hardware in FIFO order
7. Host returns immediately; call `SpyreStream.Synchronize()` when results are needed

#### Workflow 4: Tiled Execution

```
 torch-spyre                                    flex
┌──────────────────────────────────┐  ┌──────────────────────────────────────────────┐
│                                  │  │                                              │
│  LaunchKernel(spyre_stream,      │  │                                              │
│    exec_plan, tensors)           │  │                                              │
│         │                        │  │                                              │
│  SpyreStream.Launch(job, tensors,│  │                                              │
│    allow_tiled_launch=true)      │  │                                              │
│         │                        │  │                                              │
│  (extract AllocationIndices)     │  │                                              │
│         │                        │  │                                              │
│         └────────────────────────┼──┼─▶ FlexStream.Launch(job, alloc_indices,      │
│                                  │  │     allow_tiled_launch=true)                 │
│                                  │  │         │                                    │
└──────────────────────────────────┘  │  Detects shapes exceed tile size             │
                                      │  allow_tiled_launch=true → proceed           │
                                      │  Infer: 4 iterations (4096/1024)             │
                                      │         │                                    │
                                      │    ┌────┴────┐                               │
                                      │    │ iter 0  │─▶ HostOp₀→DMA₀→Compute₀     │
                                      │    ├─────────┤                               │
                                      │    │ iter 1  │─▶ HostOp₁→DMA₁→Compute₁     │
                                      │    ├─────────┤                               │
                                      │    │ iter 2  │─▶ HostOp₂→DMA₂→Compute₂     │
                                      │    ├─────────┤                               │
                                      │    │ iter 3  │─▶ HostOp₃→DMA₃→Compute₃     │
                                      │    └─────────┘                               │
                                      │   (8 CBs total: 2 per iter, FIFO in stream)  │
                                      │   (4 HostOps run on CPU, not CBs)            │
                                      │                     │                        │
                                      │                     ▼                        │
                                      │  ┌──────────────────────────────────────┐    │
                                      │  │            Scheduler                 │    │
                                      │  │  Drains all 8 CBs onto HW:          │    │
                                      │  │  DMA₀→Comp₀→DMA₁→Comp₁→            │    │
                                      │  │  DMA₂→Comp₂→DMA₃→Comp₃             │    │
                                      │  └──────────────────────────────────────┘    │
                                      └──────────────────────────────────────────────┘
```

1. `LaunchKernel(spyre_stream, exec_plan, tensors)` is called from torch-spyre
2. For each Job, delegates to `SpyreStream.Launch(job, tensors, allow_tiled_launch=true)`
3. SpyreStream extracts AllocationIndices from SpyreTensors and calls `FlexStream.Launch(job, allocation_indices, allow_tiled_launch=true)`
4. FlexStream compares tensor shapes against the DeviceCompute's expected input shapes — detects shapes exceed tile size
5. `allow_tiled_launch` is true → FlexStream proceeds with tiled execution
6. FlexStream infers tiling dimension and `num_iterations = 4096 / 1024 = 4`
7. For each iteration `i`, FlexStream walks the Job's JobPlan with updated addresses:
   * Computes updated tensor offsets for iteration `i`
   * **HostOperation** (on CPU): converts tensor virtual address metadata for this iteration using correction metadata → correction tensor
   * **DMA** (ToDevice): enqueues DMA CB — correction tensor → device (segment 7, address 0)
   * **DeviceCompute**: enqueues compute CB — launch unified binary for this tile slice
8. All 8 control blocks (2 per iteration × 4 iterations) are forwarded to the Scheduler. 4 HostOperations ran on the CPU.
9. Scheduler drains all control blocks onto hardware in FIFO order
10. Host returns immediately; call `SpyreStream.Synchronize()` to block until all iterations complete

#### Workflow 5: End-to-End

```
                                                    # Default SpyreStream created at runtime start

1. tensor_a = torch.randn(4096, 1024)             # Create on CPU
2. spyre_a = tensor_a.to("spyre")                 # Job([DMA ToDevice]) → FlexStream walks plan → Scheduler
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
                                                    # (8 CBs: 2 per iteration + 4 HostOps on CPU)

6. result_cpu = result.to("cpu")                   # Job([DMA FromDevice]) → FlexStream walks plan → Scheduler
```

#### Workflow 6: Multi-Stream (Future Hardware)

```
stream_a = SpyreStream()                           # Stream for layer 1
stream_b = SpyreStream()                           # Stream for layer 2

# Enqueue independent work on separate streams
LaunchKernel(stream_a, exec_plan_1, tensors_1)     # Async — SpyreStream passes through to FlexStream
LaunchKernel(stream_b, exec_plan_2, tensors_2)     # Async — no ordering w.r.t. stream_a

# With current hardware: Scheduler serializes both streams' jobs
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

FlexStream is modeled after CUDA streams and shares the two fundamental guarantees: **async enqueue** (all methods return control to the host immediately) and **intra-stream FIFO ordering** (jobs within a stream execute sequentially, with no ordering guarantees across streams).

The key difference is where work submission lives. A CUDA stream is a **passive handle** — you pass it as a parameter to external API calls like `cudaMemcpyAsync(dst, src, size, kind, stream)` or kernel launches via `<<<grid, block, sharedMem, stream>>>`. FlexStream is an **active object** — jobs are submitted as methods on the stream itself (`stream.Launch(op, ...)`, `stream.Synchronize()`).

The following table maps CUDA stream capabilities to their FlexStream equivalents:

| Capability | CUDA Stream | FlexStream |
|------------|-------------|------------|
| Copy | `cudaMemcpyAsync(dst, src, size, kind, stream)` — standalone primitive call | Data transfers are expressed as DMA steps within a Job's JobPlan; FlexStream maps them to DMA control blocks internally |
| Launch | `kernel<<<grid, block, sharedMem, stream>>>()` — takes grid/block dims, function pointer, args | `Launch(Job, allocation_indices, allow_tiled_launch=true)` — accepts a compound Job and decomposes it into the correct sequence of low-level steps (no grid/block dims since Spyre is not SIMT). Automatically detects when tensor shapes exceed the compiled tile size and handles tiled execution transparently when `allow_tiled_launch` is true. |
| Tiled launch | User changes grid dims to cover larger tensors | Handled automatically by `Launch` — no separate API. FlexStream iterates over tiles, enqueuing multiple decompositions with updated offsets. |
| Synchronize | `cudaStreamSynchronize(stream)` | `Synchronize()` — identical semantics |
| Events | `cudaEventRecord` / `cudaStreamWaitEvent` for inter-stream dependencies | Not present (see unresolved question #1) |
| Query | `cudaStreamQuery()` — non-blocking completion poll | Not present (see unresolved question #6) |
| Priorities | `cudaStreamCreateWithPriority()` | Not present (see unresolved question #7) |
| Host callbacks | `cudaLaunchHostFunc()` — enqueue a host-side function on the stream | HostOperation steps in a Job's JobPlan — host-side functions that execute in stream order as part of job processing |

| Default stream | NULL stream with legacy blocking semantics | Not yet specified (see unresolved question #7) |

FlexStream is a deliberately minimal subset — 2 methods vs. CUDA's dozens. The omitted features (events, query, priorities) are captured as unresolved questions for future consideration as hardware and runtime requirements evolve.

### SpyreStream and FlexStream — Compound Jobs

In CUDA, a kernel launch is self-contained — the user or library (cuBLAS, cuDNN) manually issues the correct sequence of memcpy and launch calls on a stream. Spyre jobs have inherent **compound structure** that requires the runtime to decompose a single logical job into multiple low-level steps:

* **A CUDA matmul**: 1 kernel launch
* **A Spyre matmul**: host operation (convert tensor vaddr metadata) → DMA correction tensor → device compute (3 steps, 2 device CBs)

FlexStream owns this decomposition as a first-class part of the runtime. The closest CUDA analogs are vendor libraries like cuBLAS and cuDNN, which internally issue multi-step sequences on a user-provided stream — but those are external libraries, not a core stream layer. In the Spyre stack, compound decomposition is built into FlexStream because every Spyre compute requires it, not just optimized library calls.

SpyreStream, by contrast, is a thin PyTorch Stream wrapper with no CUDA analog needed — it simply bridges PyTorch runtime conventions (torch tensors, device metadata) with FlexStream's types (Jobs, AllocationIndices).

FlexStream's automatic tiling within `Launch` also has no CUDA equivalent. CUDA kernels accept arbitrary dimensions as launch parameters (`<<<grid, block>>>`), so the user simply changes grid dims to cover larger tensors. Spyre kernels are compiled for fixed tile shapes, so FlexStream's `Launch` must detect this and iterate over tiles, enqueuing multiple rounds of low-level steps with updated tensor offsets — all transparently within the same `Launch` call.

## **How we teach this**

TBD

## **Unresolved questions**

1. **Stream events / synchronization primitives**: FlexStream will need inter-stream synchronization (e.g., CUDA-style events) to express dependencies between streams without full synchronization. The requirement is clear, but the implementation approach is still open.

2. **Reduction tiling**: When tiling along the reduction dimension (K in matmul), partial results must be accumulated. Should the runtime handle accumulation internally, or should this be expressed as a separate job in the ExecutionPlan?

3. **Multi-dimensional tiling**: If both M and N exceed the tile size, the runtime needs a nested loop. Should tiled execution support arbitrary nesting, or should this be constrained to single-dimension tiling with multi-dim handled by the compiler?

4. **Async iteration overlap**: Currently each tiled iteration within a stream is sequential. Could we use separate streams or double-buffering to overlap iteration N+1's data movement with iteration N's compute?

5. **Program correction across PF/VF modes**: The HostOperation (conversion of tensor virtual address metadata) takes VirtualAddresses as input — these are resolved from the tensor AllocationIndices by FlexStream calling `FlexAllocator.resolve()`. Since VirtualAddress is a single type (`region_id` + `offset`) in both modes, the HostOperation can process them uniformly — but the correction program on device may still need to interpret `region_id` differently (firmware lookup index vs. physical address). How should this mode distinction be communicated to the correction program?

6. **Non-blocking completion query**: Should FlexStream support a `Query()` method (analogous to `cudaStreamQuery()`) that returns whether all enqueued jobs have completed without blocking? This would allow polling-based patterns and avoid unnecessary synchronization.

7. **Stream priorities**: Should FlexStream support creating streams with priority levels (analogous to `cudaStreamCreateWithPriority()`)? This is related to but distinct from unresolved question #6 (Scheduler policies) — #6 asks how the Scheduler drains multiple streams, while this asks whether streams themselves carry priority metadata that the Scheduler could use.

8. **Pipelining across jobs**: The hardware supports a 3-stage pipeline across jobs: while one job's DeviceCompute CB executes, the HostOperation and DMA for the *next* job can overlap (since DMA does not engage AI cores). This inter-job pipelining is distinct from unresolved question #4 (which covers intra-job tiled iteration overlap). How should FlexStream and the Scheduler coordinate to exploit this overlap — should FlexStream speculatively walk ahead in the job queue, or should the Scheduler handle this internally?

9. **Symbolic shapes as HostOperation inputs**: Symbolic shapes (not just addresses) can be inputs to the HostOperation if the kernel was compiled with symbolic shapes. Should the frontend (inductor) use program selection (multiple static kernels) or pass symbolic shape values through to the runtime for the HostOperation to resolve? This affects what the HostOperation receives as input and whether the runtime needs to handle dynamic shapes.

10. **Correction tensor size and segment 7 allocation**: The size of the correction tensor produced by the HostOperation varies kernel-to-kernel and is determined by the Job's `program_correction_metadata`. The DMA step's `size` must match this. Should the DMA `size` be derived from the correction metadata at runtime, or should the compiler encode the expected correction tensor size explicitly in the JobPlan? Additionally, the reserved space at segment 7, address 0 must be statically allocated — how large does this reservation need to be? It must accommodate the largest correction tensor across all Jobs, but the maximum size may not be known until all Jobs in an ExecutionPlan are compiled.

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
