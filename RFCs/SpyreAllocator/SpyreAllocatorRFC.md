# SpyreAllocator

**Authors:**
* @JRosenkranz
* @andrea-fasoli
* @wangchen615

## **Summary**

This RFC introduces the Spyre memory allocation architecture, composed of two layers: `SpyreAllocator` (in torch-spyre) and `FlexAllocator` (in flex). `SpyreAllocator` is a thin wrapper that implements PyTorch's `at::Allocator` interface and delegates all memory management to `FlexAllocator`. `FlexAllocator` is the core memory manager — it manages a pool of dynamically-acquired memory regions on device and carves individual allocations from them as contiguous, 128-byte-aligned blocks. It returns a `VirtualAddress` (`region_id` + `offset`) for every allocation, providing a uniform addressing model that the rest of the stack — SpyreTensor, Operation, FlexStream — consumes without needing to know whether the system is running in PF or VF mode.

The allocator operates in two modes, selected by the `FLEX_DEVICE` environment variable. Both modes use the same allocation logic — dynamically acquiring memory regions and carving individual allocations from them as blocks, returning a `VirtualAddress` (`region_id` + `offset`). The only difference between modes is the interpretation of `region_id`:

- **PF Mode (Physical Function)**: `region_id` is the physical address of the start of the memory region.
- **VF Mode (Virtual Function)**: `region_id` is an index into a firmware lookup table that maps to the physical address of the start of the memory region.

The allocator is the sole owner of the mapping between host-visible allocation metadata and device-side addressing.

## **Motivation**

The current PF-mode allocator returns a physical address for each tensor allocation — each `allocate()` call acquires a dedicated device memory region from the flex allocator, and the returned address directly identifies that region in physical memory. This works in PF (single-tenant) mode because the host has unrestricted access to the device's physical address space.

**VF mode introduces a hardware constraint that fundamentally changes the allocation model.** In VF (multi-tenant) mode, the device exposes only a limited number of handles (memory region slots) to each virtual function — the firmware provides a small lookup table (on the order of 8–16 entries) that maps region indices to physical addresses. This means the allocator cannot acquire a separate device memory region for every tensor; it would exhaust the available handles almost immediately. Instead, the allocator must manage a small set of memory regions and carve many individual allocations from within them — effectively implementing its own virtual memory management on top of the limited hardware handles.

This constraint motivates SpyreAllocator's core design:

1. **Sub-allocation within regions**: Rather than one device region per tensor, SpyreAllocator acquires a small number of large regions (bounded by the hardware handle limit) and sub-allocates blocks within them. Each block is an offset within a region, not a separate device handle. This allows hundreds or thousands of live tensors while consuming only a handful of firmware table entries.

2. **Uniform addressing via VirtualAddress**: The `VirtualAddress` type (`region_id` + `offset`) provides a single addressing abstraction that works in both modes (see Summary for `region_id` interpretation). All downstream components — SpyreTensor, Operation, FlexStream — consume VirtualAddress identically, with no mode-aware branching.

3. **Encapsulated memory management**: The region pool, block allocation, coalescing, and thread safety are all encapsulated within `FlexAllocator` in flex. torch-spyre only ever receives a `VirtualAddress` from `FlexAllocator`, which it stores as part of the `at::DataPtr`. torch-spyre never interacts with regions, blocks, `DeviceMemoryAllocationPtr`, or raw byte offsets. This decouples torch-spyre from flex's internal memory representation and isolates future allocator changes (new search strategies, compaction, memory pressure callbacks) from the rest of the stack.

## **Proposed Implementation**

### Core Components

#### VirtualAddress

A reference to a location in device memory, produced by FlexAllocator. A VirtualAddress identifies only a location — size is stored separately alongside it.

Properties:

| Property | Description |
|----------|-------------|
| `region_id` | Identifies the memory region. In VF mode, this is an index into a firmware lookup table that maps to the physical address of the region. In PF mode, this is the physical address of the region itself. |
| `offset` | Byte offset of the allocation within the region. Always 128-byte aligned. |

**Relationship to `at::DataPtr`**: `SpyreAllocator.allocate()` delegates to `FlexAllocator.allocate()`, which returns a `VirtualAddress`. SpyreAllocator wraps this in an `at::DataPtr` with a `SharedOwnerCtx` as its opaque context. The `VirtualAddress` is stored directly — `region_id` identifies the region and `offset` is the block's byte position within it.

#### MemoryBlock

A contiguous interval within a `MemoryRegion`, representing either an occupied allocation or a free region available for allocation.

Properties:

| Property | Description |
|----------|-------------|
| `start` | Byte offset of the block's start within the region. Always 128-byte aligned. |
| `end` | Byte offset of the block's end within the region (exclusive). |
| `is_free` | `true` if this block is unallocated and available for use. |

Derived:

| Method | Description |
|--------|-------------|
| `size()` | Returns `end - start` — the byte size of this block. |

Ordering: Blocks are ordered by `start` offset within a region. This ordering is critical for the coalescing algorithm — when a block is freed, the allocator checks the immediately preceding and following blocks (by offset order) and merges any that are also free into a single larger free block.

**Invariants:**
- Blocks within a region are non-overlapping and contiguous — the `end` of one block equals the `start` of the next.
- No two adjacent blocks are both free — the allocator always coalesces on deallocation.
- `start` is always 128-byte aligned.

#### MemoryRegion

A contiguous region of device memory acquired from the `DeviceMemoryAllocator` through a `TryAllocate` call in flex. A region is the unit of device memory acquisition — the allocator obtains entire regions from the device and then carves individual allocations from them as `MemoryBlock` entries.

Properties:

| Property | Description |
|----------|-------------|
| `region_id` | Unique identifier for this region, derived from the `DeviceMemoryAllocationPtr` returned by the `DeviceMemoryAllocator`. In VF mode, this is the VF index from the `DeviceMemoryAllocationPtr`. In PF mode, this is the `dmpa_bytes_` (physical address) from the `DeviceMemoryAllocationPtr`. This value is used directly as the `region_id` in the returned `VirtualAddress`. |
| `data` | `DeviceMemoryAllocationPtr` — shared handle to the underlying device memory. All blocks within this region share the same `DeviceMemoryAllocationPtr`. |
| `total_size` | Total byte size of this region. |
| `free_size` | Sum of all free block sizes in this region. Updated on every allocation and deallocation. Used for load-balancing region selection. |
| `blocks` | `std::set<MemoryBlock>` — ordered set of all blocks (free and occupied) in this region, sorted by `start` offset. |
| `ctx_to_block` | `std::unordered_map<SharedOwnerCtx*, MemoryBlock*>` — maps each active allocation's context to its occupied block, enabling O(1) lookup during deallocation. |
| `free_sizes` | `std::multiset<size_t>` — multiset of free block sizes. Enables O(log R) lookup to determine whether a region can satisfy a request of a given size without scanning all blocks (where R = number of free ranges in the region). |

**Relationship to the ControlBlock segment table:** When a ControlBlock is constructed to dispatch compute to the device, it contains a segment table whose entries map to the `region_id` of each MemoryRegion. The segment table is static — it is populated once from the set of acquired MemoryRegions and does not change across dispatches. The device firmware uses the segment table to resolve `region_id` values in `VirtualAddress` operands to physical memory locations during execution.

**Lifecycle:**
1. A region is created when `allocateNewRegion()` succeeds — it starts as a single free `MemoryBlock` spanning the entire region.
2. Allocations carve occupied blocks from free blocks, potentially splitting a free block into an occupied block and a smaller free remainder.
3. Deallocations mark blocks as free and coalesce adjacent free blocks.
4. A region is never returned to the `DeviceMemoryAllocator` during normal operation (regions are long-lived).

#### SharedOwnerCtx

The opaque context carried inside an `at::DataPtr`. It bridges PyTorch's reference-counted memory management with the allocator's block tracking.

Properties:

| Property | Description |
|----------|-------------|
| `owner` | `VirtualAddress` — the allocation's location on device, as returned by `FlexAllocator.allocate()`. Contains the `region_id` and `offset` needed to identify and later free this block. |
| `device_id` | Identifies which Spyre device this allocation belongs to. |

When PyTorch drops the last reference to a tensor, the `at::DataPtr`'s custom deleter is invoked, which receives the `SharedOwnerCtx` and calls back into SpyreAllocator to free the corresponding block via `FlexAllocator.deallocate(owner)`.

#### SpyreAllocator

A thin wrapper in torch-spyre that implements the `at::Allocator` interface. SpyreAllocator holds an instance of `FlexAllocator` and delegates all memory management to it. Its role is to bridge PyTorch's allocator interface with the flex memory management layer — translating `allocate()` calls into `FlexAllocator` operations and wrapping the result in an `at::DataPtr` with the appropriate custom deleter.

Methods:

| Method | Description |
|--------|-------------|
| `allocate(size_t nbytes) -> at::DataPtr` | Implements `at::Allocator::allocate`. Delegates to `FlexAllocator.allocate(nbytes)`, which returns a `VirtualAddress`. Wraps the result in an `at::DataPtr` with a `SharedOwnerCtx` and custom deleter. |
| `static instance() -> SpyreAllocator&` | Returns the singleton allocator. |
| `static ReportAndDelete(void* ctx)` | Custom deleter for `at::DataPtr`. Delegates to `FlexAllocator.deallocate()` to free the corresponding block, then deletes the `SharedOwnerCtx`. |

#### RegionAcquisitionStrategy

An interface that controls how FlexAllocator selects regions for allocation. The strategy determines the order in which existing regions are tried and, implicitly, when a new region must be acquired (i.e., when no existing region selected by the strategy can satisfy the request).

Methods:

| Method | Description |
|--------|-------------|
| `selectRegions(regions, nbytes) -> ordered list of MemoryRegion*` | Given the current set of regions and a requested size, returns an ordered list of candidate regions to try. FlexAllocator tries each in order until one can satisfy the request. If none can, a new region is acquired (if `regions_locked` is false). |

##### LoadBalancingStrategy

The default strategy. Sorts regions by descending `free_size`, so that the region with the most free space is tried first. This distributes allocations across regions and reduces fragmentation by keeping free space consolidated.

**Behavior:** Regions with the most available space are preferred. Allocations spread across all regions, which maximizes the chance of finding a large contiguous block but consumes more region handles.

**Best for:** Workloads where fragmentation is the primary concern and region handles are not scarce (PF mode, or VF mode with low region pressure).

##### FillFirstStrategy

Sorts regions by ascending `free_size` (least free space first, among those that can still satisfy the request), packing allocations tightly into regions that are already partially filled before moving to emptier ones.

**Behavior:** The most-filled region that can still satisfy the request is preferred. Allocations concentrate into fewer regions, which minimizes the number of active region handles. Regions that are nearly full are tried first, keeping the remaining regions as free as possible for future large allocations.

**Best for:** VF (multi-tenant) mode where region handles are scarce (8–16 entries). By packing allocations into fewer regions, this strategy delays the need to acquire new regions, leaving handles available for other uses (e.g., ExecutionPlan binary loading).

#### BlockAcquisitionStrategy

An interface that controls how FlexAllocator selects a free block within a region. The strategy determines which free block is chosen when multiple free blocks in a region can satisfy the request.

Methods:

| Method | Description |
|--------|-------------|
| `selectBlock(blocks, free_sizes, nbytes) -> MemoryBlock*` | Given the ordered set of blocks and the multiset of free sizes for a region, returns a pointer to the free block to allocate from, or `nullptr` if no suitable block exists. The `free_sizes` multiset enables an O(log R) pre-check — if the largest value in `free_sizes` is less than `nbytes`, the strategy can return `nullptr` immediately without scanning blocks. |

##### FirstFitStrategy

The default strategy. Scans blocks in offset order and returns the first free block whose `size() >= nbytes`.

**Behavior:** The earliest (lowest-offset) free block that fits is selected. This is fast — it stops scanning as soon as a match is found — and tends to pack allocations toward the beginning of the region, leaving larger contiguous free space toward the end.

**Best for:** General-purpose workloads where allocation speed is the priority. First-fit has good average-case performance and avoids the overhead of scanning all free blocks.

##### BestFitStrategy

Scans all free blocks and returns the smallest free block whose `size() >= nbytes`.

**Behavior:** The tightest-fitting free block is selected. This minimizes the size of the leftover remainder after splitting, reducing the creation of small, hard-to-use free fragments. However, it requires scanning all free blocks in the region (O(B) where B = number of blocks).

**Best for:** Workloads with highly variable allocation sizes where fragmentation is the primary concern. By choosing the tightest fit, this strategy preserves larger free blocks for future large allocations at the cost of slower allocation.

#### FlexAllocator

The actual memory management implementation, residing in flex. FlexAllocator manages a dynamic pool of `MemoryRegion` instances and carves individual allocations from them as `MemoryBlock` entries.

Constants:

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_REGIONS` | 12 | Upper bound on the number of regions the allocator will acquire. |
| `MIN_ALLOC_BYTES` | 128 | Minimum allocation size and alignment boundary (bytes). All allocations are rounded up to a multiple of this value. |

State:

| Field | Type | Description |
|-------|------|-------------|
| `regions` | `std::vector<MemoryRegion>` | Dynamic list of acquired regions. |
| `block_to_region` | `std::unordered_map<SharedOwnerCtx*, MemoryRegion*>` | Maps each active allocation to its containing region for O(1) deallocation lookup. |
| `regions_locked` | `bool` | When `true`, no new regions will be acquired — all allocations must be satisfied from existing regions. Transitions to `true` when `MAX_REGIONS` is reached or when all fallback sizes fail. |
| `fallback_sizes` | `std::vector<size_t>` | Ordered list of region sizes to attempt: `{12GB, 8GB, 4GB}`. |
| `max_regions` | `size_t` | Maximum number of regions (default: 12). |
| `region_acquisition_strategy` | `RegionAcquisitionStrategy` | Controls region selection order. Defaults to `LoadBalancingStrategy`. |
| `block_acquisition_strategy` | `BlockAcquisitionStrategy` | Controls block selection within a region. Defaults to `FirstFitStrategy`. |
| `allocator_mutex` | `std::mutex` | Protects all region and block operations. |

Methods:

| Method | Description |
|--------|-------------|
| `allocate(size_t nbytes) -> VirtualAddress` | Main entry point. Acquires the mutex, aligns `nbytes` to 128 bytes, finds or creates a suitable block, and returns a `VirtualAddress` identifying the allocation's location on device. |
| `deallocate(VirtualAddress)` | Frees the block corresponding to the given `VirtualAddress`. Coalesces with adjacent free blocks (predecessor and successor by offset). Updates `free_size`, `free_sizes`, and removes mappings. |
| `allocateNewRegion(DeviceMemoryAllocatorPtr) -> bool` | Attempts to acquire a new region from the device, trying each size in `fallback_sizes` in order. Returns `true` on success. On failure (all sizes exhausted or `max_regions` reached), sets `regions_locked = true`. |
| `findFreeBlock(size_t nbytes, DeviceMemoryAllocatorPtr) -> AllocationInfo` | Locates a suitable free block. Uses `region_acquisition_strategy.selectRegions(regions, nbytes)` to produce an ordered list of candidate regions, then tries each by calling `allocateInRegion()`. If no existing region can satisfy the request and `regions_locked` is `false`, attempts `allocateNewRegion()`. |
| `allocateInRegion(MemoryRegion*, size_t nbytes) -> MemoryBlock*` | Selects a free block within the given region using the `block_acquisition_strategy`, then marks it as occupied. If the selected block is larger than needed, splits it into an occupied block of size `nbytes` and a free remainder block. Returns `nullptr` if the strategy finds no suitable block. Updates `free_size` and `free_sizes`. |
| `setMinSpyreAllocation(size_t nbytes) -> size_t` | Rounds `nbytes` up to the nearest multiple of `MIN_ALLOC_BYTES` (128). |

### Allocation Algorithm

The allocator uses a two-phase strategy: **region acquisition** followed by **block assignment**.

#### Phase 1: Region Acquisition

Region acquisition determines which region to allocate from. The behavior depends on the `region_acquisition_strategy`:

- **`LoadBalancingStrategy`**: If `regions_locked` is `false`, a new region is acquired first — the allocator tries each size in `fallback_sizes` (`{12GB, 8GB, 4GB}`) until one succeeds. The newly acquired region (fully free) is added to the pool. The strategy then orders all regions by descending `free_size`, so the new region (with the most free space) is tried first. This distributes allocations across regions.

- **`FillFirstStrategy`**: Existing regions are tried first. The strategy orders regions by ascending `free_size` (most-filled first, among those that can satisfy the request), packing allocations tightly. A new region is only acquired when no existing region can satisfy the request and `regions_locked` is `false`.

New region acquisition follows a fallback sequence:

1. The allocator attempts to acquire a region of the largest fallback size (12 GB).
2. If that fails (insufficient device memory), it tries the next fallback size (8 GB), then the next (4 GB).
3. If a region is successfully acquired, it is added to the `regions` vector as a single free `MemoryBlock` spanning the entire region.
4. If all fallback sizes fail, or if `max_regions` has been reached, `regions_locked` is set to `true` — no further region acquisition is attempted for the lifetime of the allocator.

The `region_acquisition_strategy` produces an ordered list of candidate regions via `selectRegions(regions, nbytes)`. FlexAllocator tries each candidate in order by calling `allocateInRegion()`.

#### Phase 2: Block Assignment

Once a region is selected, `allocateInRegion()` handles block assignment:

1. **Block search**: The `block_acquisition_strategy` selects a free block via `selectBlock(blocks, free_sizes, nbytes)`. The strategy receives the region's ordered block set and free-size multiset, and returns the block to allocate from (or `nullptr` if no suitable block exists). With `FirstFitStrategy` (default), this scans blocks in offset order and returns the first free block whose `size() >= nbytes`. With `BestFitStrategy`, this scans all free blocks and returns the smallest one that fits. Both strategies use the `free_sizes` multiset for an O(log R) pre-check — if the largest value in `free_sizes` is less than `nbytes`, the strategy returns `nullptr` immediately without scanning blocks.

2. **Block splitting**: If the found free block is larger than `nbytes`, it is split into:
   - An occupied block of exactly `nbytes` (starting at the original block's `start`)
   - A free remainder block (starting at `start + nbytes`, ending at the original `end`)

3. **Registration**: The new occupied block is registered in `ctx_to_block` and `block_to_region` for O(1) deallocation lookup.

**Complexity:**
- Allocation: O(S x R + log R) where S = regions, R = free ranges per region
- Deallocation: O(S + log R) — O(1) region lookup via `block_to_region`, O(log R) for coalescing via `blocks` set operations

### Deallocation and Coalescing

When a block is freed (via the `at::DataPtr` custom deleter):

1. **Lookup**: Find the region via `block_to_region[ctx]` and the block via `region.ctx_to_block[ctx]`.

2. **Mark free**: Set `is_free = true` on the block.

3. **Coalesce with predecessor**: Using the ordered `blocks` set, check the block immediately before this one (by offset). If it is also free, merge: create a new free block spanning both, remove the two originals, insert the merged block.

4. **Coalesce with successor**: Check the block immediately after. If free, merge similarly.

5. **Update bookkeeping**: Update `free_size`, `free_sizes`, remove entries from `ctx_to_block` and `block_to_region`, delete the `SharedOwnerCtx`.

This ensures the invariant that no two adjacent free blocks exist — free space is always maximally coalesced.

### Thread Safety

All allocation and deallocation operations acquire `allocator_mutex` before modifying any state. This is a single coarse-grained mutex covering the entire allocator. The lock scope includes:

- Region acquisition (`allocateNewRegion`)
- Block search and splitting (`findFreeBlock`, `allocateInRegion`)
- Block freeing and coalescing (`deallocateBlock`)
- All map updates (`block_to_region`, `ctx_to_block`)

The `ReportAndDelete` static deleter also acquires the mutex, since it is called from arbitrary threads when PyTorch drops tensor references.

### Alignment

All allocations are rounded up to a multiple of 128 bytes (`MIN_ALLOC_BYTES`). This is a Spyre hardware requirement — the device expects memory addresses to be 128-byte aligned. The `setMinSpyreAllocation()` method performs this rounding:

```
aligned = ((nbytes + MIN_ALLOC_BYTES - 1) / MIN_ALLOC_BYTES) * MIN_ALLOC_BYTES
```

Block `start` offsets are always 128-byte aligned because:
- The first block in a region starts at offset 0 (aligned).
- When a block is split, the occupied portion is a multiple of 128 bytes (due to alignment rounding), so the remainder's `start` is also aligned.

### Debugging

When the `TORCH_SPYRE_ALLOC_DEBUG=1` environment variable is set, the allocator emits verbose logging for every allocation and deallocation, including:
- Region acquisition (size, fallback level)
- Block assignment (region id, offset, size)
- Block freeing and coalescing details
- Region free space summaries

This can also be enabled alongside `TORCH_SPYRE_DEBUG=1` for broader runtime debug output.

### Workflows

#### Workflow 1: First Allocation (New Region)

```
┌───────────────┐     ┌───────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  allocate(N)  │────▶│  findFreeBlock(N)  │────▶│ allocateNewRegion │────▶│DeviceMemAllocator│
│  (align to    │     │  regions_locked   │     │  try 12GB          │     │ acquire region   │
│   128 bytes)  │     │  = false           │     │  try 8GB (fallback)│     │ → DeviceMemAlloc │
└───────────────┘     └───────────────────┘     │  try 4GB (fallback)│     └──────────────────┘
                                                 └────────────────────┘
                                                          │
                                                          ▼
                                                 ┌────────────────────┐     ┌──────────────────┐
                                                 │  MemoryRegion     │────▶│  allocateInRegion│
                                                 │  [████████░░░░░░]  │     │  split free block│
                                                 │   ^occupied  ^free │     │  → MemoryBlock*  │
                                                 └────────────────────┘     └──────────────────┘
                                                                                     │
                                                                                     ▼
                                                                            ┌──────────────────┐
                                                                            │  at::DataPtr     │
                                                                            │  SharedOwnerCtx  │
                                                                            │  owner = VAddr   │
                                                                            │  (region+offset) │
                                                                            └──────────────────┘
```

1. `allocate(nbytes)` is called. `nbytes` is rounded up to a 128-byte multiple.
2. `findFreeBlock()` is called. `regions_locked` is `false` and no regions exist yet.
3. `allocateNewRegion()` attempts to acquire a 12 GB region from the `DeviceMemoryAllocator`. If that fails, it tries 8 GB, then 4 GB.
4. On success, a new `MemoryRegion` is created with a single free `MemoryBlock` spanning the full region.
5. `allocateInRegion()` splits that free block: the first `nbytes` become an occupied block; the remainder stays free.
6. FlexAllocator returns a `VirtualAddress` with `region_id` = the MemoryRegion's `region_id` and `offset` = the occupied block's `start` offset.
7. A `SharedOwnerCtx` is created with `owner` = the returned `VirtualAddress`.
8. An `at::DataPtr` wrapping the `SharedOwnerCtx` is returned to the caller.

#### Workflow 2: Subsequent Allocation (Existing Region)

```
┌───────────────┐     ┌───────────────────┐     ┌─────────────────────────────────────┐
│  allocate(N)  │────▶│  findFreeBlock(N)  │────▶│  region_acquisition_strategy:              │
│               │     │  regions exist     │     │    select region ordering            │
│               │     │                   │     │  block_acquisition_strategy:         │
│               │     │                   │     │    select block within region         │
└───────────────┘     └───────────────────┘     └─────────────────────────────────────┘
                                                          │
                                                          ▼
                                                 ┌────────────────────┐
                                                 │  MemoryRegion     │
                                                 │  [██░░██░░░░░░░░]  │
                                                 │  allocateInRegion │
                                                 │  block_strategy   │
                                                 │    → select block  │
                                                 │  split → occupied  │
                                                 │  [██░██░██░░░░░░]  │
                                                 └────────────────────┘
                                                          │
                                                          ▼
                                                 ┌──────────────────┐
                                                 │  at::DataPtr     │
                                                 │  (same region    │
                                                 │   DeviceMemAlloc)│
                                                 └──────────────────┘
```

1. `allocate(nbytes)` is called.
2. `findFreeBlock()` evaluates existing regions. The `region_acquisition_strategy` produces an ordered list of candidate regions.
3. For each candidate region, `allocateInRegion()` is called. Inside `allocateInRegion`, the `block_acquisition_strategy` calls `selectBlock(blocks, free_sizes, nbytes)` to find a suitable free block within the region. If the strategy returns `nullptr`, the next candidate region is tried.
4. `allocateInRegion()` splits the selected block and returns the occupied portion.
5. The new allocation's `SharedOwnerCtx` holds a `VirtualAddress` with the same `region_id` as other blocks in that region — only `offset` differs.

#### Workflow 3: Deallocation with Coalescing

```
┌───────────────────┐     ┌──────────────────┐     ┌────────────────────────────────┐
│  ReportAndDelete  │────▶│  block_to_region │────▶│  deallocate(VirtualAddress)     │
│  (PyTorch drops   │     │  [ctx → region]  │     │                                │
│   last reference) │     │  O(1) lookup     │     │  Before: [██ ░░ ██₂ ░░ ██]    │
└───────────────────┘     └──────────────────┘     │  Free block ██₂               │
                                                    │  Check predecessor ░░ → free  │
                                                    │  Check successor ░░ → free    │
                                                    │  Merge all three:             │
                                                    │  After:  [██ ░░░░░░░░░░ ██]   │
                                                    │  free_size += block.size()     │
                                                    └────────────────────────────────┘
```

1. PyTorch drops the last reference to a tensor — `ReportAndDelete` is invoked with the `SharedOwnerCtx`.
2. The allocator acquires the mutex and looks up the region via `block_to_region`.
3. `deallocate()` marks the block as free, then checks adjacent blocks:
   - If the predecessor (by offset) is free, the two are merged into a single larger free block.
   - If the successor (by offset) is free, it is merged as well.
4. `free_size` and `free_sizes` are updated. Mappings are removed. The `SharedOwnerCtx` is deleted.

#### Workflow 4: Region Exhaustion and Locking

```
┌───────────────────┐     ┌─────────────────────────────────────────────┐
│  allocateNewReg   │────▶│  Try 12GB → fail                            │
│  (all fallbacks)  │     │  Try 8GB  → fail                            │
│                   │     │  Try 4GB  → fail                            │
│                   │     │  regions_locked = true                      │
│                   │     │  ─────────────────────────                   │
│                   │     │  All future allocations must use existing    │
│                   │     │  regions. If no region can satisfy the       │
│                   │     │  request → allocation failure (OOM).         │
│                   │     └─────────────────────────────────────────────┘
│                   │
│  OR               │     ┌─────────────────────────────────────────────┐
│  max_regions     │────▶│  regions.size() == MAX_REGIONS              │
│  reached          │     │  regions_locked = true                      │
│                   │     │  Same behavior — no new regions attempted.   │
└───────────────────┘     └─────────────────────────────────────────────┘
```

Once `regions_locked` becomes `true`, no new regions are acquired — all allocations must be satisfied from existing regions using the configured `region_acquisition_strategy`. If no region has a free block large enough, the allocation fails (out-of-memory).

### Integration with the Execution Pipeline

FlexAllocator produces the `VirtualAddress` values that flow through the entire execution pipeline described in the Program Execution RFC:

```
SpyreAllocator.allocate(N)
        │ delegates to FlexAllocator.allocate(N)
        ▼
  at::DataPtr (SharedOwnerCtx with owner + offset)
        │
        ▼
  SpyreTensor (carries VirtualAddress derived from SharedOwnerCtx)
        │
        ▼
  SpyreStream.Launch(op, tensors)
        │ translate SpyreTensors → VirtualAddresses
        ▼
  FlexStream.Launch(op, virtual_addresses)
        │ decompose Operation → low-level steps using VirtualAddresses
        ▼
  Scheduler → Hardware
```

Key integration points:

1. **Tensor allocation** (`tensor.to("spyre")`): SpyreAllocator delegates to FlexAllocator, which produces a `VirtualAddress`. SpyreAllocator wraps it in an `at::DataPtr`. A `SpyreTensor` is constructed carrying that `VirtualAddress`.

2. **Binary loading** (ExecutionPlan loading): FlexAllocator allocates device memory for program binaries, returning a `VirtualAddress` for each. Each binary's `VirtualAddress` is stored on its Operation.

3. **Program correction**: FlexStream assembles correction input buffers containing the `VirtualAddress` values of resident tensors. These addresses originate from FlexAllocator.

4. **Tiled execution**: FlexStream computes per-iteration offsets by adjusting the `offset` field of tensor `VirtualAddress` values. The `region_id` stays the same — tiling moves within a region, not across regions.

## **Metrics**

TBD

## **Drawbacks**

1. **Single mutex bottleneck**: The coarse-grained `allocator_mutex` serializes all allocation and deallocation operations. Under high concurrency (many tensors being created/destroyed simultaneously), this could become a bottleneck. A per-region lock would reduce contention but adds complexity.

2. **Regions are never returned**: Once a region is acquired, it is held for the lifetime of the allocator. If the workload's memory footprint shrinks significantly after a peak, the device memory remains reserved. A region release policy could reclaim underused regions, but this adds complexity around live block migration.

3. **First-fit fragmentation**: First-fit allocation can leave small free fragments scattered across regions. Over time, this may lead to situations where the total free memory is sufficient but no single contiguous block can satisfy a request. More sophisticated strategies (best-fit, buddy system) could reduce fragmentation at the cost of allocation latency or implementation complexity.

4. **Fixed fallback sizes**: The `{12GB, 8GB, 4GB}` fallback sequence is hardcoded. Different workloads or hardware configurations may benefit from different region sizes. Making this configurable adds flexibility but requires careful default selection.

## **Alternatives**

### 1. Per-Allocation Device Calls (No Sub-Allocation) — Current PF Strategy

This is the current PF-mode allocator: every `allocate()` call goes directly to the `DeviceMemoryAllocator` to acquire a dedicated device memory region, and every deallocation releases that region back to the device. There is no sub-allocation — each tensor gets its own region.

**Pros:** No fragmentation, no region management, no coalescing logic.
**Cons:** High overhead per allocation (`DeviceMemoryAllocator` calls are expensive), and critically, the firmware lookup table has a fixed number of entries — sub-allocation within regions is necessary to support more allocations than table entries. This approach is unsustainable in VF mode where region handles are scarce (8–16 entries).

## **Prior Art**

### CUDA Memory Allocator (`cudaMalloc` / `cudaFree`)

CUDA's default allocator (`cudaMalloc`) acquires memory directly from the GPU's memory manager for each allocation — one device allocation per request, no sub-allocation.

To address the overhead of frequent device allocator calls, PyTorch implements its own caching allocator (`c10::cuda::CUDACachingAllocator`) on top of `cudaMalloc`. The caching allocator maintains pools of previously-allocated blocks organized by size, reusing freed blocks without returning them to the GPU. Key similarities with SpyreAllocator / FlexAllocator:

| Capability | CUDA Caching Allocator | SpyreAllocator / FlexAllocator |
|------------|----------------------|--------------------------|
| Sub-allocation | Splits large blocks into smaller ones | Splits free blocks within regions |
| Coalescing | Merges adjacent free blocks | Merges adjacent free blocks on deallocation |
| Size classes | Multiple pools organized by block size | Single pool with `free_sizes` multiset for O(log R) lookup |
| Region management | Acquires "segments" (large cudaMalloc calls) and caches them | Acquires regions with adaptive fallback sizes |
| Thread safety | Per-device mutex + per-stream free lists | Single `allocator_mutex` |
| Alignment | 512-byte alignment | 128-byte alignment (Spyre hardware requirement) |

The key difference is the addressing model: CUDA allocations return raw GPU pointers, while FlexAllocator returns `VirtualAddress` values that abstract over PF/VF firmware addressing. This indirection is necessary because Spyre's VF mode uses a firmware lookup table rather than direct physical addressing.

### Linux Kernel `vmalloc` / Slab Allocator

The Linux kernel's virtual memory allocator (`vmalloc`) acquires pages from the physical page allocator and maps them into a contiguous virtual address range. The slab allocator (`kmem_cache_create`) builds on top of this, pre-allocating pages and carving objects of a fixed size from them.

SpyreAllocator's region/block model is structurally similar to slab allocation:
- **Regions** are analogous to slabs (contiguous memory acquired from a lower-level allocator)
- **Blocks** are analogous to objects within a slab (carved from the slab's memory)

The key difference is that slab allocators use fixed-size objects within each slab, while SpyreAllocator supports variable-size blocks with first-fit search and coalescing — more like a general-purpose heap allocator than a true slab allocator.

## **How we teach this**

The SpyreAllocator / FlexAllocator pair should be taught as the "memory layer" of the Spyre stack — SpyreAllocator bridges PyTorch's `at::Allocator` interface, while FlexAllocator is the core engine that produces the `VirtualAddress` values the execution pipeline consumes. Key teaching points:

1. **VirtualAddress is the universal currency**: Every component below the allocator speaks VirtualAddress. FlexAllocator is the only component that knows how to produce one.

2. **PF vs. VF is hidden**: Users and most developers never need to know which mode is active. The allocator produces VirtualAddresses that look the same in both modes — only `region_id` interpretation differs, and that's handled by firmware.

3. **Regions are acquired lazily, blocks are carved eagerly**: The allocator doesn't pre-reserve all device memory. It acquires regions on demand (with fallback sizes) and carves blocks from them. This adaptive approach balances memory efficiency with allocation speed.

4. **Coalescing prevents fragmentation**: When blocks are freed, adjacent free blocks are automatically merged. This is the allocator's primary defense against fragmentation and requires no user intervention.

## **Testing**

Testing spans both layers of the allocator architecture:

### FlexAllocator Unit Tests (flex)

FlexAllocator should have a dedicated set of unit tests in flex that exercise the core memory management logic in isolation, without requiring a device or PyTorch. These tests should cover:

- **Block allocation and splitting**: Verify that allocations carve correctly-sized occupied blocks from free blocks, with proper 128-byte alignment and correct remainder splitting.
- **Deallocation and coalescing**: Verify that freeing a block merges it with adjacent free predecessors and successors, maintaining the invariant that no two adjacent blocks are both free.
- **Region acquisition and fallback**: Verify the fallback size sequence (`{12GB, 8GB, 4GB}`), `regions_locked` transitions, and `max_regions` enforcement.
- **Strategy behavior**: Verify that `RegionAcquisitionStrategy` (`LoadBalancingStrategy`, `FillFirstStrategy`) and `BlockAcquisitionStrategy` (`FirstFitStrategy`, `BestFitStrategy`) produce the expected region ordering and block selection for given inputs.
- **Fragmentation scenarios**: Allocate and free blocks in patterns that produce fragmentation, then verify that coalescing recovers contiguous free space and that subsequent allocations succeed.
- **Thread safety**: Concurrent allocation and deallocation from multiple threads to verify correctness under contention.

### SpyreAllocator Integration Tests (torch-spyre)

SpyreAllocator should be validated through upstream PyTorch tests that exercise the `at::Allocator` interface end-to-end. These tests run against the full stack (SpyreAllocator + FlexAllocator + device) and verify that tensor allocation, computation, and deallocation work correctly through PyTorch's standard APIs.

Additionally, torch-spyre should include tests that cover edge cases specific to Spyre's allocation model:

- **Allocation edge cases**: Large allocations that span most of a region, many small allocations that stress block management, and allocation patterns that exhaust all regions (`regions_locked` = `true`).
- **Program loading**: Verify that ExecutionPlan binary loading allocates device memory correctly alongside tensor allocations, and that the segment table is populated from the acquired MemoryRegions.
- **OOM behavior**: Verify graceful failure when all regions are locked and no region can satisfy a request.

## **Unresolved questions**

1. **Region release policy**: Should the allocator ever return regions to the `DeviceMemoryAllocator`? Currently regions are held for the allocator's lifetime. If the workload's memory footprint shrinks significantly, those regions represent wasted device memory. A release policy could reclaim underused regions, but live blocks within a region prevent release without migration.

2. **Per-region locking**: The current single-mutex design serializes all operations. Would per-region locks (with a global lock only for region acquisition) meaningfully reduce contention? This depends on allocation/deallocation frequency and concurrency patterns.

3. **Fallback size tuning**: The `{12GB, 8GB, 4GB}` fallback sequence and `MAX_REGIONS = 12` are initial values. How should these be tuned for different hardware configurations and workload profiles?

4. **Memory pressure callbacks**: Should SpyreAllocator support registering callbacks that fire when device memory is under pressure (e.g., when all regions are locked and fragmentation is high)? This could enable higher-level components to evict cached data or trigger garbage collection.

5. **Compaction / defragmentation**: Should the allocator support a `compact()` operation that relocates live blocks to eliminate fragmentation? This would require cooperation with all components holding `VirtualAddress` references to the moved blocks.

6. **Multi-device allocation**: The current design assumes a single device. How should SpyreAllocator extend to multi-device scenarios? Should there be one allocator per device, or a single allocator managing multiple devices?

7. **Interaction with ExecutionPlan binary loading**: When ExecutionPlan binaries are loaded to device, they consume SpyreAllocator blocks. Should these allocations be treated differently from tensor allocations (e.g., pinned to prevent eviction, allocated in a separate pool)?

8. **Eliminating try-catch for allocation attempts**: The current implementation uses try-catch around `DeviceMemoryAllocator` calls to handle allocation failures during region acquisition. Should this be replaced with a non-throwing API (e.g., `tryAllocate` returning `std::optional`)?

9. **`VirtualAddress.region_id` as `DeviceMemoryAllocationPtr`**: `DeviceMemoryAllocationPtr` already encapsulates the PF/VF distinction — it carries either the VF firmware table index or the PF physical address. Should `region_id` be a `DeviceMemoryAllocationPtr` rather than a derived integer? This would eliminate the separate extraction step and remove the need for `data` as a separate field on `MemoryRegion` (since the handle would be embedded in every `VirtualAddress`). The tradeoff is that `VirtualAddress` becomes a shared-ownership type rather than a lightweight value type — every live `VirtualAddress` would keep the entire region's device allocation alive via refcount, which could complicate future region release policies.

10. **Pre-allocated fixed region pool**: Should the allocator support allocating all regions up front at startup rather than on demand? This would eliminate allocation-time region acquisition latency and simplify the state machine (no `regions_locked` transition), but wastes device memory if the workload doesn't need all regions. The optimal number and size of regions depends on the workload, which isn't known at startup.

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
