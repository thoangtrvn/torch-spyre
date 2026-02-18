# Tensor Dimension and Layout Concepts in Torch-Spyre

## Introduction

The torch-spyre compiler frontend transforms PyTorch operations into a form (SDSC - Spyre Device Specification Code) that the backend compiler can consume and execute on Spyre hardware. This transformation requires careful tracking of multiple representations of tensor dimensions and layouts as data flows from the high-level PyTorch abstraction through the compiler to the physical device execution.

This document describes five fundamental concepts that are central to understanding how torch-spyre represents and transforms tensor operations:

1. **Operation Dimensions** - The abstract iteration space of a computation
2. **Host Tensor Layout** - PyTorch's logical view of tensor organization
3. **Host Tensor Dimensions** - The individual axes of PyTorch tensors
4. **Device Tensor Layout** - Spyre's physical memory organization
5. **Device Tensor Dimensions** - The axes of the tiled device representation

Understanding these concepts and their relationships is essential for comprehending how torch-spyre compiles PyTorch operations for efficient execution on Spyre hardware.

## Operation Dimensions

**Operation dimensions** represent the abstract iteration space of a computation, independent of how input and output tensors are physically stored in memory. They define the logical dimensions that an operation conceptually works over.

### Characteristics

- **Abstract and Logical**: Operation dimensions describe what the computation does, not how data is laid out
- **Operation-Specific**: Different operations have different dimensional structures
- **Named Semantically**: Often given meaningful names like M, N, K for matrix operations or B for batch dimensions
- **Independent of Storage**: The same operation dimensions can work with tensors having different physical layouts

### Examples

**Matrix Multiplication**: A matrix multiplication of `[M, K] @ [K, N] -> [M, N]` has three operation dimensions:
- **M**: The number of rows in the first matrix and output
- **K**: The reduction dimension (columns of first matrix, rows of second)
- **N**: The number of columns in the second matrix and output

**Batched Matrix Multiplication**: A 3D batched matrix multiplication `[B, M, K] @ [B, K, N] -> [B, M, N]` has four operation dimensions:
- **B**: The batch dimension
- **M**: Rows in each matrix of the first batch
- **K**: The reduction dimension
- **N**: Columns in each matrix of the second batch

**Element-wise Operations**: An element-wise addition of two tensors of size `[128, 256, 512]` has three operation dimensions corresponding to the three axes being iterated over.

### Role in Compilation

Operation dimensions serve as the common reference frame for describing how different tensors participate in a computation. The compiler uses operation dimensions to:
- Determine which tensor dimensions correspond to which parts of the computation
- Identify broadcast dimensions (where a tensor has size 1 but the operation dimension is larger)
- Identify reduction dimensions (where an input dimension is summed away in the output)
- Plan work division across multiple cores

## Host Tensor Layout

**Host tensor layout** describes how PyTorch represents a tensor in memory. This is the standard PyTorch abstraction that programmers interact with, consisting of a size vector and a stride vector.

### Characteristics

- **PyTorch Standard**: Uses PyTorch's conventional representation with size and stride
- **Logical View**: Represents how the programmer thinks about the tensor
- **CPU Memory**: Corresponds to how data is organized in host (CPU) memory
- **Stride-Based**: Uses strides to map multi-dimensional indices to linear memory offsets
- **Canonical Form**: Dimensions of size 1 are typically eliminated (canonicalized) for consistency

### Components

**Size Vector**: Specifies the number of elements along each dimension. For example, a tensor with size `[128, 256, 512]` has 128 elements in dimension 0, 256 in dimension 1, and 512 in dimension 2.

**Stride Vector**: Specifies the memory offset between consecutive elements along each dimension. Row-major layout (PyTorch's default) has strides that decrease from left to right.

### Examples

**2D Tensor**: A float16 tensor with size `[1024, 256]` in row-major layout has:
- Size: `[1024, 256]`
- Stride: `[256, 1]` (moving one step in dimension 0 skips 256 elements, dimension 1 skips 1 element)

**3D Tensor**: A tensor with size `[128, 256, 512]` in row-major layout has:
- Size: `[128, 256, 512]`
- Stride: `[131072, 512, 1]` (dimension 0 stride = 256 × 512, dimension 1 stride = 512, dimension 2 stride = 1)

### Role in Compilation

The host tensor layout is the starting point for compilation. The compiler:
- Receives tensors with host layouts from PyTorch
- Uses host layouts to understand the logical structure of operations
- Maps host layouts to device layouts for efficient execution
- Ensures data transfers between host and device correctly transform between layouts

## Host Tensor Dimensions

**Host tensor dimensions** are the individual axes (dimensions) of a PyTorch tensor. Each dimension has an index and a size.

### Characteristics

- **Indexed from Zero**: Dimensions are numbered 0, 1, 2, ... up to rank-1
- **Each Has a Size**: The number of elements along that axis
- **Rank**: The total number of dimensions is called the tensor's rank
- **Semantic Meaning**: Often correspond to meaningful concepts (batch, height, width, channels, etc.)

### Examples

**2D Tensor**: A tensor of size `[1024, 256]` has:
- 2 host dimensions (rank = 2)
- Dimension 0: size 1024 (often represents rows)
- Dimension 1: size 256 (often represents columns)

**3D Tensor**: A tensor of size `[32, 128, 256]` has:
- 3 host dimensions (rank = 3)
- Dimension 0: size 32 (might represent batch)
- Dimension 1: size 128 (might represent height)
- Dimension 2: size 256 (might represent width)

**4D Tensor**: A tensor of size `[16, 32, 128, 256]` has:
- 4 host dimensions (rank = 4)
- Dimension 0: size 16 (might represent batch)
- Dimension 1: size 32 (might represent channels)
- Dimension 2: size 128 (might represent height)
- Dimension 3: size 256 (might represent width)

### Role in Compilation

Host tensor dimensions are the bridge between operation dimensions and device dimensions:
- The compiler maps operation dimensions to host dimensions to understand which tensor axes participate in which parts of the computation
- Host dimensions are then mapped to device dimensions to determine the physical layout
- This two-step mapping (operation → host → device) allows the compiler to handle broadcasting, reductions, and layout transformations

## Device Tensor Layout

**Device tensor layout** describes how a tensor is physically stored in Spyre device memory. This layout is optimized for Spyre's hardware architecture and can differ significantly from the host layout.

### Characteristics

- **Hardware-Optimized**: Designed for efficient access by Spyre's SIMD and systolic array architecture
- **Tiled**: Breaks dimensions into tiles for better memory locality and parallelism
- **Padded**: Adds padding to meet alignment requirements (128-byte stick boundaries)
- **Higher Rank**: Often has more dimensions than the host layout due to tiling
- **Stick-Oriented**: Organized around 128-byte sticks as the fundamental unit of memory access
- **Row-Major Device Order**: Always uses row-major ordering in device memory (no explicit strides)

### Components

**Device Size**: A vector specifying the size of each device dimension, including tiling and padding. This is always larger or equal to the host size in terms of total elements due to padding.

**Dim Map**: A vector that maps each device dimension back to its corresponding host dimension. This enables reconstruction of the host view from the device layout. Elements can be:
- Non-negative integers: map to a specific host dimension
- `-1`: represents a synthetic dimension (like the stick dimension) that doesn't correspond to any host dimension

**Device Dtype**: The data type format used on the device, which determines elements per stick.

**Stick Dimension**: The innermost dimension (always the last device dimension) that corresponds to the 128-byte stick granularity. Each stick contains a fixed number of elements based on the data type (e.g., 64 float16 values).

### Examples

**2D Tensor Tiling**: A host tensor of size `[1024, 256]` with float16 elements might have device layout:
- Device Size: `[4, 1024, 64]`
- Dim Map: `[1, 0, 1]`
- Interpretation: Host dimension 1 (the 256 columns) is split into 4 tiles of 64 elements each, forming device dimension 0. Host dimension 0 (1024 rows) becomes device dimension 1. Device dimension 2 is the stick dimension (64 float16 values per stick).

**3D Tensor with Padding**: A host tensor of size `[128, 256, 200]` with float16 elements might have device layout:
- Device Size: `[256, 4, 128, 64]`
- Dim Map: `[1, 2, 0, 2]`
- Interpretation: Host dimension 1 becomes device dimension 0. Host dimension 2 (200 elements) is padded to 256 and split into 4 tiles (device dimension 1) of 64 elements each. Host dimension 0 becomes device dimension 2. The last 56 elements of each stick in dimension 1 are padding.

### Role in Compilation

The device tensor layout is crucial for:
- **Memory Allocation**: Determining how much device memory to allocate
- **Data Transfer**: Transforming data between host and device representations during DMA operations
- **Code Generation**: Generating correct memory access patterns in device kernels
- **Work Division**: Splitting computation across multiple cores based on device dimensions
- **Layout Propagation**: Ensuring compatible layouts between operation inputs and outputs

## Device Tensor Dimensions

**Device tensor dimensions** are the individual axes of the device tensor layout. Due to tiling and padding, there are typically more device dimensions than host dimensions.

### Characteristics

- **Higher Rank**: Device rank ≥ host rank due to tiling
- **Tiling Dimensions**: Some device dimensions represent tiles of host dimensions
- **Stick Dimension**: The last device dimension is always the stick dimension
- **Mapped to Host**: Each device dimension maps back to a host dimension (or -1 for synthetic dimensions)
- **Contiguous Tiles**: Dimensions are ordered so that tiles are contiguous in memory

### Examples

**Simple Tiling**: For a 2D host tensor `[1024, 256]` with device layout `[4, 1024, 64]`:
- Device dimension 0: size 4 (tiles of host dimension 1)
- Device dimension 1: size 1024 (host dimension 0)
- Device dimension 2: size 64 (stick dimension, part of host dimension 1)

**Complex Tiling**: For a 3D host tensor `[128, 256, 512]` with device layout `[256, 8, 128, 64]`:
- Device dimension 0: size 256 (host dimension 1)
- Device dimension 1: size 8 (tiles of host dimension 2)
- Device dimension 2: size 128 (host dimension 0)
- Device dimension 3: size 64 (stick dimension, part of host dimension 2)

### Role in Compilation

Device tensor dimensions are used for:
- **Physical Memory Layout**: Determining the actual memory organization on the device
- **Access Pattern Generation**: Creating efficient memory access patterns in kernels
- **Core Division**: Splitting work across cores by dividing device dimensions
- **Stride Calculation**: Computing memory offsets for multi-dimensional access

## Relationships and Mappings

Understanding how these five concepts relate to each other is essential for comprehending the compilation process.

### Operation Dimensions ↔ Host Dimensions: The Scales Mapping

The **scales** mapping connects operation dimensions to host tensor dimensions for each tensor involved in an operation. This mapping is represented as a list of integers, one per operation dimension, where:

- **Non-negative value**: The host dimension index that corresponds to this operation dimension
- **-1**: This operation dimension is a broadcast dimension (tensor has size 1) or reduction dimension (dimension is summed away)
- **-3**: This operation dimension has size 1 and is elided from the device layout

**Example - Matrix Multiplication**: For `[M, K] @ [K, N] -> [M, N]` with operation dimensions (M, K, N):
- First input tensor `[M, K]` scales: `[0, 1, -1]` (op dim M → host dim 0, op dim K → host dim 1, op dim N is not in this tensor)
- Second input tensor `[K, N]` scales: `[-1, 0, 1]` (op dim M is not in this tensor, op dim K → host dim 0, op dim N → host dim 1)
- Output tensor `[M, N]` scales: `[0, -1, 1]` (op dim M → host dim 0, op dim K is reduced away, op dim N → host dim 1)

**Example - Batched Matrix Multiplication**: For `[B, M, K] @ [B, K, N] -> [B, M, N]` with operation dimensions (B, M, K, N):
- First input tensor `[B, M, K]` scales: `[0, 1, 2, -1]` (B→host dim 0, M→host dim 1, K→host dim 2, N is not in this tensor)
- Second input tensor `[B, K, N]` scales: `[0, -1, 1, 2]` (B→host dim 0, M is not in this tensor, K→host dim 1, N→host dim 2)
- Output tensor `[B, M, N]` scales: `[0, 1, -1, 2]` (B→host dim 0, M→host dim 1, K is reduced away, N→host dim 2)

**Example - Broadcasting**: For element-wise addition `[128, 1, 512] + [128, 256, 512] -> [128, 256, 512]`:
- First input scales: `[0, -1, 2]` (dimension 1 is broadcast, indicated by -1)
- Second input scales: `[0, 1, 2]` (all dimensions participate normally)
- Output scales: `[0, 1, 2]`

### Host Dimensions ↔ Device Dimensions: The Dim Map

The **dim_map** in the device tensor layout maps device dimensions back to host dimensions. This mapping enables:
- Reconstruction of the host view from device memory
- Understanding which device dimensions correspond to which logical dimensions
- Handling tiling where one host dimension becomes multiple device dimensions

**Example - 2D Tiling**: Host size `[1024, 256]`, device size `[4, 1024, 64]`, dim_map `[1, 0, 1]`:
- Device dimension 0 (size 4) → host dimension 1 (tiling the columns)
- Device dimension 1 (size 1024) → host dimension 0 (the rows)
- Device dimension 2 (size 64) → host dimension 1 (stick dimension, part of columns)

**Example - 3D Tiling**: Host size `[128, 256, 512]`, device size `[256, 8, 128, 64]`, dim_map `[1, 2, 0, 2]`:
- Device dimension 0 (size 256) → host dimension 1
- Device dimension 1 (size 8) → host dimension 2 (tiling)
- Device dimension 2 (size 128) → host dimension 0
- Device dimension 3 (size 64) → host dimension 2 (stick dimension)

### Complete Mapping Chain: Operation → Host → Device

The complete transformation from operation dimensions to physical device memory involves two mappings:

1. **Operation → Host**: Via the scales mapping, determining which host dimensions participate in which operation dimensions
2. **Host → Device**: Via the dim_map, determining how host dimensions are tiled and laid out on the device

**Example - Matrix Multiplication End-to-End**:

Consider `[1024, 512] @ [512, 256] -> [1024, 256]` with float16 elements:

**Operation Dimensions**: (M=1024, K=512, N=256) - 3 operation dimensions

**First Input (A)**:
- Host: size `[1024, 512]`, dimensions 0 and 1
- Scales: `[0, 1, -1]` (op dim M→host dim 0, op dim K→host dim 1, op dim N not in this tensor)
- Device: size `[8, 1024, 64]`, dim_map `[1, 0, 1]`
  - Device dim 0: tiles of host dim 1 (K dimension)
  - Device dim 1: host dim 0 (M dimension)
  - Device dim 2: stick dimension (part of K)

**Second Input (B)**:
- Host: size `[512, 256]`, dimensions 0 and 1
- Scales: `[-1, 0, 1]` (op dim M not in this tensor, op dim K→host dim 0, op dim N→host dim 1)
- Device: size `[4, 512, 64]`, dim_map `[1, 0, 1]`
  - Device dim 0: tiles of host dim 1 (N dimension)
  - Device dim 1: host dim 0 (K dimension)
  - Device dim 2: stick dimension (part of N)

**Output (C)**:
- Host: size `[1024, 256]`, dimensions 0 and 1
- Scales: `[0, -1, 1]` (op dim M→host dim 0, op dim K is reduced away, op dim N→host dim 1)
- Device: size `[4, 1024, 64]`, dim_map `[1, 0, 1]`
  - Device dim 0: tiles of host dim 1 (N dimension)
  - Device dim 1: host dim 0 (M dimension)
  - Device dim 2: stick dimension (part of N)

This complete mapping allows the compiler to:
- Understand that operation dimension M corresponds to device dimension 1 in all tensors
- Understand that operation dimension K is a reduction dimension
- Understand that operation dimension N corresponds to device dimension 0 (tiles) and 2 (stick) in B and C
- Generate correct memory access patterns and work division

## Compilation Context

These five concepts work together throughout the torch-spyre compiler frontend, which transforms PyTorch operations into SDSC (Spyre Device Specification Code) that the backend compiler consumes. The frontend compilation process consists of three main stages:

### Stage 1: PyTorch to Inductor IR

1. **Input**: PyTorch operations with host tensor layouts
2. **Process**:
   - Canonicalize host layouts (remove size-1 dimensions)
   - Determine operation dimensions from the operation semantics
   - Create initial scales mappings
3. **Output**: Inductor IR with FixedLayout (host representation)

### Stage 2: Layout Propagation and Validation

1. **Input**: Inductor IR with host layouts
2. **Process**:
   - Determine device layouts for input tensors (from example inputs or defaults)
   - Propagate device layouts through the computation graph
   - Upgrade FixedLayout to FixedTiledLayout (adding device_layout field)
   - Validate layout compatibility between operation inputs and outputs
3. **Output**: Inductor IR with FixedTiledLayout (host + device representation)

### Stage 3: SDSC Generation

1. **Input**: Inductor IR with complete layout information
2. **Process**:
   - Generate SDSC (Spyre Device Specification Code) using all five concepts:
     - Operation dimensions define the iteration space
     - Scales map operation dimensions to host dimensions
     - Host dimensions map to device dimensions via dim_map
     - Device dimensions determine physical memory layout
     - Device layout determines memory allocation and access patterns
   - Plan work division across cores using device dimensions
   - Generate data movement operations (DMA) using host-to-device mappings
3. **Output**: SDSC that the backend compiler can execute on Spyre hardware

### Key Compilation Challenges

**Layout Compatibility**: Ensuring that tensors with different device layouts can be used together in operations. Some operations require specific layout relationships (e.g., matrix multiplication inputs must have compatible tiling).

**Layout Propagation**: Determining the device layout of computed tensors based on input layouts and operation semantics. The compiler must choose layouts that are both valid and efficient.

**Memory Planning**: Calculating accurate memory requirements using device layouts (which include padding) rather than host layouts.

**Work Division**: Splitting computation across multiple cores by dividing device dimensions, which requires understanding the relationship between operation dimensions and device dimensions.

**Sparse Tensors**: Handling tensors where reduction operations produce one element per stick, requiring special device layouts with synthetic dimensions.

## Summary

The five concepts work together to enable efficient compilation of PyTorch operations for Spyre hardware:

1. **Operation Dimensions** provide the abstract computational framework
2. **Host Tensor Layout** and **Host Tensor Dimensions** represent PyTorch's logical view
3. **Device Tensor Layout** and **Device Tensor Dimensions** represent Spyre's physical organization
4. **Scales** map operation dimensions to host dimensions
5. **Dim Map** maps device dimensions to host dimensions

Together, these concepts enable the compiler to transform high-level PyTorch operations into efficient device code while maintaining correctness across multiple representations of the same data.

## References

- [Tiled Tensors RFC](../RFCs/0047-TiledTensors/0047-TiledTensorsRFC.md) - Detailed RFC on the motivation and design of tiled tensor layouts
- [Tensor Layouts Documentation](tensor_layouts.md) - Comprehensive guide to tensor layouts in torch-spyre
- [Compiler Architecture](compiler_architecture.md) - Overview of the torch-spyre compiler pipeline
- [Work Division Planning](work_division_planning.md) - How work is divided across cores using these concepts
