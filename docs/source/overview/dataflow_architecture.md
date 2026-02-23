# Dataflow Accelerator Architecture

This document provides a reference overview of the dataflow accelerator
model as implemented in the IBM Spyre AI Card. It is intended both as
context for Torch-Spyre developers and as a general reference for the
dataflow accelerator design pattern.

## What is a Dataflow Accelerator?

Traditional von Neumann processors execute instructions sequentially,
fetching data from memory on demand. **Dataflow accelerators** invert
this model: computation is expressed as a graph of operations, and
each operation fires as soon as its input data is available. This
eliminates most control-flow overhead and enables deeply pipelined,
high-throughput execution.

Key characteristics:
- Operations are scheduled by data availability, not program counters
- Tensors are staged in local scratchpad memories close to compute units
- The compiler is responsible for all data movement and scheduling

## Spyre Architecture Highlights

:::{figure} https://research-website-prod-cms-uploads.s3.us.cloud-object-storage.appdomain.cloud/IMAGE_4_FOR_MEDIA_Telum_II_Spyre_Chip_ddfc9615b8.png
:alt: Telum II processor and IBM Spyre Accelerator chip comparison
:width: 680px
:align: center

The IBM Telum II processor (left) and Spyre Accelerator chip (right), illustrating the architectural generations of IBM's AI acceleration platform. *Image credit: [IBM Research](https://research.ibm.com/blog/spyre-for-z).*
:::

| Feature | Detail |
|---------|--------|
| Cores | 32 AI accelerator cores |
| Technology | 5 nm |
| Memory per card | Up to 128 GB LPDDR5 |
| Peak performance | >300 TOPS per card |
| Power envelope | 75 W per card |
| Max ensemble | 8 cards / 1 TB memory |

## Memory Hierarchy

:::{figure} https://research-website-prod-cms-uploads.s3.us.cloud-object-storage.appdomain.cloud/IBM_AIU_PCIE_05_d6a1bd0d18.jpg
:alt: IBM AIU PCIe card — reverse side
:width: 560px
:align: center

The IBM Spyre Accelerator PCIe card (reverse side), showing the physical form factor for IBM Z and Power systems. *Image credit: [IBM Research](https://research.ibm.com/blog/spyre-for-z).*
:::

Spyre exposes two levels of memory visible to the compiler:

1. **DDR (device DRAM)** — large, off-core storage for full tensors.
2. **LX Scratchpad** — fast, on-core storage for tiles actively being
   processed. The compiler is responsible for explicit DMA transfers
   between DDR and scratchpad.

Data flows from the host → DDR → scratchpad → compute units → scratchpad
→ DDR → host. The front-end compiler generates the DCI (Data Copy
Instructions) and SuperDSC specifications that drive this pipeline.

## Execution Model

Each Spyre core executes a **kernel** — a self-contained computation
on a tile of data. The compiler determines:

1. **Work division** — how to split a tensor operation across cores
   (see [Work Division Planning](../compiler/work_division_planning.md))
2. **Data staging** — when and how to DMA tiles into scratchpad
3. **Kernel specification** — the SuperDSC JSON describing the operation

Cores execute in SPMD (Single Program, Multiple Data) fashion: all
cores run the same program but on different data tiles identified by
their core ID.

:::{figure} ../_static/images/spyre-core-microarchitecture.png
:alt: Spyre core microarchitecture showing PT units, PE, SFP, LX Scratchpad, and HBM
:width: 85%
:align: center

Spyre core microarchitecture. An array of Processing Threads (PT) feeds through a Processing Element (PE) and Special Function Processor (SFP). Computed data is staged in fast on-core **LX Scratchpad** memory; the **HBM** (High-Bandwidth Memory) sits below and is the primary bandwidth bottleneck. Custom kernels can optimize tiling, fusion, and LX usage to reduce HBM pressure. *Source: Torch-Spyre contributor presentation.*
:::

## Comparison with GPU Execution

| Aspect | GPU (CUDA) | Spyre Dataflow |
|--------|-----------|----------------|
| Scheduling | Warp-level SIMT | Data-driven, core SPMD |
| Memory model | Shared memory + global | Scratchpad + DDR |
| Data movement | Implicit caching | Explicit DMA via compiler |
| Parallelism granularity | Thread blocks | Core tiles |

## Further Reading

- [IBM Spyre Accelerator Overview](spyre_accelerator.md)
- [Compiler Architecture](../compiler/architecture.md)
- [Tensor Layouts](../user_guide/tensors_and_layouts.md)
