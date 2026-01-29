# Overview

This document complements the [Tiled Tensor RFC](../RFCs/0047-TiledTensors/0047-TiledTensorsRFC.md)
by precisely describing the specific device memory layouts used for Tensors in Torch-Spyre.

As detailed in more detail in the Motivation section of the Tiled Tensor RFC, the memory layout used on
Spyre is more complex than PyTorch's standard stride-based layouts.  Tensor entries are
grouped into *sticks* of 128 bytes. The sticks of a tensor are linearized in device memory using
layout algorithm that results in a tiling of the dimensions.  All Spyre Tensors carry metadata
that can be accessed at runtime that describes their layout.

The runtime defines a default memory layout for Tensors, often referred to as the "generic stick" layout.
This default layout is used for all Tensors created on the device (via `to` or similar APIs)
unless overridden by providing a `SpyreTensorLayout` as an explicit argument to the operation that creates
the Tensor.  Conceptually the default layout (a) pads all dimensions to be evenly divisible into sticks
(b) designates the last dimension as the stick dimension and (c) tiles along the first dimension.

# Runtime Details

The layout metadata is encoded by the runtime class `SpyreTensorLayout` (see [spyre_tensor_impl.h](../torch_spyre/csrc/spyre_tensor_impl.h)).
It can be accessed in Python via an added Tensor method `device_tensor_layout()`.
The key elements of metadata are:
+ `device_size`: analagous to PyTorch's `size` but with padded values an extra dimensions for tiling
+ `dim_map`: a vector of the same length as `device_size` giving the index in the PyTorch `size` array for each element of `device_size`
+ `format`: either `Dense` (normal case) or `Sparse` (resulting from reductions in the stick dimension).
+ `device_dtype`: the datatype of the Tensor.

As a concrete example, run the following program:

```
import torch
x = torch.rand(100, 200, , dtype=torch.float16)
y = x.to("spyre")
stl = y.device_tensor_layout()
print(stl)
```

You should see something like:

```
SpyreTensorLayout(device_size=[7, 4, 3, 64], dim_map =[1, 2, 0, 2], num_stick_dims=1, format=StickFormat.Dense, device_dtype=DataFormats.SEN169_FP16)
```

The 3-D tensor has a 4-D `device_size`.  A float16 is two bytes, therefore each stick contains

The `4` and `32`
