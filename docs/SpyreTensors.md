# Overview

This document complements the [Tiled Tensor RFC](../RFCs/0047-TiledTensors/0047-TiledTensorsRFC.md)
by describing the specific device memory layouts and related APIs used for Tensors in Torch-Spyre.

As detailed in more detail in the Motivation section of the Tiled Tensor RFC, the memory layout used on
Spyre is more complex than PyTorch's standard stride-based layouts.  Tensor entries are
grouped into *sticks* of 128 bytes. The sticks of a tensor are linearized in device memory using
a layout algorithm that results in a tiling of the dimensions.  All Spyre Tensors carry metadata
that describes their layout. The metadata can be accessed via both C++ and Python APIs and is
used by the compiler to specialize a compiled graph to its inputs.

The runtime defines a default memory layout for Tensors, often referred to by Torch-Spyre developers
as the "generic stick" layout.
This default layout is used for all Tensors created on the device
unless overridden by providing a `SpyreTensorLayout` as an explicit argument to the operation that creates
the Tensor.  Conceptually the default layout (a) pads all dimensions to be evenly divisible into sticks
(b) designates the last dimension as the stick dimension and (c) tiles along the first dimension.

# Details and Default Layout

The layout metadata is encoded by the runtime C++ class `SpyreTensorLayout` (see [spyre_tensor_impl.h](../torch_spyre/csrc/spyre_tensor_impl.h)).
An instance of this class is embedded as a field in the `SpyreTensorImpl` class.
It can be accessed in Python via an added Tensor method `device_tensor_layout()`.
The key elements of metadata are:
+ `device_size`: analagous to PyTorch's `size` but with padded values an extra dimensions for tiling.
+ `dim_map`: a vector of the same length as `device_size` giving the index in the PyTorch `size` array for each element of `device_size`.
+ `format`: either `Dense` (normal case) or `Sparse` (resulting from reductions in the stick dimension).
+ `device_dtype`: the datatype of the Tensor.

As a concrete example, run the following program:

```
import torch
x = torch.rand(5, 100, 150, dtype=torch.float16)
y = x.to("spyre")
stl = y.device_tensor_layout()
print(stl)
```

You should see something like:

```
SpyreTensorLayout(device_size=[128, 3, 64, 64], dim_map =[1, 2, 0, 2], num_stick_dims=1, format=StickFormat.Dense, device_dtype=DataFormats.SEN169_FP16)
```

The 3-D tensor has a 4-D `device_size`.  
A float16 is two bytes, therefore each stick contains 64 data values.
The stick dimension of `150` has been padded to `192` and broken into two device dimensions of (`3` and `64`).
The non-stick dimensions of `5` and `100` have been padded to `64` and `128` respectively.
The total device memory allocated for the tensor is 3MB (`128*3*64 = 24,576` 128 byte sticks).
This 3MB of memory contains only 150,000 bytes of real data; the remainder of the memory is padding
that ensures that the data values are arranged in a fashion that will enable the tensor to be used
as an input to any legal sequence of compute operations.

# Controlling Layout Decisions

The runtime provides APIs that enable both the compiler and the expert programmer
to control the memory layout of tensors. When using these APIs, the programmer
must be aware of any layout constraints imposed by the compute operations that
use the tensors as input values.

All of the APIs work by first constructing a `SpyreTensorLayout` the describes the desired
memory layout. The `SpyreTensorLayout` is then passed as a keyword argument to `to`,
`empty_strided` or similar torch function.

## Explicit API with default layout

The minimal constructor for a `SpyreTensorLayout` takes a `size` and `dtype` and
builds a instance that encodes the default generic stick layout.  This constructor
is what is used behind the scenes when the user does not specify a layout.

As an example, we can explictly request the default layout in a `to` by doing:

```
import torch
from torch_spyre._C import SpyreTensorLayout
x = torch.rand(5, 100, 150, dtype=torch.float16)
stl = SpyreTensorLayout((5, 100, 150), torch.float16)
y = x.to("spyre",device_layout=stl)
print(y.device_tensor_layout())
```

You should see exactly the same output as before:

```
SpyreTensorLayout(device_size=[128, 3, 64, 64], dim_map =[1, 2, 0, 2], num_stick_dims=1, format=StickFormat.Dense, device_dtype=DataFormats.SEN169_FP16)
```

## Disabling Non-stick Padding

Padding of all non-stick dimensions can be disabled via a boolean flag.

```
import torch
from torch_spyre._C import SpyreTensorLayout
x = torch.rand(5, 100, 150, dtype=torch.float16)
stl = SpyreTensorLayout((5, 100, 150), torch.float16, pad_all_dims=False)
y = x.to("spyre",device_layout=stl)
print(y.device_tensor_layout())
```

You should see:

```
SpyreTensorLayout(device_size=[100, 3, 5, 64], dim_map =[1, 2, 0, 2], num_stick_dims=1, format=StickFormat.Dense, device_dtype=DataFormats.SEN169_FP16)
```

Only the stick dimension has been padded, resulting in a total allocation of 187.5KB (1500 sticks).

The memory footprint of this tensor is greatly reduced, but it may not be in a supported input
format for some compute operations.

## Fine-grained control of padding and dimension order

A second constructor of `SpyreTensorLayout` enables finer-grained control.
It takes a `padded_size` and `dim_order` allowing the programmer or compiler
to fine-tune the layout based on their knowledge of how the Tensor will be used
in computation.

For example,

```
import torch
from torch_spyre._C import SpyreTensorLayout
x = torch.rand(5, 100, 150, dtype=torch.float16)
stl = SpyreTensorLayout((5, 128, 192), torch.float16, [0,1,2])
y = x.to("spyre",device_layout=stl)
print(y.device_tensor_layout())
```

Yields a tensor padded in the second and third dimension.

```
SpyreTensorLayout(device_size=[128, 3, 5, 64], dim_map =[1, 2, 0, 2], num_stick_dims=1, format=StickFormat.Dense, device_dtype=DataFormats.SEN169_FP16)
```

Changing the constructor in the above program to

```
stl = SpyreTensorLayout((5, 100, 192), torch.float16, [1,0,2])
```

Yields a tensor padded only in the third (stick) dimension with the tiling inverted

```
SpyreTensorLayout(device_size=[5, 3, 100, 64], dim_map =[0, 2, 1, 2], num_stick_dims=1, format=StickFormat.Dense, device_dtype=DataFormats.SEN169_FP16)
```

NOTE: When using the constructor that takes `padded_size` and `dim_order`,
the user must ensure whichever dimension is chosen as the stick dimension
is padded appropriately. An error will be raised if this requirement is not met.
Running the incorrect program below

```
import torch
from torch_spyre._C import SpyreTensorLayout
x = torch.rand(5, 100, 150, dtype=torch.float16)
stl = SpyreTensorLayout((5, 3, 7), torch.float16, [0,1,2])
y = x.to("spyre",device_layout=stl)
print(y.device_tensor_layout())
```

will result in a RuntimeError:

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Invalid padding: padded_size[stick_dim] not even multiple of elems_in_stick
```
