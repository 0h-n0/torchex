[![PYTHON version](https://img.shields.io/badge/python-3.6,3.7,3.8-blue.svg)](https://github.com/0h-n0/torchex)
[![PyPI version](https://img.shields.io/pypi/v/torchex.svg)](https://badge.fury.io/py/torchex)
[![Downloads](https://img.shields.io/pypi/dm/torchex.svg)](https://pypi.org/project/torchex/)

# (WIP) `torchex library`

`torchex` library provides advanced Neural Network Layers. You can easily use them like using original pytorch.

## Installation

```
$ pip install torchex
```

## Requirements

* Pytorch >= 1.0

## Documentation

* https://torchex.readthedocs.io/en/latest/index.html

## How to use

### Lazy Style Model Definition

```python
import torch
import torchex.nn as exnn

net = exnn.Linear(10)
# You don't need to give the size of input for this module.
# This network is equivalent to `nn.Linear(100, 10)`.

x = troch.randn(10, 100)

y = net(x)
```

### torchex.nn list

* `torchex.nn.Pass`
* `torchex.nn.Flatten`
* `torchex.nn.Linear`
  * Lazy style
* `torchex.nn.Conv1d`
  * Lazy style
* `torchex.nn.Conv2d`
  * Lazy style
* `torchex.nn.Conv3d`
  * Lazy style
* `torchex.nn.Conv2dLocal`
* `torchex.nn.GlobalAvgPool1d`
* `torchex.nn.GlobalAvgPool2d`
* `torchex.nn.GlobalMaxPool1d`
* `torchex.nn.GlobalMaxPool2d`
* `torchex.nn.MaxAvgPool2d`
* `torch.nn.Crop2d`
* `torch.nn.Crop3d`
* `torch.nn.MLPConv2d`
* `torch.nn.UpsampleConvLayer`
* `torch.nn.CordConv2d`
* `torch.nn.DFT1d`
* `torch.nn.DFT2d`
* `torch.nn.PeriodicPad2d`
* `torch.nn.PeriodicPad3d`
* `torch.nn.Highway`
* `torch.nn.Inception`
* `torch.nn.InceptionBN`
* `torch.nn.IndRNNCell`
* `torch.nn.IndRNNTanhCell`
* `torch.nn.IndRNNReLuCell`
* `torch.nn.IndRNN`
* `torch.nn.GraphLinear`
* `torch.nn.GraphConv`
* `torch.nn.SparseMM`
* `torch.nn.GraphBatchNrom`

### torchex.data.transforms

* `torchex.data.transforms.PadRandomSift`
* `torchex.data.transforms.RandomResize`

### torchex.data.attribute

for visualization

* `torchex.attribute.IntegratedGradients`
