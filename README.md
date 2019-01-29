[![GitHub license](https://img.shields.io/github/license/0h-n0/torchex.svg)](https://github.com/0h-n0/torchex)
[![PYTHON version](https://img.shields.io/badge/python-3.5,3.6-blue.svg)](https://github.com/0h-n0/torchex)
[![PyPI version](https://img.shields.io/pypi/v/torchex.svg)](https://badge.fury.io/py/torchex)
[![CircleCI](https://circleci.com/gh/0h-n0/torchex.svg?style=svg&circle-token=99e93ba7bf6433d0cd33adbec2fbd042d141353d)](https://circleci.com/gh/0h-n0/torchex)
[![Build Status](https://travis-ci.org/0h-n0/torchex.svg?branch=master)](https://travis-ci.org/0h-n0/torchex)
[![codecov](https://codecov.io/gh/0h-n0/torchex/branch/master/graph/badge.svg)](https://codecov.io/gh/0h-n0/torchex)
[![Documentation Status](https://readthedocs.org/projects/torchex/badge/?version=latest)](https://torchex.readthedocs.io/en/latest/?badge=latest)
[![Maintainability](https://api.codeclimate.com/v1/badges/7cd6c99f10d22db13ee8/maintainability)](https://codeclimate.com/github/0h-n0/torchex/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/7cd6c99f10d22db13ee8/test_coverage)](https://codeclimate.com/github/0h-n0/torchex/test_coverage)
[![BCH compliance](https://bettercodehub.com/edge/badge/0h-n0/torchex?branch=master)](https://bettercodehub.com/)
[![Downloads](https://img.shields.io/pypi/dm/torchex.svg)](https://pypi.org/project/torchex/)

# (WIP) `torchex library

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

