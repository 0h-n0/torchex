import torch

import torchex.nn as exnn


def test_foward_mlpconv():
    a = torch.randn(2, 2, 32, 32)
    mlpconv2d = exnn.MLPConv2d(2, [30, 20, 10], 5)
    x = mlpconv2d(a)
    assert list(x.shape) == [2, 10, 28, 28]

def test_foward_mlpconv():
    x = torch.randn(2, 2, 32, 32)
    unsampconv = exnn.UpsampleConvLayer(2, 10)
    x = unsampconv(x)
    assert list(x.shape) == [2, 10, 32, 32]    

