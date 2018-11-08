import torch

import torchex.nn as exnn


def test_cordconv():
    x = torch.randn(10, 2, 28, 28)
    c = exnn.CordConv2d(2, 3, 1)
    y = c(x)
    assert list(y.shape) == [10, 3, 28, 28]

def test_cordconv_with_r():
    x = torch.randn(10, 2, 28, 28)
    c = exnn.CordConv2d(2, 3, 1, with_r=True)
    y = c(x)
    assert list(y.shape) == [10, 3, 28, 28]
    
