import torch

import torchex.nn as exnn


def test_dft2d_norm():
    x = torch.randn(10, 2, 3, 3)
    dft = exnn.DFT2d()
    y = dft(x)
    assert y.shape == x.shape

def test_dft2d_concatinate():
    x = torch.randn(10, 2, 3, 3)
    dft = exnn.DFT2d(norm=False, concatinate=True)
    y = dft(x)
    assert list(y.shape) == [10, 2, 3, 6]

def test_dft2d_with_no_aditional_options():
    x = torch.randn(10, 2, 3, 3)
    dft = exnn.DFT2d(norm=False, concatinate=False, normlized=False)
    y = dft(x)
    assert list(y.shape) == [10, 2, 3, 3, 2]
    
