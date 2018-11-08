import pytest
import torch

from torchex.utils import (to_complex,
                           to_real,
                           complex_norm)

def test_to_complex_1d():
    x = torch.range(1, 10).view(1, 1, 10)
    y = to_complex(x)
    assert list(y.shape) == [1, 1, 10, 2]
    assert y[:, :, :, 0].shape == x.shape
    assert y[:, :, :, 1].shape == x.shape
    for i in range(1, 11):
        assert int(y[0, 0, i-1, 0]) == i
        assert int(y[0, 0, i-1, 1]) == 0        

def test_to_complex_2d():
    x = torch.range(1, 25).view(1, 1, 5, 5)
    y = to_complex(x)
    assert list(y.shape) == [1, 1, 5, 5, 2]
    assert y[:, :, :, :, 0].shape == x.shape
    assert y[:, :, :, :, 1].shape == x.shape
    idx = 1    
    for i in range(5):
        for j in range(5):
            assert int(y[0, 0, i, j, 0]) == idx
            assert int(y[0, 0, i, j, 1]) == 0
            idx += 1

def test_to_complex_3d():
    x = torch.range(1, 125).view(1, 1, 5, 5, 5)
    y = to_complex(x)
    assert list(y.shape) == [1, 1, 5, 5, 5, 2]
    assert y[:, :, :, :, :, 0].shape == x.shape
    assert y[:, :, :, :, :, 1].shape == x.shape
    idx = 1    
    for i in range(5):
        for j in range(5):
            for k in range(5):            
                assert int(y[0, 0, i, j, k, 0]) == idx
                assert int(y[0, 0, i, j, k, 1]) == 0
                idx += 1
    

def test_to_real():
    x = torch.range(1, 20).view(1, 1, 10, 2)
    y = to_real(x)
    assert list(y.shape) == [1, 1, 10]
    for idx, ele in enumerate(range(1, 20, 2)):
        assert int(y[0, 0, idx]) == ele

        
def test_complex_norm():
    x = torch.range(1, 20).view(1, 1, 10, 2)
    y = complex_norm(x)    
    assert list(y.shape) == [1, 1, 10]    
    out = [  2.2361,   5.0000,   7.8102,  10.6301,  13.4536,  16.2788,
     19.1050,  21.9317,  24.7588,  27.5862]
    for i in range(10):
        assert pytest.approx(float(y.data[0, 0, i]), 1e-4) ==  out[i]
    
