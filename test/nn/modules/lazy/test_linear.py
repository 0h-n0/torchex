import pytest
import torch
import torchex.nn as exnn

def test_linear():
    net = exnn.Linear(3)
    x = torch.randn(10, 20)
    y = net(x)
    assert list(y.shape) == [10, 3]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU')
def test_cuda_linear():
    net = exnn.Linear(3).to('cuda')
    x = torch.randn(10, 20).to('cuda')
    y = net(x)
    assert list(y.shape) == [10, 3]
    
