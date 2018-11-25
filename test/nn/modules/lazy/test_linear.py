import torch
import torchex.nn as exnn

def test_linear():
    net = exnn.Linear(3)
    x = torch.randn(10, 20)
    y = net(x)
    assert list(y.shape) == [10, 3]
