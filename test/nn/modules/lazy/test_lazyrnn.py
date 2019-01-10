import pytest
import torch
import torchex.nn as nn

@pytest.mark.skip(reason="WIP: pytorch 0.4 => 1.0")
def test_lazyrnnbasse():
    net = nn.LazyRNNBase('LSTM', 10)
    x = torch.randn(3, 20, 10)
    y = net(x)
