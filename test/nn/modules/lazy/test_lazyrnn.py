import pytest
import torch
import torchex.nn as nn


def test_lazyrnnbasse():
    net = nn.LazyRNNBase('LSTM', 10)
    x = torch.randn(3, 20, 10)
    y = net(x)
