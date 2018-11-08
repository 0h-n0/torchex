import torch
import torch.nn as nn

from torchex.attribute import IntegratedGradients

def test_IntegratedGradients():
    x = torch.randn((1, 10), requires_grad=True)
    net = nn.Linear(10, 10)
    ig = IntegratedGradients(net)
    y = ig(x, 1)
    assert x.shape == y.shape

