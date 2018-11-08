import torch

import torchex.zoo as exzoo

def test_TransformerNet():
    x = torch.randn(10, 10, 28, 28)
    net = exzoo.ImageTransformerNet(10)
    y = net(x)
    assert list(y.shape) == [10, 10, 28, 28]
