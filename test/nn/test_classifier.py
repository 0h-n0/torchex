import unittest

import torch
import torch.nn as nn

import torchex.nn as exnn

def test_foward():
    x = torch.randn(1, 3)
    y = torch.randn(1, 10)
    n = nn.Linear(3, 10)
    loss = torch.nn.MSELoss()
    c = exnn.Classifier(n, loss)
    loss = c(x, y)
    assert loss.dtype == torch.float32


if __name__ == '__main__':
    unittest.run()
