import unittest

import torch

import torchex.nn as exnn

class TestConv2dLocal(unittest.TestCase):
    def test_foward(self):
        x = torch.randn(1, 3, 10, 10)
        nn = exnn.Conv2dLocal(3, 10, 1)


if __name__ == '__main__':
    unittest.run()
