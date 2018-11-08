import unittest

import torch

import torchex.nn as exnn

class TestHighway(unittest.TestCase):
    def test_foward(self):
        x = torch.randn(1, 3)
        nn = exnn.Highway(3)
        y = nn(x)
        self.assertEqual(list(y.shape), [1, 3])

if __name__ == '__main__':
    unittest.run()
