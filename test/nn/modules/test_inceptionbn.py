import unittest

import torch

import torchex.nn as exnn

class TestHighway(unittest.TestCase):
    def test_foward(self):
        x = torch.randn(1, 3, 10, 10)
        nn = exnn.InceptionBN(3, 10, 10 ,10, 10, 10, 'avg', 10)
        y = nn(x)
        self.assertEqual(list(y.shape), [1, 40, 10, 10])

if __name__ == '__main__':
    unittest.run()
