import unittest

import torch
import pytest

import torchex as ex


class TestFlatten(unittest.TestCase):
    def test_output_shape(self):
        a = torch.randn(2, 2, 2, 2)
        g = ex.nn.Flatten()
        x = g(a)
        self.assertEqual([2, 8], list(x.shape))        

if __name__ == '__main__':
    unittest.run()
    
