import unittest

import torch
import pytest

import torchex.nn as exnn


class TesTPeriodicPad2d(unittest.TestCase):
    def test_output_shape(self):
        x = torch.randn(4, 4, 10, 10)
        pad = exnn.PeriodicPad2d(2, 2, 2, 2)
        y = pad(x)
        self.assertEqual(list(y.shape), [4, 4, 14, 14])

        x = torch.randn(4, 4, 10, 10)
        pad = exnn.PeriodicPad2d(2, 1, 2, 2)
        y = pad(x)
        self.assertEqual(list(y.shape), [4, 4, 14, 13])

        x = torch.randn(4, 4, 10, 10)
        pad = exnn.PeriodicPad2d(1, 1, 2, 2)
        y = pad(x)
        self.assertEqual(list(y.shape), [4, 4, 14, 12])

        x = torch.randn(4, 4, 10, 10)
        pad = exnn.PeriodicPad2d(1, 1, 2, 3)
        y = pad(x)
        self.assertEqual(list(y.shape), [4, 4, 15, 12])

        x = torch.randn(4, 4, 10, 10)
        pad = exnn.PeriodicPad2d(1, 1, 3, 3)
        y = pad(x)
        self.assertEqual(list(y.shape), [4, 4, 16, 12])
    
    def test_check_size_after_forward(self):
        pass


class TesTPeriodicPad3d(unittest.TestCase):
    def test_output_shape(self):
        x = torch.randn(4, 4, 10, 10, 10)
        pad = exnn.PeriodicPad3d(2)
        y = pad(x)
        self.assertEqual(list(y.shape), [4, 4, 14, 14, 14])


if __name__ == '__main__':
    unittest.run()
