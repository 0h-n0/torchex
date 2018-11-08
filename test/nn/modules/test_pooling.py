import unittest

import torch
import pytest

import torchex.nn as exnn


class TestGlobalAvgPool1d(unittest.TestCase):
    def test_check_size_after_forward(self):
        a = torch.randn(2, 2, 2)
        g = exnn.GlobalAvgPool1d()
        x = g(a)
        self.assertEqual([2, 2, 1], list(x.shape))

class TestGlobalAvgPool2d(unittest.TestCase):
    def test_check_size(self):
        a = torch.randn(2, 2, 2, 2)
        g = exnn.GlobalAvgPool2d()
        x = g(a)
        self.assertEqual([2, 2, 1, 1], list(x.shape))

class TestGlobalMaxPool1d(unittest.TestCase):
    def test_check_size_after_forward(self):
        a = torch.randn(2, 2, 2)
        g = exnn.GlobalMaxPool1d()
        x = g(a)
        self.assertEqual([2, 2, 1], list(x.shape))

class TestGlobalMaxPool2d(unittest.TestCase):
    def test_check_size(self):
        a = torch.randn(2, 2, 2, 2)
        g = exnn.GlobalMaxPool2d()
        x = g(a)
        self.assertEqual([2, 2, 1, 1], list(x.shape))
        

class TestMaxAvgPool2d(unittest.TestCase):
    def test_check_size(self):
        pass
    

if __name__ == '__main__':
    unittest.run()
