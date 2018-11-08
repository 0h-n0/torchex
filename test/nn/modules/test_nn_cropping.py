import unittest

import torch
import pytest

import torchex.nn as exnn


class TestCrop2d(unittest.TestCase):
    def test_output_shape(self):
        x = torch.randn(4, 4, 10, 10)
        pad = exnn.Crop2d(2, 2, 2, 2)
        y = pad(x)
        self.assertEqual(list(y.shape), [4, 4, 6, 6])


class TestCrop3d(unittest.TestCase):
    def test_output_shape(self):
        x = torch.randn(4, 4, 10, 10, 10)
        pad = exnn.Crop3d(2)
        y = pad(x)
        self.assertEqual(list(y.shape), [4, 4, 6, 6, 6])
