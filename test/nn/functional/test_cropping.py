import torch

from torchex.nn.functional import (crop_2d, crop_3d)


def test_crop_2d():
    x = torch.randn(4, 4, 10, 10)
    cropped_x = crop_2d(x, 2, 2, 2, 2)
    assert list(cropped_x.shape) == [4, 4, 6, 6]
    

def test_crop_3d():
    x = torch.randn(4, 4, 10, 10, 10)
    cropped_x = crop_3d(x, 2)
    assert list(cropped_x.shape) == [4, 4, 6, 6, 6]    
