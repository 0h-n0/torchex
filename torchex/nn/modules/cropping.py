import torch
import torch.nn as nn

from ..functional import (crop_2d,
                         crop_3d)

class Crop2d(nn.Module):
    """
    :params torch.Tensor input: Input(B, C, W, H)
    """    
    def __init__(self,
                 crop_left: int=0, crop_right: int=0,
                 crop_top: int=0, crop_bottom: int=0):
        super(Crop2d, self).__init__()
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def forward(self, input):
        return crop_2d(input,
                       self.crop_left,
                       self.crop_right,
                       self.crop_top,
                       self.crop_bottom)
        
class Crop3d(nn.Module):
    """
    :params torch.Tensor input: Input(B, C, D, W, H)
    """    
    def __init__(self, crop_size):
        super(Crop3d, self).__init__()
        self.crop_size = crop_size

    def forward(self, input):
        return crop_3d(input, self.crop_size)
        
