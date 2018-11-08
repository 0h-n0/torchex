import torch
import torch.nn as nn

class PeriodicPad2d(nn.Module):
    """

    :params torch.Tensor input: Input(B, C, W, H)

    # https://github.com/ZichaoLong/aTEAM/blob/master/nn/functional/utils.py
    """    
    def __init__(self,
                 pad_left: int=0, pad_right: int=0,
                 pad_top: int=0, pad_bottom: int=0):
        super(PeriodicPad2d, self).__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom

    def forward(self, input):
        assert input.dim() == 4, 'only support Input(B, C, W, H) or Input(B, C, H, W)'
        
        B, C, H, W = input.size()
        
        left_pad = input[:, :, :, W-(self.pad_left):]
        right_pad = input[:, :, :, :self.pad_right]
        input = torch.cat([left_pad, input, right_pad], dim=3)
        top_pad = input[:, :, H-(self.pad_top):, :]
        bottom_pad = input[:, :, :self.pad_bottom, :]
        input = torch.cat([top_pad, input, bottom_pad], dim=2)
        return input


class PeriodicPad3d(nn.Module):
    '''
    Only support isotropic padding
    '''
    def __init__(self, pad: int=0):
        super(PeriodicPad3d, self).__init__()
        self.pad = pad
        
    def forward(self, input):
        '''
        :params torch.Tensor input: Input(B, C, D, W, H)    
        '''
        assert input.dim() == 5, 'only support Input(B, C, D, W, H)'
        
        B, C, D, H, W = input.size()        
        pad_0 = input[:, :, D-(self.pad):, :, :]
        pad_1 = input[:, :, :self.pad, :, :]
        input = torch.cat([pad_0, input, pad_1], dim=2)
        pad_0 = input[:, :, :, H-(self.pad):, :]
        pad_1 = input[:, :, :, :self.pad, :]
        input = torch.cat([pad_0, input, pad_1], dim=3)
        pad_0 = input[:, :, :, :, W-(self.pad):]
        pad_1 = input[:, :, :, :, :self.pad]
        input = torch.cat([pad_0, input, pad_1], dim=4)
        return input

if __name__ == '__main__':

    x = torch.range(1, 25).view(1, 1, 5, 5)
    print(x)    
    pad = PeriodicPad2d(2, 2, 2, 1)
    print(pad(x))
    print(pad(x).shape)    
    
    x = torch.range(1, 27).view(1, 1, 3, 3, 3)
    pad = PeriodicPad3d(1)
    print(pad(x))
