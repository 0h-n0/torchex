import torch
import torch.nn as nn

# TODO: implement 1d and 3d.

class CordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CordConv2d, self).__init__()
        self.with_r = with_r

        if self.with_r:
            in_channels += 3
        else:
            in_channels += 2
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        
    def forward(self, x):
        B, C, H, W = x.shape
        _c_H = torch.arange(0., H, dtype=x.dtype)
        _c_W = torch.arange(0., W, dtype=x.dtype)
        _c_H = torch.t(_c_H.repeat(W, B)).view(B, 1, H, W) / (H - 1)
        _c_W = torch.t(_c_W.repeat(H, B).t()).view(B, 1, H, W) / (W - 1)
        _c_H = _c_H * 2 - 1
        _c_W = _c_W * 2 - 1

        x = torch.cat([x, _c_H, _c_W], dim=1)
        
        if self.with_r:
           rr = torch.sqrt(torch.mul(_c_H-0.5, _c_H-0.5) + torch.mul(_c_W-0.5, _c_W-0.5))
           x = torch.cat([x, rr], dim=1)
        
        return self.conv(x)
        
