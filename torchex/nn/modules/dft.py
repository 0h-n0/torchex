import torch
import torch.nn as nn

from ...utils import (to_complex,
                      to_real,
                      complex_norm)


class DFT1d(nn.Module):
    def __init__(self, in_channels, out_channels, normlized=True, absolute=True):
        super(DFT1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.absolute = absolute

    def forward(self, x):
        if self.absolute:
            x = torch.abs(x)
        return x

    
class DFT2d(nn.Module):
    def __init__(self, normlized=True, norm=True, concatinate=False):
        super(DFT2d, self).__init__()        
        self.normlized = normlized
        self.norm = norm
        self.concatinate = concatinate

    def forward(self, x):
        if x.size(-1) != 2:
            x = to_complex(x)
        x = torch.fft(x, 2, self.normlized)
        if self.concatinate:
            _dim = x[..., 1].dim() - 1
            x = torch.cat([x[..., 0], x[..., 1]], dim=_dim)
        if self.norm and not self.concatinate:
            x = complex_norm(x)
        return x

    
class DFT3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass


class iDFT1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass


class iDFT2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass


class iDFT3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass
    

class RFFT1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass


class RFFT2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass


class RFFT3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass


class iRFFT1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass


class iRFFT2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass


class iRFFT3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass

    
    
class iSTFT1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass


class iSTFT2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self):
        pass



if __name__ == '__main__':
    x = torch.randn(10, 1, 28, 28)
    dft2d = DFTConv2d(1, 3)
    y = dft2d(x)
    print(y.shape)
