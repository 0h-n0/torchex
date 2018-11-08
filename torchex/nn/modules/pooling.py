import torch.nn as nn
import torch.nn.functional as F

class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 3, x.size()
        B, C, L = x.size()
        return F.avg_pool1d(x, L)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 4, x.size()
        B, C, W, H = x.size()
        return F.avg_pool2d(x, (W, H))


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 3, x.size()
        B, C, L = x.size()
        return F.avg_pool1d(x, L)

    
class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool2d, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 4, x.size()
        B, C, W, H = x.size()
        return F.max_pool2d(x, (W, H))
    

class MaxAvgPool2d(nn.Module):
    '''
    :param kernel_size: the size of the window to take a max and average over
    :param stride: the size of stride to move kernel
    :param padding: implicit zero padding to be added on both sides
    :param dilation: a parameter that controls the stride of elements in the window
    :param return_indices: if True, will return the max indices along with the outputs. Useful when Unpooling later
    :param ceil_mode: when True, will use ceil instead of floor to compute the output shape    

    :type kernel_size: int or list
    :type stride: int or list
    '''

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False,
                 count_include_pad=True):
        super(MaxAvgPool2d, self).__init__()
        kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode)
        self.max_pool = nn.MaxPool2d(**kwargs)
        del kwargs['dilation']
        del kwargs['return_indices']        
        kwargs['count_include_pad'] = count_include_pad
        self.avg_pool = nn.AvgPool2d(**kwargs)

    def forward(self, x):
        mx = self.max_pool(x)
        ax = self.avg_pool(x)
        return mx + ax
