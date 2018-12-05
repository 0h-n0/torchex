import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from .utils import _single, _pair, _triple
from .base import LazyBase

class _ConvNd(LazyBase):
    def __init__(self, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        
        self.in_channels = None
        self.weight = None
        self.bias = bias

    def _initialize_weight(self, in_channels):
        self.in_channels = in_channels
        if self.in_channels % self.groups != 0:
            raise ValueError('in_channels must be divisible by groups')

        if self.transposed:
            self.weight = torch.Tensor(self.in_channels, self.out_channels // self.groups, *self.kernel_size)
        else:
            self.weight = torch.Tensor(self.out_channels, self.in_channels // self.groups, *self.kernel_size)
                
        if self.bias:
            self.bias = torch.Tensor(self.out_channels)

        if self.to_args is not None:
            self.weight = self.weight.to(*self.to_args, **self.to_kwargs)
            if self.bias is not None:
                self.bias = self.bias.to(*self.to_args, **self.to_kwargs)

        self.weight = nn.Parameter(self.weight)                    
        if self.bias is None:
            self.register_parameter('bias', None)
        else:
            self.bias = nn.Parameter(self.bias)                
        self._reset_parameters()

    def _reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)        
            
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv1d(_ConvNd):
    def __init__(self, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)
    
    def forward(self, x):
        in_channels = x.size(1)

        if self.weight is None:
            self._initialize_weight(in_channels)
            
        return F.conv1d(x, self.weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

    
class Conv2d(_ConvNd):
    def __init__(self, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, x):
        in_channels = x.size(1)

        if self.weight is None:
            self._initialize_weight(in_channels)            

        return F.conv2d(x, self.weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)


class Conv3d(_ConvNd):
    def __init__(self, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(Conv3d, self).__init__(
            out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias)
    
    def forward(self, x):
        in_channels = x.size(1)

        if self.weight is None:
            self._initialize_weight(in_channels)                        

        return F.conv3d(x, self.weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
    
