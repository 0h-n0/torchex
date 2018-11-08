import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from ..functional.local import conv2d_local
from .util import _pair

def _conv_output_length(input_length, filter_size, stride):
    output_length = input_length - filter_size + 1
    return output_length

class Conv2dLocal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, in_size=None,
                 padding=0, bias=True):
        super(Conv2dLocal, self).__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.in_size = in_size
        self.padding = padding
        self.bias_flag = bias
        self.bias = None
        self.weight = None

        kh, kw = _pair(self.kernel_size)
        ih, iw = _pair(in_size)

        if (self.in_channels is not None) and (self.in_size is not None):
            self._initialize_params(in_channels, in_size)
            self.reset_parameters()

    def _initialize_params(self, in_channels, in_size):
        kh, kw = _pair(self.kernel_size)
        ih, iw = _pair(in_size)
        oh = _conv_output_length(ih, kh, self.stride[0])
        ow = _conv_output_length(iw, kw, self.stride[1])
        W_shape = (self.out_channels, oh, ow, in_channels, kh, kw)
        bias_shape = (self.out_channels, oh, ow,)
        self.weight = Parameter(torch.Tensor(*W_shape))
        if self.bias_flag:
            self.bias = Parameter(torch.Tensor(*bias_shape))            
        else:
            self.register_parameter('bias', None)
        
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)        

    def forward(self, input):
        if self.weight is None:
            self._initialize_params(input.shape[1], input.shape[2:])
            self.reset_parameters()
        return conv2d_local(input, self.weight, self.bias, self.stride)

