import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from .utils import _single, _pair, _triple
from .base import LazyBase
from ...init import feedforward_init

class _ConvNd(LazyBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels                    
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        self.weight = nn.Parameter(None)
        self.bias = nn.Parameter(None)
        self.bias_flag = bias
        self._register_load_state_dict_pre_hook(self._lazy_load_state_dict_hook)        

    def _initialize_weight(self, in_channels, xavier_init: bool):

        self.in_channels = in_channels
        if self.in_channels % self.groups != 0:
            raise ValueError('in_channels must be divisible by groups')

        if self.transposed:
            self.weight.data = torch.Tensor(self.in_channels,
                                            self.out_channels // self.groups,
                                            *self.kernel_size)
        else:
            self.weight.data = torch.Tensor(self.out_channels,
                                            self.in_channels // self.groups,
                                            *self.kernel_size)
                
        if self.bias_flag:
            self.bias.data = torch.Tensor(self.out_channels)

        self.weight = self._to_device(self.weight)
        self.bias = self._to_device(self.bias)            
        self._reset_parameters()
        
        if xavier_init:
            feedforward_init(self)        

    def _reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _lazy_load_state_dict_hook(self, state_dict, prefix, local_metadata, strict,
                                   missing_keys, unexpected_keys, error_msgs):
        for name, data in state_dict.items():
            if prefix in name:
                if 'weight' in name:
                    self.in_channels = data.shape[0]
                    self.weight.data = data
                elif 'bias' in name:
                    self.bias.data = data
                else:
                    raise ValueError(name)
            
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
    '''
    :param out_channels: the size of the window to take a max and average over
    :param kernel_size: the size of the window to take a max and average over
    :param stride: the size of stride to move kernel
    :param padding: implicit zero padding to be added on both sides
    :param dilation: a parameter that controls the stride of elements in the window
    :param return_indices: if True, will return the max indices along with the outputs. Useful when Unpooling later
    :param ceil_mode: when True, will use ceil instead of floor to compute the output shape    

    :type kernel_size: int or list
    :type stride: int or list
    
    Examples::

        import torch
        import torchex.nn as exnn
     
        net = exnn.Conv1d(10, 2)
        # You don't need to give the size of input for this module.
        # This network is equivalent to `nn.Conv1d(3, 10, 2)`.
     
        x = troch.randn(10, 3, 28)
        y = net(x)
    '''
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int or list=None, stride: int or list=1,
                 padding: int=0, dilation: int=1, groups: int=1,
                 bias: bool=True, xavier_init: bool=True):
        if kernel_size is None:
            kernel_size = out_channels            
            out_channels = in_channels
            in_channels =  None
        else:
            in_channels = in_channels                    
            out_channels = out_channels
            kernel_size = kernel_size
        
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        self.xavier_init = xavier_init        
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)
    
    def forward(self, x):
        if len(self.weight) == 0:
            in_channels = x.size(1)            
            self._initialize_weight(in_channels, self.xavier_init)
            
        return F.conv1d(x, self.weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

    
class Conv2d(_ConvNd):
    '''
    Examples::

        import torch
        import torchex.nn as exnn
     
        net = exnn.Conv2d(10, 2)
        # You don't need to give the size of input for this module.
        # This network is equivalent to `nn.Conv2d(3, 10, 2)`.
     
        x = troch.randn(10, 3, 28, 28)
        y = net(x)
    '''
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int or list=None, stride: int or list=1,
                 padding: int=0, dilation: int=1, groups: int=1, 
                 bias: bool=True, xavier_init: bool=True):        
        if kernel_size is None:
            kernel_size = out_channels            
            out_channels = in_channels
            in_channels =  None
        else:
            in_channels = in_channels                    
            out_channels = out_channels
            kernel_size = kernel_size
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.xavier_init = xavier_init
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, x):
        if len(self.weight) == 0:
            in_channels = x.size(1)            
            self._initialize_weight(in_channels, self.xavier_init)            

        return F.conv2d(x, self.weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

class Conv3d(_ConvNd):
    '''
    Examples::

        import torch
        import torchex.nn as exnn
     
        net = exnn.Conv3d(10, 2)
        # You don't need to give the size of input for this module.
        # This network is equivalent to `nn.Conv3d(3, 10, 2)`.
     
        x = troch.randn(10, 3, 100, 28, 28)
        y = net(x)
    '''
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int or list=None, stride: int or list=1,
                 padding: int=0, dilation: int=1, groups: int=1,
                 bias: bool=True, xavier_init: bool=True):        
        if kernel_size is None:
            kernel_size = out_channels            
            out_channels = in_channels
            in_channels =  None
        else:
            in_channels = in_channels                    
            out_channels = out_channels
            kernel_size = kernel_size
        
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        self.xavier_init = xavier_init        
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias)
    
    def forward(self, x):
        if len(self.weight) == 0:
            in_channels = x.size(1)            
            self._initialize_weight(in_channels, self.xavier_init)                        

        return F.conv3d(x, self.weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
    
