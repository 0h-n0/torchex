import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LazyBase

class Linear(LazyBase):
    '''
    Examples::

        import torch
        import torchex.nn as exnn
     
        net = exnn.Linear(10)
        # You don't need to give the size of input for this module.
        # This network is equivalent to `nn.Linear(100, 10)`.
     
        x = troch.randn(10, 100)
        y = net(x)

    '''
    def __init__(self, in_features, out_features=None, use_bias=True):
        super(Linear, self).__init__()
        if out_features is None:
            self.in_features, self.out_features = None, in_features
        else:
            self.in_features = in_features            
            self.out_features = out_features
        self.use_bias = use_bias
        
        self.initialize = True        
        self.weight = nn.Parameter(None)
        self.bias = nn.Parameter(None)
        self._register_load_state_dict_pre_hook(self._lazy_load_state_dict_hook)

    def _lazy_load_state_dict_hook(self, state_dict, prefix, local_metadata, strict,
                                   missing_keys, unexpected_keys, error_msgs):
        for name, data in state_dict.items():
            if prefix in name:
                if 'weight' in name:
                    self.in_features = data.shape[-1]
                    self.weight.data = data
                elif 'bias' in name:
                    self.bias.data = data
                else:
                    raise ValueError(name)

    def forward(self, x):
        if len(self.weight.data) == 0:
            self.in_features = x.shape[-1]
            self.weight.data =  torch.Tensor(self.out_features, self.in_features)
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            
            if self.use_bias:
                self.bias.data = torch.Tensor(self.out_features)
                self.bias.data.uniform_(-stdv, stdv)

            self.weight = self._to_device(self.weight)
            self.bias = self._to_device(self.bias)            
            
        return F.linear(x, self.weight, self.bias)
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )    
    
if __name__ == "__main__":
    x = torch.randn(10, 2).to('cuda')
    net = Linear(10)
    net = net.to('cuda')
    y = net(x)
    print(y)
