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
    def __init__(self, out_features, use_bias=True):
        super(Linear, self).__init__()
        self.out_features = out_features
        self.use_bias = use_bias
        
        self.initialize = True        
        self.weight = nn.Parameter(None)
        self.bias = nn.Parameter(None)

    def forward(self, x):
        _, in_features = x.shape
        print(self.weight.data)        
        if len(self.weight.data) == 0:
            print(self.weight.data)
            print(self.to_args)
            self.weight.data =  torch.Tensor(self.out_features, in_features)
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            
            if self.use_bias:
                self.bias.data = torch.Tensor(self.out_features)
                self.bias.data.uniform_(-stdv, stdv)

            self.weight = self._to_device(self.weight)
            self.bias = self._to_device(self.bias)            
            
        return F.linear(x, self.weight, self.bias)
    
if __name__ == "__main__":
    x = torch.randn(10, 2).to('cuda')
    net = Linear(10)
    net = net.to('cuda')
    y = net(x)
    print(y)
