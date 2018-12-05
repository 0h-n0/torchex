import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, out_features, use_bias=True):
        super(Linear, self).__init__()
        self.out_features = out_features
        self.use_bias = use_bias
        
        self.initialize = True        
        self.weight = None
        self.bias = None
        self.to_args = None
        self.to_kwargs = None        

    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs = kwargs
        return super().to(*args, **kwargs)

    def forward(self, x):
        _, in_features = x.shape
        
        if self.weight is None:
            self.weight =  torch.Tensor(self.out_features, in_features)
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.use_bias:
                self.bias = torch.Tensor(self.out_features)
                self.bias.data.uniform_(-stdv, stdv)

            if self.to_args is not None:
                self.weight = self.weight.to(*self.to_args, **self.to_kwargs)
                if self.bias is not None:
                    self.bias = self.bias.to(*self.to_args, **self.to_kwargs)
            
        return F.linear(x, self.weight, self.bias)
    
