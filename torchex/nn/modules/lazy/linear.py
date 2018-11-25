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

    def forward(self, x):
        _, in_features = x.shape
        if self.initialize:
            self.weight =  nn.Parameter(torch.Tensor(self.out_features, in_features))
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.use_bias:
                self.bias = nn.Parameter(torch.Tensor(self.out_features))
                self.bias.data.uniform_(-stdv, stdv)
            self.initialize = False
            
        return F.linear(x, self.weight, self.bias)
    
