import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
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
        self.to_args = None
        self.to_kwargs = None

    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs = kwargs
        return super().to(*args, **kwargs)

    def forward(self, x):
        _, in_features = x.shape
        
        if len(self.weight.data) == 0:            
            self.weight.data =  torch.Tensor(self.out_features, in_features)
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            
            if self.use_bias:
                self.bias.data = torch.Tensor(self.out_features)
                self.bias.data.uniform_(-stdv, stdv)
            if self.to_args is not None:
                self.weight.data = self.weight.data.to(*self.to_args, **self.to_kwargs)
                if self.bias is not None:
                    self.bias.data = self.bias.data.to(*self.to_args, **self.to_kwargs)
            
        return F.linear(x, self.weight, self.bias)
    
if __name__ == "__main__":
    x = torch.randn(10, 2).to('cuda')
    net = Linear(10)
    net = net.to('cuda')
    y = net(x)
    print(y)
