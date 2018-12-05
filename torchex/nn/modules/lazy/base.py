import torch.nn as nn


class LazyBase(nn.Module):
    def __init__(self):
        super(LazyBase, self).__init__()
        self.to_args = None
        self.to_kwargs = None

    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs = kwargs
        return super().to(*args, **kwargs)
        
