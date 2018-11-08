import torch.nn as nn

class Classifier(nn.Module):
    '''
    When you use multi-GPU, This module provides calculating loss function on each GPUs.
    If you don't use this module, I know the training speed slightly decrease.
    '''
    def __init__(self, model: nn.Module, criterion: nn.Module):
        super(Classifier, self).__init__()
        self.model = model
        self.criterion = criterion

    def __call__(self, x, t):
        out = self.model(x)
        loss = self.criterion(out, t)
        return loss
