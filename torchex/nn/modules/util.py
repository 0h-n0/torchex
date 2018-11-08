import collections.abc
from itertools import repeat

import torch.nn as nn

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
