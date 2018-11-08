import torch.nn as nn


class GraphSortPool(nn.Module):
    '''

    Refs: https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf    
    '''
    def __init__(self):
        super(GraphSortPool, self).__init__()
        pass
        


class GraphMaxPool(nn.Module):
    ''' GraphMaxPool wraps MaxPool1d.
    '''
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(GraphPool, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size,
                                 stride,
                                 padding,
                                 dilation,
                                 return_indices,
                                 ceil_mode)

    def forward(self, x):
        return self.pool(x)


class GraphAvgPool(nn.Module):
    ''' GraphAvgPool wraps AvgPool1d.
    '''
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(GraphPool, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size,
                                 stride,
                                 padding,
                                 dilation,
                                 return_indices,
                                 ceil_mode)

    def forward(self, x):
        return self.pool(x)
    
