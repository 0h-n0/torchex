import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDCell(nn.GRUCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUDCell, self).__init__(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.reset_layer_i = nn.Linear(input_size, hidden_size, bias)
        self.reset_layer_h = nn.Linear(hidden_size, hidden_size, bias)
        
        self.current_memory_i = nn.Linear(input_size, hidden_size, bias)
        self.current_memory_h = nn.Linear(hidden_size, hidden_size, bias)

    def forward(self, input, hx=None):
        #self.check_forward_input(input)
        #self.check_forward_hidden(input, hx)
        z = F.sigmoid(input + hx)
        rh = F.sigmoid(self.reset_layer_i(input) + self.reset_layer_h(hx))

        cx = self.current_memory_i(input)
        ch = self.current_memory_h(hx)
        c = torch.tanh(cx + torch.mul(rh, ch))
        
        h = torch.mul(z, hx) + torch.mul(1 - z, c)
        
        return h, h

class GRUD(nn.Module):
    '''

    refs: https://www.nature.com/articles/s41598-018-24271-9
    '''
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=True,
                 dropout=0, bidirectional=False):
        # not support multi layers
        # not support bidirectional
        # not suport dropout
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = GRUDCell(input_size, hidden_size, bias)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            hx = torch.autograd.Variable(input.data.new(max_batch_size,
                                                        self.hidden_size).zero_(),
                                         requires_grad=False)

        layer_output_list = []
        last_state_list   = []
            
        if self.batch_first:
            B, T, F = input.size()
        else:
            T, B, F = input.size()
        x = input
        h = hx
        for t in range(T):
            o, h = self.cell(x[:, t, :], h)
            layer_output = torch.stack(o, dim=1)
            layer_output_list.append(layer_output)

        return torch.stack(layer_output), h
    

if __name__ == "__main__":
    ## WIP
    from torch.nn.utils.rnn import PackedSequence
    
    grud = GRUD(20, 20)
    a = torch.randn(1, 10, 20)
    print(grud)
    grud(torch.autograd.Variable(a))
