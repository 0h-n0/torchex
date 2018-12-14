import numbers

import torch
import torch.nn as nn


def chrono_init(rnn: torch.nn.Module, Tmax=None, Tmin=1):
    '''chrono initialization(Ref: https://arxiv.org/abs/1804.11188)
    '''
    
    assert isinstance(Tmin, numbers.Number), 'Tmin must be numeric.'
    assert isinstance(Tmax, numbers.Number), 'Tmax must be numeric.'    
    for name, p in rnn.named_parameters():
        if 'bias' in name:
            n = p.nelement()
            hidden_size = n // 4            
            p.data.fill_(0)
            if isinstance(rnn, (torch.nn.LSTM, torch.nn.LSTMCell)):
                p.data[hidden_size: 2*hidden_size] = \
                    torch.log(torch.nn.init.uniform_(p.data[0: hidden_size], 1, Tmax - 1))
                # forget gate biases = log(uniform(1, Tmax-1))
                p.data[0: hidden_size] = -p.data[hidden_size: 2*hidden_size]
                # input gate biases = -(forget gate biases)
    return rnn


def feedforward_init(dnn: nn.Module,
                     init_mean=0,
                     init_std=1,
                     init_xavier: bool=True,
                     init_normal: bool=True,
                     init_gain: float=1.0):
    for name, p in dnn.named_parameters():
        if 'bias' in name:
            p.data.zero_()
        if 'weight' in name:            
            if init_xavier:
                if init_normal:
                    nn.init.xavier_normal(p.data, init_gain)
                else:
                    nn.init.xavier_uniform(p.data, init_gain)
            else:
                if init_normal:
                    nn.init.normal(p.data, init_gain)
                else:
                    nn.init.uniform(p.data, init_gain)

                    
def rnn_init(rnn: nn.Module,
             init_xavier: bool=True,
             init_normal: bool=True,
             init_gain: float=1.0,
             init_mean: float=0.0,
             init_std: float=0.1,
             init_lower: float=0.0,
             init_upper: float=0.04):
    
    for name, p in rnn.named_parameters():
        if 'bias' in name:
            p.data.fill_(0)
            if isinstance(rnn, (torch.nn.LSTM, torch.nn.LSTMCell)):
                n = p.nelement()
                p.data[n // 4:n // 2].fill_(1)  # forget bias
        elif 'weight' in name:
            if init_xavier:
                if init_normal:
                    nn.init.xavier_normal(p, init_gain)
                else:
                    nn.init.xavier_uniform(p, init_gain)
            else:
                if init_normal:
                    try:
                        # from pytorch 4.0
                        nn.init.normal_(p, init_mean, init_std)
                    except:
                        # pytorch 3.1
                        nn.init.normal(p, init_mean, init_std)                        
                else:
                    try:
                        nn.init.uniform_(p, init_lower, init_upper)
                    except:
                        nn.init.uniform(p, init_lower, init_upper)                        
                    
