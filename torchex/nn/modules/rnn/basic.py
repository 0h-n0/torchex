from typing import Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from ...init import rnn_init


class SequenceLinear(nn.Module):
    '''
    x[T, B, F] -> nn.Linear(x[T, BxF]) -> x[T, B, F]
    if batch_first is True,
    x[B, T, F] -> x[T, B, F] -> nn.Linear(x[T, BxF]) -> x[T, B, F]
    '''
    def __init__(self,
                 in_features,
                 out_features,
                 batch_first=True,
                 nonlinearity='sigmoid',
                 init_mean=0,
                 init_std=1,
                 init_xavier: bool=True,
                 init_normal: bool=True,
                 init_gain = None,
                 dropout=0.0,
                 bias=True,
    ):
        super(SequenceLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.batch_first = batch_first
        self.out_features = out_features
        self.nonlinearity = nonlinearity
        if not init_gain:
            init_gain = nn.init.calculate_gain(nonlinearity)
        
        feedforward_init(self.linear,
                         init_mean,
                         init_std,
                         init_xavier,
                         init_normal,
                         init_gain)
        
    def forward(self, x: Tensor):
        assert len(x.size()) == 3, x.size()
        if self.batch_first:
            B, T, F = x.size()
            x = torch.transpose(x, 0, 1)
        else:
            T, B, F = x.size()
        x = x.contiguous()
        x = x.view(T*B, F)
        x = self.linear(x)
        x = getattr(TF, self.nonlinearity)(x)
        
        if self.batch_first:
            x = x.view(T, B, self.out_features)            
            x = torch.transpose(x, 0, 1)
        else:
            x = x.view(T, B, self.out_features)
        return x



class LSTM(nn.Module):
    """Basic LSTM model. """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.0,
                 batch_first=True,
                 init_xavier: bool=True,
                 init_normal: bool=True,
                 init_gain: float=1.0,
                 concat: bool=True,
                 init_mean=0,                 
                 init_std=0.1,
                 init_lower=0,                 
                 init_upper=0.04,
                 concat_between_layers=False,
                 ):
        super(LSTMModel, self).__init__()
        self.in_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.concat = concat
        self.concat_between_layers = concat_between_layers
        
        if concat_between_layers and bidirectional:
            _rnn = []
            _rnn.append(nn.LSTM(input_size, hidden_size, num_layers=1,
                                bidirectional=bidirectional,
                                dropout=dropout, batch_first=batch_first))
            for i in range(num_layers - 1):
                _rnn.append(nn.LSTM(2 * hidden_size, hidden_size, num_layers=1,
                                    bidirectional=bidirectional,
                                    dropout=dropout, batch_first=batch_first))
            for irnn in _rnn:
                rnn_init(irnn,
                         init_xavier=init_xavier,
                         init_normal=init_normal,
                         init_gain=init_gain,
                         init_mean=init_mean,
                         init_std=init_std)
                
            self.rnn = nn.ModuleList(_rnn)
        else:
            self.rnn =\
                       nn.LSTM(input_size, hidden_size, num_layers,
                               bidirectional=bidirectional,
                               dropout=dropout, batch_first=batch_first,
                       )
            rnn_init(self.rnn,
                     init_xavier=init_xavier,
                     init_normal=init_normal,
                     init_gain=init_gain,
                     init_mean=init_mean,
                     init_std=init_std)
            

    def forward(self,
                x: Union[Tensor, PackedSequence],
                hx: Union[Tensor, Tuple[Tensor, ...]]=None) ->\
                Tuple[Tensor, Tensor]:
        
        assert isinstance(x, Tensor) or\
            isinstance(x, PackedSequence), type(x)

        if self.concat_between_layers and self.bidirectional:
            output = x
            for idx, _rnn in enumerate(self.rnn):
                _rnn.flatten_parameters()                
                output, hx = _rnn(output)
                _rnn.flatten_parameters()                
        else:
            self.rnn.flatten_parameters()
            output, hx = self.rnn(x, hx)
            self.rnn.flatten_parameters()            
        
        if (not self.concat) and self.bidirectional:
            B, T, F = output.size()
            output = output[:, :, :F//2] + output[:, :, F//2:]
            
        return output, hx
