import torch
import torch.nn as nn


class LazyBase(nn.Module):
    def __init__(self):
        super(LazyBase, self).__init__()
        self.to_args = None
        self.to_kwargs = None
        self._fn = []
        self._register_load_state_dict_pre_hook(self._state_dict_hook)        

    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs = kwargs
        return super().to(*args, **kwargs)
        
    def _apply(self, fn):
        self._fn.append(fn)
        super()._apply(fn)
        return self

    def _state_dict_hook(self, state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
        for name, data in state_dict.items():
            if 'weight' in name:
                self.weight.data = data
            elif 'bias' in name:
                self.bias.data = data
            else:
                raise ValueError(name)
    
    def _to_device(self, param):
        if param is None:
            return None
        
        if self.to_args is not None:
            param.data = param.data.to(*self.to_args, **self.to_kwargs)

        if self._fn:
            for f in self._fn:
                if param is not None:
                    # Tensors stored in modules are graph leaves, and we don't
                    # want to create copy nodes, so we have to unpack the data.
                    param.data = f(param.data)
                    if param._grad is not None:
                        param._grad.data = f(param._grad.data)            
        return param
