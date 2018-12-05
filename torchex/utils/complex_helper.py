import torch

def to_complex(x: torch.Tensor) -> torch.Tensor:
    '''
    change tensor from real to complex.
    note: to_complex(to_complex(x)) is not supported.
    '''
    _x = torch.zeros_like(x)
    return torch.stack([x, _x], dim=x.dim())

def to_real(x: torch.Tensor) -> torch.Tensor:
    '''
    change tensor from complex to real.    
    '''
    assert 2 == x.size(-1), 'last dimmension of complex tensor must be2 (x.size(-1) == 2)'
    return x[..., 0]

def complex_norm(x: torch.Tensor) -> torch.Tensor:
    x = x.type(torch.float)
    assert 2 == x.size(-1)
    return torch.sqrt(torch.pow(x[..., 0], 2) + torch.pow(x[..., 1], 2)).type(x.dtype)
