from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchex.nn as exnn

def test_linear():
    net = exnn.Linear(3)
    x = torch.randn(10, 20)
    y = net(x)
    assert list(y.shape) == [10, 3]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No avilable GPU')
def test_to_cuda_linear():
    net = exnn.Linear(3).to('cuda')
    x = torch.randn(10, 20).to('cuda')
    y = net(x)
    assert list(y.shape) == [10, 3]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No avilable GPU')
def test_to_cuda_linear_with_sequential():
    net = nn.Sequential(
        exnn.Linear(3)
        )
    x = torch.randn(10, 20).to('cuda')
    net = net.to('cuda')
    y = net(x)
    assert list(y.shape) == [10, 3]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No avilable GPU')
def test_to_cuda_linear_with_class_definition():
    class MyLinear(nn.Module):
        def __init__(self):
            super(MyLinear, self).__init__()
            self.linear = exnn.Linear(3)
        def forward(self, x):
            return self.linear(x)
                                      
    x = torch.randn(10, 20).to('cuda')
    net = MyLinear().to('cuda')
    y = net(x)
    assert list(y.shape) == [10, 3]
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason='No avilable GPU')
def test_cuda_linear():
    net = exnn.Linear(3).to('cuda')
    x = torch.randn(10, 20).cuda()
    y = net(x)
    assert list(y.shape) == [10, 3]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No avilable GPU')
def test_cuda_linear_with_sequential():
    net = nn.Sequential(
        exnn.Linear(3)
        )
    x = torch.randn(10, 20).cuda()
    net = net.cuda()
    y = net(x)
    assert list(y.shape) == [10, 3]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No avilable GPU')
def test_cuda_linear_with_class_definition():
    class MyLinear(nn.Module):
        def __init__(self):
            super(MyLinear, self).__init__()
            self.linear = exnn.Linear(3)
        def forward(self, x):
            return self.linear(x)
                                      
    x = torch.randn(10, 20).cuda()
    net = MyLinear().cuda()
    y = net(x)
    assert list(y.shape) == [10, 3]
    
def test_load_model(tmpdir):
    class MyLinear(nn.Module):
        def __init__(self):
            super(MyLinear, self).__init__()
            self.linear = exnn.Linear(3)
        def forward(self, x):
            return self.linear(x)

    path = Path(tmpdir) / 'model.pth'
    
    x = torch.randn(10, 20)
    net = MyLinear()
    net(x)
    torch.save(net.state_dict(), path)

    net2 = MyLinear()
    net2.load_state_dict(torch.load(path))
    y = net2(x)
    assert list(y.shape) == [10, 3]

def test_load_sequential_model(tmpdir):
    class MyLinear(nn.Module):
        def __init__(self):
            super(MyLinear, self).__init__()
            self.linear = nn.Sequential(
                exnn.Linear(3)
                )
        def forward(self, x):
            return self.linear(x)

    path = Path(tmpdir) / 'model2.pth'
    
    x = torch.randn(10, 20)
    net = MyLinear()
    net(x)
    torch.save(net.state_dict(), path)

    net2 = MyLinear()
    net2.load_state_dict(torch.load(path))
    y = net2(x)
    assert list(y.shape) == [10, 3]
    
    
    
