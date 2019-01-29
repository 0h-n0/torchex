from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchex.nn as exnn


def test_conv1d_print():
    net = exnn.Conv1d(3, 2)
    print(net)

def test_conv1d():
    net = exnn.Conv1d(3, 2)
    x = torch.randn(10, 10, 20)
    y = net(x)
    assert list(y.shape) == [10, 3, 19]
    
def test_conv2d():
    net = exnn.Conv2d(3, 2)
    x = torch.randn(10, 20, 28, 28)
    y = net(x)
    assert list(y.shape) == [10, 3, 27, 27]

def test_conv3d():
    net = exnn.Conv3d(3, 2)
    x = torch.randn(10, 20, 28, 28, 28)
    y = net(x)
    assert list(y.shape) == [10, 3, 27, 27, 27]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU')
def test_cuda_conv1d():
    net = exnn.Conv1d(3, 2).to('cuda')
    x = torch.randn(10, 10, 20).to('cuda')
    y = net(x)
    assert list(y.shape) == [10, 3, 19]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU')
def test_cuda_conv2d():
    net = exnn.Conv2d(3, 2).to('cuda')
    x = torch.randn(10, 20, 28, 28).to('cuda')
    y = net(x)
    assert list(y.shape) == [10, 3, 27, 27]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU')
def test_cuda_conv3d():
    net = exnn.Conv3d(3, 2).to('cuda')
    x = torch.randn(10, 20, 28, 28, 28).to('cuda')
    y = net(x)
    assert list(y.shape) == [10, 3, 27, 27, 27]
    

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU')
def test_cuda_conv1d_with_seq():
    net = nn.Sequential(
        exnn.Conv1d(3, 2)
        )
    net = net.to('cuda')

    x = torch.randn(10, 10, 20).to('cuda')
    y = net(x)
    assert list(y.shape) == [10, 3, 19]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU')
def test_cuda_conv2d_with_seq():
    net = nn.Sequential(
        exnn.Conv2d(3, 2)
        )
    net = net.to('cuda')
    x = torch.randn(10, 20, 28, 28).to('cuda')
    y = net(x)
    assert list(y.shape) == [10, 3, 27, 27]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU')
def test_cuda_conv3d_with_seq():
    net = nn.Sequential(
        exnn.Conv3d(3, 2)
        )
    net = net.to('cuda')
    
    x = torch.randn(10, 20, 28, 28, 28).to('cuda')
    y = net(x)    
    assert list(y.shape) == [10, 3, 27, 27, 27]    

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU')
def test_cuda_conv1d_with_cls():
    class MyConv(nn.Module):
        def __init__(self):
            super(MyConv, self).__init__()
            self.conv = exnn.Conv1d(3, 2)
        def forward(self, x):
            return self.conv(x)
    net = MyConv().to('cuda')                

    x = torch.randn(10, 10, 20).to('cuda')
    y = net(x)
    assert list(y.shape) == [10, 3, 19]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU')
def test_cuda_conv2d_with_cls():
    class MyConv(nn.Module):
        def __init__(self):
            super(MyConv, self).__init__()
            self.conv = exnn.Conv2d(3, 2)
        def forward(self, x):
            return self.conv(x)
    net = MyConv().to('cuda')        
    x = torch.randn(10, 20, 28, 28).to('cuda')
    y = net(x)
    assert list(y.shape) == [10, 3, 27, 27]

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU')
def test_cuda_conv3d_with_cls():
    class MyConv(nn.Module):
        def __init__(self):
            super(MyConv, self).__init__()
            self.conv = exnn.Conv3d(3, 2)
        def forward(self, x):
            return self.conv(x)
    net = MyConv().to('cuda')
    
    x = torch.randn(10, 20, 28, 28, 28).to('cuda')
    y = net(x)    
    assert list(y.shape) == [10, 3, 27, 27, 27]    

def test_load_conv1d_model(tmpdir):
    class MyConv(nn.Module):
        def __init__(self):
            super(MyConv, self).__init__()
            self.conv = exnn.Conv1d(3, 2)            
        def forward(self, x):
            return self.conv(x)

    path = Path(tmpdir) / 'model.pth'
    x = torch.randn(10, 10, 20)
    net = MyConv()
    y = net(x)
    torch.save(net.state_dict(), path)

    net2 = MyConv()
    net2.load_state_dict(torch.load(path))
    assert list(y.shape) == [10, 3, 19]

def test_load_many_conv1d_layers_model(tmpdir):
    class MyConv(nn.Module):
        def __init__(self):
            super(MyConv, self).__init__()
            self.conv = nn.Sequential(
                exnn.Conv1d(20, 2),
                exnn.Conv1d(30, 2),
                exnn.Flatten(),
                )
                
        def forward(self, x):
            return self.conv(x)

    path = Path(tmpdir) / 'model.pth'
    x = torch.randn(10, 10, 20)
    net = MyConv()
    y = net(x)
    torch.save(net.state_dict(), path)

    net2 = MyConv()
    net2.load_state_dict(torch.load(path))
    assert list(y.shape) == [10, 540]

    
def test_load_conv2d_model(tmpdir):
    class MyConv(nn.Module):
        def __init__(self):
            super(MyConv, self).__init__()
            self.conv = exnn.Conv2d(3, 2)            
        def forward(self, x):
            return self.conv(x)

    path = Path(tmpdir) / 'model.pth'
    x = torch.randn(10, 10, 20)
    net = MyConv()
    y = net(x)
    torch.save(net.state_dict(), path)

    net2 = MyConv()
    net2.load_state_dict(torch.load(path))
    assert list(y.shape) == [10, 3, 27, 27]    

def test_load_many_conv2d_layers_model(tmpdir):
    class MyConv(nn.Module):
        def __init__(self):
            super(MyConv, self).__init__()
            self.conv = nn.Sequential(
                exnn.Conv2d(20, 2),
                exnn.Conv2d(30, 2),
                exnn.Flatten(),
                )
                
        def forward(self, x):
            return self.conv(x)

    path = Path(tmpdir) / 'model.pth'
    x = torch.randn(10, 10, 20)
    net = MyConv()
    y = net(x)
    torch.save(net.state_dict(), path)

    net2 = MyConv()
    net2.load_state_dict(torch.load(path))
    assert list(y.shape) == [10, 540]
    
