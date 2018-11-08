import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBN(nn.Module):

    """Inception module of the new GoogLeNet with BatchNormalization.
    This chain acts like :class:`Inception`, while InceptionBN uses the
    :class:`BatchNormalization` on top of each convolution, the 5x5 convolution
    path is replaced by two consecutive 3x3 convolution applications, and the
    pooling method is configurable.
    See: `Batch Normalization: Accelerating Deep Network Training by Reducing \
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_.
    Args:
        in_channels (int or None): Number of channels of input arrays.
        out1 (int): Output size of the 1x1 convolution path.
        proj3 (int): Projection size of the single 3x3 convolution path.
        out3 (int): Output size of the single 3x3 convolution path.
        proj33 (int): Projection size of the double 3x3 convolutions path.
        out33 (int): Output size of the double 3x3 convolutions path.
        pooltype (str): Pooling type. It must be either ``'max'`` or ``'avg'``.
        proj_pool (int or None): Projection size in the pooling path. If
            ``None``, no projection is done.
        stride (int): Stride parameter of the last convolution of each path.
    .. seealso:: :class:`Inception`
    """

    def __init__(self, in_channels, out1, proj3, out3, proj33, out33,
                 pooltype='max', proj_pool=None, stride=1):
        super(InceptionBN, self).__init__()
        self.out1 = out1
        self.proj_pool = proj_pool
        self.stride = stride
        self.pooltype = pooltype
        if pooltype != 'max' and pooltype != 'avg':
            raise NotImplementedError()
        
        self.proj3 = nn.Conv2d(in_channels, proj3, 1, bias=False)
        self.conv3 = nn.Conv2d(proj3, out3, 3, padding=1, stride=stride, bias=False)
        self.proj33 = nn.Conv2d(in_channels, proj33, 1, bias=False)
        self.conv33a = nn.Conv2d(proj33, out33, 3, padding=1, bias=False)
        self.conv33b = nn.Conv2d(out33, out33, 3, padding=1, stride=stride, bias=False)
        self.proj3n = nn.BatchNorm2d(proj3)
        self.conv3n = nn.BatchNorm2d(out3)
        self.proj33n = nn.BatchNorm2d(proj33)
        self.conv33an = nn.BatchNorm2d(out33)
        self.conv33bn = nn.BatchNorm2d(out33)

        if out1 > 0:
            assert stride == 1
            assert proj_pool is not None
            self.conv1 = nn.Conv2d(in_channels, out1, 1, stride=stride, bias=False)
            self.conv1n = nn.BatchNorm2d(out1)

        if proj_pool is not None:
            self.poolp = nn.Conv2d(in_channels, proj_pool, 1, bias=False)
            self.poolpn = nn.BatchNorm2d(proj_pool)

    def forward(self, x):
        outs = []

        if self.out1 > 0:
            h1 = self.conv1(x)
            h1 = self.conv1n(h1)
            h1 = F.relu(h1)
            outs.append(h1)

        h3 = F.relu(self.proj3n(self.proj3(x)))
        h3 = F.relu(self.conv3n(self.conv3(h3)))
        outs.append(h3)

        h33 = F.relu(self.proj33n(self.proj33(x)))
        h33 = F.relu(self.conv33an(self.conv33a(h33)))
        h33 = F.relu(self.conv33bn(self.conv33b(h33)))
        outs.append(h33)

        if self.pooltype == 'max':
            p = F.max_pool2d(x, 3, stride=self.stride, padding=1,
                             cover_all=False)
        else:
            p = F.avg_pool2d(x, 3, stride=self.stride,
                             padding=1)
        if self.proj_pool is not None:
            p = F.relu(self.poolpn(self.poolp(p)))
        outs.append(p)

        y = torch.cat(outs, dim=1)
        return y
