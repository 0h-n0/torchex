import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    """Inception module of GoogLeNet.
    It applies four different functions to the input array and concatenates
    their outputs along the channel dimension. Three of them are 2D
    convolutions of sizes 1x1, 3x3 and 5x5. Convolution paths of 3x3 and 5x5
    sizes have 1x1 convolutions (called projections) ahead of them. The other
    path consists of 1x1 convolution (projection) and 3x3 max pooling.
    The output array has the same spatial size as the input. In order to
    satisfy this, Inception module uses appropriate padding for each
    convolution and pooling.
    See: `Going Deeper with Convolutions <https://arxiv.org/abs/1409.4842>`_.
    Args:
        in_channels (int or None): Number of channels of input arrays.
        out1 (int): Output size of 1x1 convolution path.
        proj3 (int): Projection size of 3x3 convolution path.
        out3 (int): Output size of 3x3 convolution path.
        proj5 (int): Projection size of 5x5 convolution path.
        out5 (int): Output size of 5x5 convolution path.
        proj_pool (int): Projection size of max pooling path.

    TODO: Implement dimension reduction version.
    """

    def __init__(self, in_channels, out1, proj3, out3, proj5, out5, proj_pool):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out1, 1)
        self.proj3 = nn.Conv2d(in_channels, proj3, 1)
        self.conv3 = nn.Conv2d(proj3, out3, 3, padding=1)
        self.proj5 = nn.Conv2d(in_channels, proj5, 1)
        self.conv5 = nn.Conv2d(proj5, out5, 5, padding=2)
        self.projp = nn.Conv2d(in_channels, proj_pool, 1)

    def forward(self, x):
        """Computes the output of the Inception module.
        Args:
            x (~torch.Tensor): Input variable.
        Returns:
            Variable: Output variable. Its array has the same spatial size and
            the same minibatch size as the input array. The channel dimension
            has size ``out1 + out3 + out5 + proj_pool``.
        """
        out1 = self.conv1(x)
        out3 = self.conv3(F.relu(self.proj3(x)))
        out5 = self.conv5(F.relu(self.proj5(x)))
        pool = self.projp(F.max_pool2d(x, 3, stride=1, padding=1))
        y = F.relu(torch.cat((out1, out3, out5, pool), dim=1))
        return y
