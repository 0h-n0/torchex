# TODO: Conv2dLocal
# TODO: Conv2dMap
# TODO: ConvTranspose2dMap
import torch    
import torch.nn as nn
import torch.nn.functional as F
    
class MLPConv2d(nn.Module):
    """__init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, activation=relu.relu, conv_init=None, bias_init=None)
    Two-dimensional MLP convolution layer of Network in Network.
    This is an "mlpconv" layer from the Network in Network paper. This layer
    is a two-dimensional convolution layer followed by 1x1 convolution layers
    and interleaved activation functions.
    Note that it does not apply the activation function to the output of the
    last 1x1 convolution layer.
    Args:
        in_channels (int or None): Number of channels of input arrays.
            If it is ``None`` or omitted, parameter initialization will be
            deferred until the first forward data pass at which time the size
            will be determined.
        out_channels (tuple of ints): Tuple of number of channels. The i-th
            integer indicates the number of filters of the i-th convolution.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels) of the
            first convolution layer. ``ksize=k`` and ``ksize=(k, k)`` are
            equivalent.
        stride (int or pair of ints): Stride of filter applications at the
            first convolution layer. ``stride=s`` and ``stride=(s, s)`` are
            equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays at
            the first convolution layer. ``pad=p`` and ``pad=(p, p)`` are
            equivalent.
        activation (callable):
            Activation function for internal hidden units.
            You can specify one of activation functions from
            :doc:`built-in activation functions </reference/functions>` or
            your own function.
            It should not be an
            :doc:`activation functions with parameters </reference/links>`
            (i.e., :class:`~chainer.Link` instance).
            The function must accept one argument (the output from each child
            link), and return a value.
            Returned value must be a Variable derived from the input Variable
            to perform backpropagation on the variable.
            Note that this function is not applied to the output of this link.
            a keyword argument.
    .. note:
        From v2, `conv_init` and `bias_init` arguments must be specified as
        keyword arguments only. We impose this restriction to forbid
        users to assume the API for v1 and specify `wscale` option,
        that had been between `activation` and `conv_init` arguments in v1.
    See: `Network in Network <https://arxiv.org/abs/1312.4400v3>`_.
    Attributes:
        activation (callable):
            Activation function.
            See the description in the arguments for details.
    """  # NOQA

    def __init__(self, in_channels, out_channels, kernel_size=None, stride=1, padding=0,
                 activation=F.relu, *args, **kwargs):
        super(MLPConv2d, self).__init__()
        #if kernel_size is None:
        #    out_channels, kernel_size, in_channels = in_channels, out_channels, None

        assert len(out_channels) > 0
        
        self.convs = [nn.Conv2d(in_channels, out_channels[0], kernel_size, stride, padding)]
        for n_in, n_out in zip(out_channels, out_channels[1:]):
            self.convs.append(nn.Conv2d(n_in, n_out, 1))
        self.convs = nn.ModuleList(self.convs)
        self.activation = activation

    def forward(self, x):
        """Computes the output of the mlpconv layer.
        Args:
            x (~chainer.Variable): Input image.
        Returns:
            ~chainer.Variable: Output of the mlpconv layer.
        """
        f = self.activation
        for l in self.convs:
            x = f(l(x))
            
        return x
    
    
class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

    
