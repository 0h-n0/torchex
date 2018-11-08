import torch

class GraphLinear(torch.nn.Module):
    """Graph Linear layer.

    This function assumes its input is 3-dimensional.
    Differently from :class:`chainer.functions.linear`, it applies an affine
    transformation to the third axis of input `x`.

    .. seealso:: :class:`torch.nn.Linear`
    """
    def __init__(self, *argv, **kwargs):

        super(GraphLinear, self).__init__()
        self.linear = torch.nn.Linear(*argv, **kwargs)

    def __call__(self, x):
        """Forward propagation.

        Args:
            x (:class:`torch.Tensor`)
                Input array that should be a float array whose ``dim`` is 3.

        Returns:
            :class:`torch.Tensor`:
                A 3-dimeisional array.

        """
        # (minibatch, atom, ch)
        s0, s1, s2 = x.size()
        x = x.view(s0 * s1, s2)
        x = self.linear(x)
        x = x.view(s0, s1, -1)
        return x
