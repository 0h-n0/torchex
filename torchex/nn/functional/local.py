import torch

from .util import tensordot

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x

class Conv2dLocalFunction(torch.autograd.Function):
    ''' Conv2dLocalFunction is based on chainer implementation.

    Ref: https://github.com/chainer/chainer/blob/master/chainer/functions/connection/local_convolution_2d.py
    '''
        
    @staticmethod
    def forward(ctx, input, sy, sx):
        ctx.save_for_backward(input)
        ctx.sy = sy
        ctx.sx = sx
        
        x, W = input[:2]
        b = input[2] if len(input) == 3 else None

        stride_row, stride_col = sy, sx
        output_row, output_col = W.shape[1], W.shape[2]
        feature_dim = W.shape[3] * W.shape[4] * W.shape[5]
        output = torch.empty((x.shape[0], W.shape[0], output_row, output_col,), dtype=x.dtype)
        for i in range(output_row):
            for j in range(output_col):
                slice_row = slice(i * stride_row,
                                  i * stride_row + W.shape[4])
                slice_col = slice(j * stride_col,
                                  j * stride_col + W.shape[5])
                x_flatten = torch.reshape(x[..., slice_row, slice_col],
                                       (-1, feature_dim))
                W_flatten = torch.reshape(W[:, i, j, ...],
                                       (-1, feature_dim))
                output[..., i, j] = torch.matmul(x_flatten, torch.t(W_flatten))

        if b is not None:
            output += b[None, :, :, :]

        return output


    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        x_tensor, W_tensor  = input[:2]
        x, W = x_tensor.data, W_tensor.data
        stride_row, stride_col = ctx.sy, ctx.sx
        output_row, output_col = W.shape[1], W.shape[2]
        gy_tensor = grad_output.clone()
        gy = gy_tensor.data
        ret = []

        gx = torch.zero_like(x)
        for i in range(output_row):
            for j in range(output_col):
                slice_row = slice(i * stride_row,
                                  i * stride_row + W.shape[4])
                slice_col = slice(j * stride_col,
                                  j * stride_col + W.shape[5])
                # ochans * ichans * krows * kcols
                W_slice = W[:, i, j, ...]
                # nsamps * ochans
                gy_slice = gy[..., i, j]
                # -> nsamps * ichans * krows * kcols
                gx[:, :, slice_row, slice_col] += tensordot(
                    gy_slice, W_slice, axes=[(1,), (0,)]
                )
                # use torch.mm
            ret.append(gx.type(x.dtype))            
        return ret


def conv2d_local(x, W, b=None, stride=1):
    fnode = Conv2dLocalFunction(stride)
    sx, sy = _pair(stride)
    if b is None:
        args = (x, W)
    else:
        args = (x, W, b)
    y = fnode.apply(args, sy, sx)
    return y    


if __name__ == '__main__':
    x = torch.randn((2, 3, 7, 7), requires_grad=True)    
    W = torch.randn((2, 5, 5, 3, 3, 3), requires_grad=True)    
    b = torch.randn((2, 5, 5), requires_grad=True)    

    y = conv2d_local(x, W, b)
    t = torch.randn((2, 2, 5, 5), requires_grad=True)
    loss = y - t
    loss = torch.sum(loss)
    loss.backward()
    y.backward()
    from torch.autograd import gradcheck
    print(gradcheck(conv2d_local, [x, W]))
    
