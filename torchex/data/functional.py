import torch
import sys
import math
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import collections
import warnings

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def random_resize(img, min_size, max_size, equal_aspect, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        min_sizes (sequence[h, w] or int): Desired output minimum size. 
        max_sizes (sequence[h, w] or int): Desired output maximum size. 
        equal_aspect (bool, optional) : 
    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(min_size, int) or (isinstance(min_size, Iterable) and len(min_size) == 2)):
        raise TypeError('Got inappropriate min_size arg: {}'.format(min_size))
    if not (isinstance(max_size, int) or (isinstance(max_size, Iterable) and len(max_size) == 2)):
        raise TypeError('Got inappropriate max_size arg: {}'.format(max_size))
    min_size = [min_size, min_size] if isinstance(min_size, int) else min_size
    max_size = [max_size, max_size] if isinstance(max_size, int) else max_size

    size = []
    for i in range(2):
        if min_size[i] == max_size[i]:
            sample = min_size[i]
        elif min_size[i] > max_size[i]:
            raise ValueError('min_size[%d] (%d) must be lower than max_size[%d] (%d).' % (
                i, min_size[i], i, max_size[i]))
        else:
            sample = np.random.choice(range(min_size[i], max_size[i]))
        size.append(sample)

    if equal_aspect:
        size[1] = size[0]
    
    return img.resize(size[::-1], interpolation)
