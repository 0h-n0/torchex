from PIL import Image

from . import functional as F


class Sift(object):
    pass

class RandomSift(object):
    pass

class PadRandomSift(object):
    r""" Pad and randomly sift input PIL Image to the given size.
    
    """
    def __init__(self, canvas_size, fill=0):
        self.canvas_size = canvas_size
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return F.pad_random_sift(img, self.canvas_size, self.fill)

    def __repr__(self):
        return self.__class__.__name__ + '(canvas_size={0}, fill={1})'.format(self.canvas_size, self.fill)

class RandomResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        min_size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. 
        max_size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. 
        equal_aspect (bool, optional): height and width are equally resized.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, min_size, max_size, equal_aspect=False, interpolation=Image.BILINEAR):
        assert isinstance(min_size, int) or \
            (isinstance(min_size, Iterable) and \
             len(min_size) == 2)
        
        assert isinstance(max_size, int) or \
            (isinstance(max_size, Iterable) and \
             len(max_size) == 2)
        
        self.min_size = min_size
        self.max_size = max_size
        self.equal_aspect = equal_aspect
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return F.random_resize(img, self.min_size, self.max_size,
                               self.equal_aspect, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(min_size={0}, max_size={1}, interpolation={2})'.format(self.min_size, self.max_size, interpolate_str)
