import numpy as np
import pytest
from PIL import Image

import torchex.data.functional as F


def test_pad_random_sift():
    np.random.seed(10)
    img_array = np.random.random((20, 20))
    pil_img = Image.fromarray(np.uint8(img_array))
    out_img = F.pad_random_sift(pil_img, 100)
    assert out_img.size == (100, 100)

def test_random_resize():
    np.random.seed(10)
    img_array = np.random.random((100, 100))
    pil_img = Image.fromarray(np.uint8(img_array))
    out_img = F.random_resize(pil_img, 10, 30)
    assert out_img.size == (12, 28)

def test_random_resize_with_one_list_arguments():
    np.random.seed(10)
    img_array = np.random.random((100, 100))
    pil_img = Image.fromarray(np.uint8(img_array))
    out_img = F.random_resize(pil_img, 10, [10, 30])
    assert out_img.size == (28, 10)

def test_random_resize_with_both_list_arguments():
    np.random.seed(10)
    img_array = np.random.random((100, 100))
    pil_img = Image.fromarray(np.uint8(img_array))
    with pytest.raises(ValueError):    
        out_img = F.random_resize(pil_img, [10, 100], [100, 30])
    


