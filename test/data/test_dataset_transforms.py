import numpy as np
from PIL import Image

import torchex.data.transforms as T

def test_transform_random_resize():
    np.random.seed(10)
    img_array = np.random.random((100, 100))
    pil_img = Image.fromarray(np.uint8(img_array))
    t = T.RandomResize(10, 30)
    assert t(pil_img).size == (12, 28)

    
