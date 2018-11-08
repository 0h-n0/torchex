
def crop_2d(input,
            crop_left: int=0, crop_right: int=0,
            crop_top: int=0, crop_bottom: int=0):
    assert input.dim() == 4, 'only support Input(B, C, W, H)'
    B, C, W, H = input.size()
    return input[:, :,
                 crop_left:(W-crop_right),
                 crop_bottom:(H-crop_top)]

def crop_3d(input, crop_size: int=0):
    assert input.dim() == 5, 'only support Input(B, C, D, W, H)'
    B, C, D, W, H = input.size()
    return input[:, :,
                 crop_size:(D-crop_size),
                 crop_size:(W-crop_size),
                 crop_size:(H-crop_size)]

    
