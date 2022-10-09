import torch

from ilt.constant import TARGET_INTENSITY, MAX_DOSE, MIN_DOSE

def write_image_file(image, dose, sizex=2048, sizey=2048):
    tempimg = torch.zeros([sizey, sizex], dtype=torch.int64)
    pngimg = torch.zeros([sizey, sizex, 4], dtype=torch.int64)
    tempimg[image >= TARGET_INTENSITY] = 252
    tempimg[image < TARGET_INTENSITY] = 0
    if dose == 1.00:
        pngimg[:, :, 0] = 0
        pngimg[:, :, 1] = pngimg[:, :, 2] = tempimg
    elif dose == MAX_DOSE:
        pngimg[:, :, 0] = 0
        pngimg[:, :, 1] = pngimg[:, :, 2] = tempimg/2
    elif dose == MIN_DOSE:
        pngimg[:, :, 1] = 0
        pngimg[:, :, 0] = pngimg[:, :, 2] = tempimg/2
    pngimg[:, :, 3] = 127

    return pngimg