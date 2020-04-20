import os
import cv2
import numpy as np

def pad_image(image, targetHeight, targetWidth):        
    '''Resizes the image along the longer dimension
    and pads the shorter dimension on both sides
    equally.
    Args:
        image: 'Numpy' matrix of dtype np.float32
        targetHeight: 'Integer' specifying the height
            of the image
        targetWidth: 'Integer' specifying the width
            of the image
    Returns:
        Resized and padded image
    '''
    height, width  = image.shape
        
    height_diff = targetHeight - height
    width_diff = targetWidth - width

    if height_diff % 2 == 0:
        pad_top = height_diff // 2
        pad_bottom = height_diff // 2
    else:
        pad_top = height_diff // 2
        pad_bottom = pad_top + 1

    if width_diff % 2 == 0:
        pad_left = width_diff // 2
        pad_right = width_diff // 2
    else:
        pad_left = width_diff // 2
        pad_right = pad_left + 1

    image = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT)
     
       
    get_shape = image.shape

    return image


