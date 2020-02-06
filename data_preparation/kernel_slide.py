from PIL import Image
import numpy as np
import cv2

def get_cell_patch(image, kernel_size, unique_id):
    '''Finds the location of the cell using
    the mask image
    Args:
        image: 'Numpy' matrix of the mask
        kernel_size: 'Tuple' to mention the size
            of the kernel window
        unique_id: 'Float' to mention the class id
            of the cell of interest
    Returns:
        image: 'Numpy' matrix of tight bounding box
            segmented cell
        new_h: 'Integer' that specifies the y_coord
            starting point of the cell
        new_w: 'Integer' that specifies the x_coord
            starting point of the cell
    '''
    height, width = image.shape

    height -= kernel_size[0]
    width -= kernel_size[1]

    kernel = np.ones(
        kernel_size,
        dtype=np.float32) * unique_id

    max_count = 0

    for h in range(height):
        for w in range(width):
            patch = image[
                h : h + kernel_size[0],
                w : w + kernel_size[1]]

            similarity_matrix = patch == kernel

            count = np.sum(similarity_matrix)

            if count > max_count:
                store_h = h
                store_w = w
                max_count = count 

    image = image[
        store_h : store_h + kernel_size[0],
        store_w : store_w + kernel_size[1]]

    image, shift_h, shift_w = get_tight_bounding_box(
        image,
        unique_id)

    new_h = store_h + shift_h
    new_w = store_w + shift_w

    return image, new_h, new_w


def get_tight_bounding_box(image, unique_id):
    '''Computes a tight bounding box over the 
    cell of interest
    Args:
        image: 'Numpy' matrix of cropped segmented
            cell
        unique_id: 'Float' to mention the class id
            of the cell of interest
    Returns:
        image: 'Numpy' matrix of tight bounding box
            segmented cell
        h: 'Integer' that mentions the adjusted y_coord
        w: 'Integer' that mentions the adjusted x_coord
    ''' 
    height, width = image.shape

    for h in range(height):
        row = image[h, :]
        if unique_id in row:
            break

    for w in range(width):
        column = image[:, w]
        if unique_id in column:
            break

    image = image[h:, w:]
    
    return image, h, w 

