import numpy as np
from PIL import Image
import cv2

from kernel_slide import get_cell_patch

def get_unique_ids(image):
    '''Returns the unique elements in a
    matrix.
    Args:
        :image: 'Numpy' matrix
    returns:
        Unique values in a matrix of type
        'Numpy'
    '''
    unique_ids = np.unique(
        image)

    # :unique_ids include '0' in the first index.
    # We want to ignore since '0' represents
    # background.
    return unique_ids[1:]

# LEGACY
def get_mask_for_id(image, unique_id):
    '''Returns a patch, that captures a cell
    given the id.
    Args:

    '''
    image = np.where(
        image == unique_id,
        1,
        0)

    unique = np.unique(
        image)

    assert len(unique) == 2, (':unique should\
        return 2 values, found:{}'.format(len(unique)))

    return image


def process_image(filename, fl_filename, kernel_size):
    image = Image.open(
        filename)
    image = np.array(
        image)
    #fl_image = Image.open(
    #    fl_filename)
    #fl_image = np.array(
    #    fl_image)

    unique_ids = get_unique_ids(
        image)
    
    ctr = 0
    threshold = 500

    created_image = np.zeros(
        image.shape)

    for unique_id in unique_ids:
        mask = np.where(
            image == unique_id,
            1,
            0)

        count = np.sum(mask)

        if count > threshold:
            cell_patch, loc_h, loc_w = get_cell_patch( 
                image,
                kernel_size,
                unique_id)

            # :loc_h and :loc_w are the (y, x) coordinates
            # of where the cell starts.

            # Fetch the required cell from fluorescence
            # image using :loc_h and :loc_w. Pad the matrix
            # with zeros on all sides to allow some slack.
            # This is done because the cell in subsequent
            # frames might grow/shrink. Therefore, the
            # coordinated should be relaxed.
            start_h = loc_h - 15
            end_h = loc_h + kernel_size[0] + 15
            start_w = loc_w - 15 
            end_w = loc_w + kernel_size[1] + 15
            
            #fl_cell_patch = fl_image[
            #    start_h : end_h,
            #    start_w : end_w]
            
            # For debugging purposes
            cell_patch = np.where(
                cell_patch == unique_id,
                250,
                0)

            # created_image = created_image + masked_image

            cv2.imwrite(
                path + 'pics_2/demo_{}.png'.format(str(ctr)),
                cell_patch)

            break

        ctr += 1        

        if ctr % 500 == 0:
            print('Processed {}/{} ids'.format(
                ctr, len(unique_ids)))

        # cv2.imwrite('generated.png', created_image)

kernel_size = (80, 80)
path = '/Users/rohitsaha/Documents/Spring 2020/CSC2516HS/project/'

filename = path + 'mask_601z1c2.tif'
fl_filename = path + 'fluorescence_601z1c2.tif' 
process_image(filename, fl_filename, kernel_size)

