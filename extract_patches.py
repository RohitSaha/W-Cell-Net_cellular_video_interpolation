import numpy as np
from PIL import Image
import cv2

def get_unique_ids(image):
    '''Returns the unique elements in a
    matrix.
    Args:
        :image: 'Numpy' matrix
    returns:
        Unique values in a matrix of type
        'Numpy'
    '''
    counts = np.unique(
        image)

    # :counts include '0' in the first index.
    # We want to ignore since '0' represents
    # background.
    return counts[1:]

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

def get_patch(masked_image):
    # Get the patch with the cell of importance is
    # in the center.
    #TODO

def process_image(filename, fl_filename):
    image = Image.open(
        filename)
    image = np.array(
        image)
    fl_image = Image.open(
        fl_filename)
    fl_image = np.array(
        fl_image)

    unique_ids = get_unique_ids(
        image)
    
    ctr = 0
    threshold = 500

    created_image = np.zeros(
        image.shape)

    for unique_id in unique_ids:
        mask = get_mask_for_id(
            image,
            unique_id)

        masked_image = np.multiply(
            image,
            mask)

        # There are many false positives in some
        # of the :masked_image. One possible idea
        # is to count the number of non-zero elements
        # after masking and consider the ones that
        # have counts > threshold.
        counts = np.count_nonzero(
            masked_image == unique_id)

        
        if counts > threshold:
            masked_image = np.where(
                masked_image == unique_id,
                250,
                0)

            created_image = created_image + masked_image

            # centered_image = get_patch(
            #    masked_image)

            # cv2.imwrite('pics/demo_{}.png'.format(str(ctr)), masked_image)

        ctr += 1        

        if ctr % 500 == 0:
            print('Processed {}/{} ids'.format(
                ctr, len(unique_ids)))

        cv2.imwrite('generated.png', created_image)

filename = 'mask_601z1c2.tif'
fl_filename = 'fluorescence_601z1c2.tif' 
process_image(filename, fl_image)

