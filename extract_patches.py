import numpy as np
from PIL import Image
import cv2

from kernel_slide import get_cell_patch

import pickle
import argparse
import time

#TODO: change default folders based on Valrhona dir struct

def get_unique_ids(image):
    '''Returns the unique elements in a
    matrix.
    Args:
        image: 'Numpy' matrix
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


def get_coordinates(filename, kernel_size):
    '''Get coordinates of individual cells using a
    mask image
    Args:
        filename: 'String' that points to the location
            of the file
        kernel_size: 'Tuple' that contains the size of
            the kernel window
    '''
    image = Image.open(
        filename)
    image = np.array(
        image)

    unique_ids = get_unique_ids(
        image)
    import ipdb; ipdb.set_trace()    
    unique_id_to_coordinates = {}

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

            
            
            unique_id_to_coordinates[unique_id] = (
                loc_h, loc_w)
            
            
            #fl_cell_patch = fl_image[
            #    start_h : end_h,
            #    start_w : end_w]
            
            # For debugging purposes
            cell_patch = np.where(
                cell_patch == unique_id,
                250,
                0)

            # created_image = created_image + masked_image


        ctr += 1        

        if ctr % 500 == 0:
            print('Processed {}/{} ids'.format(
                ctr, len(unique_ids)))

    return unique_id_to_coordinates

def get_cell_patch(fl_filename, unique_id_to_coordinates,
                    slack=15, kernel_size=(80, 80),
                    IMAGE_DIR='', image_counter=0):
    '''Uses coordinates to extract cell patches from the
    fluorescent image
    Args:
        fl_filename: 'String' that has the path to the
            fluorescent image
        unique_id_to_coordinates: 'Dict' with unique_id as
            keys and coordinates as values
        slack: 'Integer' to mention allowance while cropping
        kernel_size: 'Tuple' that contains the size of the
            kernel window
        IMAGE_DIR: 'String' that has the path where the
            extracted cell patch will get saved
        image_counter: 'Integer' that counts the sequence
            of images
    Returns:
        None
    ''' 
    fl_image = Image.open(
        fl_filename)
    fl_image = np.array(
        fl_image)

    for unique_id in unique_id_to_coordinates.keys():
        loc_h, loc_w = unique_id_to_coordinates[unique_id]    
        
        # Fetch the required cell from fluorescence
        # image using :loc_h and :loc_w. Pad the matrix
        # with zeros on all sides to allow some slack.
        # This is done because the cell in subsequent
        # frames might grow/shrink. Therefore, the
        # coordinated should be relaxed.
        start_h = loc_h - slack
        end_h = loc_h + kernel_size[0] + slack
        start_w = loc_w - slack 
        end_w = loc_w + kernel_size[1] + slack

        cell_patch = fl_image[
            start_h : end_h,
            start_w : end_w]

        # saving the image to disk
        unique_id_str = str(unique_id).replace('.', '-')          
 
        image_path = os.path.join(
            IMAGE_DIR,
            unique_id_str)

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        cv2.imwrite(
            image_path + '/image_{}.png'.format(
                str(image_counter)),
            cell_patch)


def control(args):
    '''Interface function to extract patches and save
    them in respective folders
    Args:
        args: 'argparse' instance
    Returns:
        None
    '''
    start = time.time()

    # Get coordinates of each cell using a mask image
    path_to_mask = 'something'
    kernel_size = (args.kernel_size, args.kernel_size)
    unique_id_to_coordinates = get_coordinates(
        path_to_mask,
        kernel_size)

    # save :unique_id_to_coordinates to disk
    with open(
        IMAGE_DIR + '/meta_file/unique_id_to_coord.pkl',
        'wb') as handle:
        pickle.dump(
            unique_id_to_coordinates,
            handle)
    print('Coordinates calculated.....Pickle file dumped.....')

    # Extract cells from fluorescent image using the
    # above coordinates and write each extracted patch
    # to respective folder
    fl_files = os.listdir(args.FL_DIR)

    # Filter out files that have 'z1c1' in them
    fl_files = [
        args.FL_DIR + '/' + fl_file
        for fl_file in fl_files
        if 'z1c1' in fl_file]
    fl_files = sorted(fl_files)

    image_counter = 0

    for fl_file in fl_files:
        
        get_cell_patch(
            fl_file,
            unique_id_to_coordinates,
            slack=args.slack,
            kernel_size=kernel_size,
            IMAGE_DIR=args.IMAGE_DIR,
            image_counter=image_counter)
            
        if image_counter % 50 == 0:
            print('Finished writing {}/{} images for each\
                cell'.format(image_counter, len(fl_files)))

        image_counter += 1

    print('Process complete.....Time taken:{} seconds..'.format(
        str(round(time.time() - start, 3))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='params of running the experiment')

    parser.add_argument(
        '--FL_DIR',
        type=str,
        help='path to Fluorescent images')

    parser.add_argument(
        '--IMAGE_DIR',
        type=str,
        help='path where images will get saved')

    parser.add_argument(
        '--kernel_size',
        type=int,
        default=80,
        help='size of kernel window')

    parser.add_argument(
        '--slack',
        type=int,
        default=10,
        help='allowance while cropping')

    args = parser.parse_args()
    control(args)

kernel_size = (80, 80)
path = '/Users/rohitsaha/Documents/Spring 2020/CSC2516HS/project/'

filename = path + 'mask_601z1c2.tif'
fl_filename = path + 'fluorescence_601z1c2.tif' 
get_coordinates(filename, kernel_size)

