import tensorflow as tf

import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_frames(start_frames, end_frames,
                    mid_frames, rec_mid_frames,
                    iteration=100, save_path='',
                    training = False, num_plots = 3):
    '''
    Args
        iteration: current iteration on training or validation
        every: integer that enables plots to be generated when iteration % every == 0 
        training : determins which frames need to be run in tf.session
        num_plots : integer to indicate the number of samples to plot
    '''
    num_samples = np.minimum(
        num_plots,
        mid_frames.shape[0])
    
    # num samples * 2 because of the true images
    num_rows = 2 * num_samples 
    # start+ end + num_midframes
    num_cols = 2 + mid_frames.shape[1]

    selected_samples = np.random.choice(
        mid_frames.shape[0],
        num_samples,
        replace = False) # array of shuffled indicies

    # subsample the batch for plotting

    # num_samples X 100 X 100
    start_images = start_frames[selected_samples,:,:,0] 
    #num_samples X 100 X 100
    end_images = end_frames[selected_samples,:,:,0]
    #num_samples X 3 X 100 X 100
    gen_mid_images = rec_mid_frames[selected_samples,:,:,:,0]
    #num_samples X 3 X 100 X 100
    true_mid_images = mid_frames[selected_samples,:,:,:,0] 

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(10,10)) 

    for row in range (num_rows):

        col = 0
        even_row = row % 2
        idx = row // 2

        axes[row, col].axis("off")
        
        if even_row:
            axes[row,col].text(.5,.5,'Generated')
            axes[row, num_cols-1].axis("off")
            axes[row,num_cols-1].text(0,.5,'Frames')
        else:
            axes[row, col].imshow(
                start_images[idx],
                cmap="gray",
                aspect="auto")
            axes[row, num_cols-1].axis("off")
            axes[row, num_cols-1].imshow(
                end_images[idx],
                cmap="gray",
                aspect="auto")

        for col in range (1,num_cols-1):
            axes[row, col].axis("off")
            if even_row:
                axes[row, col].imshow(
                    true_mid_images[idx,col-1],
                    cmap="gray",
                    aspect="auto")
            else:
                axes[row, col].imshow(
                    gen_mid_images[idx,col-1],
                    cmap="gray",
                    aspect="auto")


    plt.subplots_adjust(
        wspace=.02,
        hspace=.05)
    plt.draw()
    filename = ['validation','training'][training]\
        +'_iteration_'\
        +str(iteration)+'.png' 

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(save_path + filename)

    return



