import tensorflow as tf

import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_frames(start_frames, end_frames,
                    mid_frames, rec_mid_frames,
                    iteration=100, save_path='',
                    training=False, num_plots=3):
    '''
    Args
    	start_frames: (batch_size X height X width X 1 )
    	end_frames: (batch_size X height X width X 1)
    	mid_frames: ground truth intermediate frames
    				(batch_size X # inter_frames X 
    				height X width X 1)
    	rec_mid_frames: generated intermediate frames
    				(batch_size X # inter_frames X 
    				height X width X 1)
        iteration: current train or validation iteration
        training : plot train or valid frames
        num_plots: number of samples to plot
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
        	# row 1, 3 ... don't show start and end frames
            axes[row,col].text(.5,.5,'Generated')
            axes[row, num_cols-1].axis("off")
            axes[row,num_cols-1].text(0,.5,'Frames')
        else:
            # update the scale for imshow
            v_min = np.min(start_images[idx])
            v_min = np.min([np.min(end_images[idx]),v_min])
            v_min = np.min([np.min(true_mid_images[idx]),v_min])
            # v_min = np.min([np.min(gen_mid_images[idx]),v_min])

            v_max = np.max(start_images[idx])
            v_max = np.max([np.max(end_images[idx]),v_max])
            v_max = np.max([np.max(true_mid_images[idx]),v_max])
            # v_max = np.max([np.max(gen_mid_images[idx]),v_max])

            print('v_min is ',v_min)
            print('v_max is ',v_max)
        	# row 0, 2 ... show start and end frames 
            axes[row, col].imshow(
                start_images[idx],
                cmap="gray",
                vmin=v_min,
                vmax=v_max,
                aspect="auto")
            axes[row, num_cols-1].axis("off")
            axes[row, num_cols-1].imshow(
                end_images[idx],
                cmap="gray",
                vmin=v_min,
                vmax=v_max,
                aspect="auto")

            
        for col in range (1,num_cols-1):
            axes[row, col].axis("off")
            if even_row:
            	# row 1, 3 ... show generated frames
                axes[row, col].imshow(
                    gen_mid_images[idx,col-1],
                    cmap="gray",
                    vmin=v_min,
                    vmax=v_max,
                    aspect="auto")

            else:
            	# row 0, 2 ... show true frames
                axes[row, col].imshow(
                    true_mid_images[idx,col-1],
                    cmap="gray",
                    vmin=v_min,
                    vmax=v_max,
                    aspect="auto")
                
    plt.draw()

    plt.subplots_adjust(
        wspace=.02,
        hspace=.05)

    filename = ['validation','training'][training]\
        +'_iteration_'\
        +str(iteration)+'.png' 

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(save_path + filename)

    return

def visualize_tensorboard(start_frames, end_frames,
    mid_frames, rec_mid_frames, num_plots = 3):
    
    '''
    Args
        start_frames: (batch_size X height X width X 1 )
        end_frames: (batch_size X height X width X 1)
        mid_frames: ground truth intermediate frames
                    (batch_size X # inter_frames X 
                    height X width X 1)
        rec_mid_frames: generated intermediate frames
                    (batch_size X # inter_frames X 
                    height X width X 1)
        num_plots: number of samples to plot

    return:
        true_images: (1 X num_plots* height X 
                    (#inter_frames+2) * width, 1 )
        fake_images: (1 X num_plots* height X 
                    (#inter_frames+2) * width, 1 )

    '''

    # Get original shape of end and mid frames
    input_shape = start_frames.get_shape().as_list()
    mid_shape = mid_frames.get_shape().as_list()

    # smallest of num_plots and batch_size
    num_samples = tf.math.minimum(\
   		num_plots,mid_shape[0])

    # Final output image shapes 
    #   (1,batch_size * h,w,1)
    start_frame_new_shape = [1,\
        num_samples*input_shape[1],\
        input_shape[2],1]
    #   (1,batch_size * h,#inter_frames * w,1)
    mid_frame_new_shape = [1,\
        num_samples*mid_shape[2],\
        mid_shape[1]*mid_shape[3], 1]
    
    # subsample and reshape
    sampled_start_frames = \
        tf.reshape(start_frames[0:num_samples],\
        start_frame_new_shape)
    sampled_end_frames = \
        tf.reshape(end_frames[0:num_samples],\
        start_frame_new_shape)

    # subsample, concatenate, and then reshape
    sampled_mid_frames = \
        tf.reshape(tf.concat([mid_frames\
            [0:num_samples,i:i+1,:,:]\
            for i in range(mid_shape[1])],axis=3),\
        mid_frame_new_shape)
    sampled_rec_frames = \
        tf.reshape(tf.concat([rec_mid_frames\
            [0:num_samples,i:i+1,:,:]\
            for i in range(mid_shape[1])],axis=3),\
        mid_frame_new_shape)

    # concatenate to form 
    #   (1,batch_size*h,(2+#inter_frames)*w,1)
    true_images = tf.concat([sampled_start_frames,\
        sampled_mid_frames, sampled_end_frames],\
        axis = 2)
    fake_images = tf.concat([sampled_start_frames,\
        sampled_rec_frames, sampled_end_frames],\
        axis = 2)
    
    img_min = tf.math.reduce_min(true_images)
    img_max = tf.math.reduce_max(true_images)
    fk_img_min = tf.math.reduce_min(fake_images)
    fk_img_max = tf.math.reduce_max(fake_images)
    
    fk_img_range = fk_img_max - fk_img_min
    img_range = img_max - img_min

    true_images_scaled = 255 * \
        (true_images - img_min) / img_range
    fake_images_scaled = 255 * \
        (fake_images - fk_img_min)/ fk_img_range

    return (true_images_scaled,fake_images_scaled)
