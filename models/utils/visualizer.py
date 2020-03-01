def visualize_frames(iteration, every = 1000, training = False, num_plots = 3):
	'''
	Args
		iteration: current iteration on training or validation
		every: integer that enables plots to be generated when iteration % every == 0 
		training : determins which frames need to be run in tf.session
		num_plots : integer to indicate the number of samples to plot
	'''

	if iteration % every == 0:
		
		import tensorflow as tf
		import matplotlib.pyplot as plt
		import numpy as np

		with tf.Session() as sess:
			if training:
				start_frames, end_frames, mid_frames, rec_mid_frames \
				= sess.run([train_fFrames, train_lFrames,train_iFrames,train_rec_iFrames])
			else:
				start_frames, end_frames, mid_frames, rec_mid_frames \
				 = sess.run([val_fFrames, val_lFrames,val_iFrames,val_rec_iFrames])

			num_samples = np.minimum(num_plots, mid_frames.shape[0])
			
			num_rows = 2 * num_samples # num samples * 2 because of the true images
			num_cols = 2 + mid_frames.shape[1] # start+ end + num_midframes

			selected_samples = np.random.choice(mid_frames.shape[0], num_samples, replace = False) # array of shuffled indicies

			# subsample the batch for plotting
			start_images = start_frames[selected_samples,:,:,0] #num_samples X 100 X 100
			end_images = end_frames[selected_samples,:,:,0] #num_samples X 100 X 100
			gen_mid_images = rec_mid_frames[selected_samples,:,:,:,0] #num_samples X 3 X 100 X 100
			true_mid_images = mid_frames[selected_samples,:,:,:,0] #num_samples X 3 X 100 X 100

			fig, axes = plt.subplots(nrows = num_rows, ncols = num_cols, figsize = (10,10)) 

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
					axes[row, col].imshow(start_images[idx], cmap="gray", aspect="auto")
					axes[row, num_cols-1].axis("off")
					axes[row, num_cols-1].imshow(end_images[idx], cmap="gray", aspect="auto")

				for col in range (1,num_cols-1):
					axes[row, col].axis("off")
					if even_row:
						axes[row, col].imshow(true_mid_images[idx,col-1], cmap="gray", aspect="auto")
					else:
						axes[row, col].imshow(gen_mid_images[idx,col-1], cmap="gray", aspect="auto")


			plt.subplots_adjust(wspace=.02, hspace=.05)
			plt.draw()
			filename = ['validation','training'][training]+'_iteration_'+str(iteration)+'.png' 
			plt.savefig(filename)

	return






