import tensorflow as tf
import math as pymath
from models.utils.losses import l2_loss

def gaussian_kernel (k_size=(7,7),mean=0,std=1):
	'''
	creates 2 probability ditributions 
	(p_dist_1,p_dist_2) with lengths determined 
	by k_size and the does the outer product

	Args:
		k_size: tuple, [kernel_height, kernel_width]
		mean: scalar, gaussian distribution mean
		std: scalar, gaussian distribution deviation

	Output:
		gauss_kernel: [kernel_height, kernel_width]
					gaussian weights
	'''

	p_dist_1 = tf.map_fn(lambda x : 
		tf.math.exp(-0.5*((x-mean)/std)**2)/(
			std*tf.math.sqrt(2*pymath.pi)),
		tf.range((-k_size[0]+1)//2,(k_size[0]+1)//2,
			dtype=tf.float32))

	p_dist_2 = tf.map_fn(lambda x : 
		tf.math.exp(-0.5*((x-mean)/std)**2)/(
			std*tf.math.sqrt(2*m.pi)),
		tf.range((-k_size[1]+1)//2,(k_size[1]+1)//2,
			dtype=tf.float32))

	gauss_kernel = tf.einsum('i,j->ij',p_dist_1, 
		p_dist_2)

	return gauss_kernel/tf.math.reduce_sum(
		gauss_kernel)

def gaussian_filter(img,k_size=(7,7),mean=0,std=3):
	
	'''
	Create a gaussian kernel of shape k_size and
	convolve with image to generate filtered image

	Args:
		k_size: tuple, [kernel_height, kernel_width]
		mean: scalar, gaussian distribution mean
		std: scalar, gaussian distribution deviation

	Output:
		blurred_image: same shape as img
	'''

	width = k_size[0]
	height = k_size[1]

	g_kernel = gaussian_kernel(k_size=k_size,
		mean=mean,std=std)

	blurred_img = tf.nn.conv2d(img,
		tf.reshape(g_kernel,[width,height,1,1]),
		strides=[1,1,1,1],padding='SAME')

	blurred_img = tf.stop_gradient(blurred_img)

	return blurred_img


def metric_repeat_fframe(fframes,mid_frames):
	'''
	Blurr both first and intermediate images then
	take their difference, assuming we predict
	the first frame for all intermediate ones

	Args:
		fframes: tensor, [B,H,W,1]
		mid_frames: tensor, [B,inter_frames,H,W,1]

	Output:
		l2 loss between predicion and ground truth
	'''

	batch_size = mid_frames.get_shape()[0]
	inter_frames = mid_frames.get_shape()[1]

	mid_frames = tf.reshape(tf.transpose(
		mid_frames,perm=[0,2,1,3,4]),
	[batch_size,100,-1,1])

	fframes = gaussian_filter(fframes,k_size=(7,7),
		mean=0,std=3)
	mid_frames = gaussian_filter(mid_frames,
		k_size=(7,7),mean=0,std=3)

	fframes_tiled = tf.tile(fframes,
		[1,1,inter_frames,1])


	return l2_loss(mid_frames,fframes_tiled)


def metric_repeat_lframe(mid_frames,lframes):
	'''
	Blurr both last and intermediate images then
	take their difference, assuming we predict
	the last frame for all intermediate ones

	Args:
		mid_frames: tensor, [B,inter_frames,H,W,1]
		lframes: tensor, [B,H,W,1]

	Output:
		l2 loss between predicion and ground truth
	'''

	batch_size = mid_frames.get_shape()[0]
	inter_frames = mid_frames.get_shape()[1]

	mid_frames = tf.reshape(tf.transpose(
		mid_frames,perm=[0,2,1,3,4]),
	[batch_size,100,-1,1])

	lframes = gaussian_filter(lframes,k_size=(7,7),
		mean=0,std=3)
	mid_frames = gaussian_filter(mid_frames,
		k_size=(7,7),mean=0,std=3)

	lframes_tiled = tf.tile(lframes, 
		[1,1,inter_frames,1])

	return l2_loss(mid_frames,lframes_tiled)



def metric_weighted_frame(fframes,mid_frames,lframes):
	'''
	Blurr all true frames then do a weighted 
	interpolation between first frame and last.
	Then take their difference between the weighted
	sum and the true intermediate frames

	Args:
		fframes: tensor, [B,H,W,1]
		mid_frames: tensor, [B,inter_frames,H,W,1]
		lframes: tensor, [B,H,W,1]

	Output:
		l2 loss between predicion and ground truth
	'''
	batch_size = mid_frames.get_shape()[0]
	inter_frames = mid_frames.get_shape()[1]

	mid_frames = tf.reshape(tf.transpose(
		mid_frames,perm=[0,2,1,3,4]),
	[batch_size,100,-1,1])

	fframes = gaussian_filter(fframes,k_size=(7,7),
		mean=0,std=3)
	mid_frames = gaussian_filter(mid_frames,
		k_size=(7,7),mean=0,std=3)
	lframes = gaussian_filter(lframes,k_size=(7,7),
		std=3)

	fframes_expanded = tf.expand_dims(fframes,axis=1)
	lframes_expanded = tf.expand_dims(lframes,axis=1)

	fframes_tiled = tf.tile(fframes_expanded, 
		[1,inter_frames,1,1,1])
	lframes_tiled = tf.tile(lframes_expanded, 
		[1,inter_frames,1,1,1])
	mid_frames = tf.transpose(tf.reshape(mid_frames,
		[batch_size,100,-1,100,1]),perm=[0,2,1,3,4])

	weighting = tf.range(1.0,inter_frames+1)
	weighting = tf.reshape(weighting, 
		[1,inter_frames,1,1,1])

	weighted_sum = (fframes_tiled * weighting + 
		lframes_tiled *(1-weighting))/tf.cast(
		(inter_frames+1),dtype=tf.float32)

	return l2_loss(mid_frames,weighted_sum)



def metric_interpolated_frame(mid_frames,
	rec_mid_frames):
	'''
	Blurr the true frames then take the difference 
	between the blurred frames and the networks
	interpolation

	Args:
		mid_frames: tensor, [B,inter_frames,H,W,1]
		rec_mid_frames: tensor, [B,inter_frames,H,W,1]
		

	Output:
		l2 loss between predicion and ground truth
	'''
	batch_size = mid_frames.get_shape()[0]

	mid_frames = tf.reshape(tf.transpose(
		mid_frames,perm=[0,2,1,3,4]),
	[batch_size,100,-1,1])

	mid_frames = gaussian_filter(mid_frames,
		k_size=(7,7),mean=0,std=3)

	rec_mid_frames = tf.reshape(tf.transpose(
		rec_mid_frames,perm=[0,2,1,3,4]),
	[batch_size,100,-1,1])

	return l2_loss(mid_frames,rec_mid_frames)


	








