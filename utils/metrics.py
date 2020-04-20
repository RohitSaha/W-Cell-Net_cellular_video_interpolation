import tensorflow as tf
import math as pymath

from utils.losses import l2_loss

def metric_repeat_fframe(fframes,mid_frames):
	'''
	Assumes we predict the first frame for all 
	intermediate ones, then takes the difference
	by broadcasting along the inter_frames axis

	Args:
		fframes: tensor, [B,H,W,1]
		mid_frames: tensor, [B,inter_frames,H,W,1]

	Output:
		l2 loss between predicion and ground truth
	'''

	return [l2_loss(mid_frames,
				tf.expand_dims(fframes,axis=1)),\
			compute_psnr(mid_frames,
				tf.expand_dims(fframes,axis=1))]


def metric_repeat_lframe(lframes, mid_frames):
	'''
	Assumes we predict the last frame for all 
	intermediate ones, then takes the difference
	by broadcasting along the inter_frames axis

	Args:
		mid_frames: tensor, [B,inter_frames,H,W,1]
		lframes: tensor, [B,H,W,1]

	Output:
		l2 loss between predicion and ground truth
	'''

	return [l2_loss(mid_frames,
				tf.expand_dims(lframes,axis=1)),\
			compute_psnr(mid_frames,
				tf.expand_dims(lframes,axis=1))]



def metric_weighted_frame(fframes,mid_frames,lframes):
	'''
	Do a weighted interpolation between first frame 
	and the last. Then take their difference between 
	the weighted sum and the true intermediate frames

	Args:
		fframes: tensor, [B,H,W,1]
		mid_frames: tensor, [B,inter_frames,H,W,1]
		lframes: tensor, [B,H,W,1]

	Output:
		l2 loss between weighted images and ground truth
	'''

	inter_frames = mid_frames.get_shape()[1]


	fframes_expanded = tf.expand_dims(fframes,axis=1)
	lframes_expanded = tf.expand_dims(lframes,axis=1)

	fframes_tiled = tf.tile(fframes_expanded, 
		[1,inter_frames,1,1,1])
	lframes_tiled = tf.tile(lframes_expanded, 
		[1,inter_frames,1,1,1])


	weighting = tf.range(1.0,inter_frames+1)
	weighting = tf.reshape(weighting, 
		[1,inter_frames,1,1,1])/tf.cast(
		(inter_frames+1),dtype=tf.float32)

	weighted_sum = (fframes_tiled * weighting + 
		lframes_tiled *(1-weighting))

	return [l2_loss(mid_frames,weighted_sum),\
			compute_psnr(mid_frames,weighted_sum)]



def metric_interpolated_frame(mid_frames,
	rec_mid_frames):
	'''
	Take the difference between the true frames 
	and the networks interpolation

	Args:
		mid_frames: tensor, [B,inter_frames,H,W,1]
		rec_mid_frames: tensor, [B,inter_frames,H,W,1]
		

	Output:
		l2 loss between predicion and ground truth
	'''

	return [l2_loss(mid_frames,rec_mid_frames),\
			compute_psnr(mid_frames,rec_mid_frames)]

def compute_psnr(ref, target):
	'''
	Computes PSNR between the true frames 
	and the networks interpolation

	Args:
		rec: 'Tensor', [B,inter_frames,H,W,1]
		target: 'Tensor', [B,inter_frames,H,W,1]
		
	Output:
		PSNR between predicion and ground truth
	'''

    diff = target - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.reduce_prod(diff.get_shape())
    mse = err / tf.cast(v, tf.float32) + 1e-12
    psnr = 10. * (tf.log(255. * 255. / mse) / tf.log(10.))

    return psnr
