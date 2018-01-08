import lasagne
import numpy as np, copy
import pickle
import skimage.transform
import scipy, time
from skimage.io import imread, imsave
import theano
import theano.tensor as T
from theano import sparse
from lasagne.utils import floatX
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
import scipy.ndimage as spi
import scipy.sparse as sps
import scipy.misc as spm
from PIL import Image


IMAGE_W1 = 700
IMAGE_W2 = 530
def build_model():
	net = {}
	net['input'] = InputLayer((1, 3, IMAGE_W1, IMAGE_W2))
	net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
	net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
	net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
	net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
	net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
	net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
	net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
	net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
	net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
	net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
	net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
	net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
	net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
	net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
	net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
	net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
	net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
	net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
	net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
	net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
	net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')
	return net

def build_model_segs():
	net = {}
	net['input'] = InputLayer((1, 1, IMAGE_W1, IMAGE_W2))
	net['conv1_1'] = PoolLayer(net['input'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['conv1_2'] = PoolLayer(net['conv1_1'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_inc_pad')
	net['conv2_1'] = PoolLayer(net['pool1'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['conv2_2'] = PoolLayer(net['conv2_1'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
	net['conv3_1'] = PoolLayer(net['pool2'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['conv3_2'] = PoolLayer(net['conv3_1'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['conv3_3'] = PoolLayer(net['conv3_2'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['conv3_4'] = PoolLayer(net['conv3_3'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
	net['conv4_1'] = PoolLayer(net['pool3'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['conv4_2'] = PoolLayer(net['conv4_1'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['conv4_3'] = PoolLayer(net['conv4_2'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['conv4_4'] = PoolLayer(net['conv4_3'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
	net['conv5_1'] = PoolLayer(net['pool4'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['conv5_2'] = PoolLayer(net['conv5_1'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['conv5_3'] = PoolLayer(net['conv5_2'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['conv5_4'] = PoolLayer(net['conv5_3'], pool_size=(3,3),stride=(1,1), pad=(1,1), ignore_border=True, mode='average_inc_pad')
	net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')
	return net

def load_seg(content_seg_path, style_seg_path, content_shape, style_shape):
	color_codes = ['BLUE', 'GREEN', 'BLACK', 'WHITE', 'RED', 'YELLOW', 'GREY', 'LIGHT_BLUE', 'PURPLE']
	def _extract_mask(seg, color_str):
		h, w, c = np.shape(seg)
		if color_str == "BLUE":
			mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
			mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
			mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
		elif color_str == "GREEN":
			mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
			mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
			mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
		elif color_str == "BLACK":
			mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
			mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
			mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
		elif color_str == "WHITE":
			mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
			mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
			mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
		elif color_str == "RED":
			mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
			mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
			mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
		elif color_str == "YELLOW":
			mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
			mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
			mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
		elif color_str == "GREY":
			mask_r = np.multiply((seg[:, :, 0] > 0.4).astype(np.uint8), (seg[:, :, 0] < 0.6).astype(np.uint8))
			mask_g = np.multiply((seg[:, :, 1] > 0.4).astype(np.uint8), (seg[:, :, 1] < 0.6).astype(np.uint8))
			mask_b = np.multiply((seg[:, :, 2] > 0.4).astype(np.uint8), (seg[:, :, 2] < 0.6).astype(np.uint8))
		elif color_str == "LIGHT_BLUE":
			mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
			mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
			mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
		elif color_str == "PURPLE":
			mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
			mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
			mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
		return np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32)

	# content_seg = np.array(Image.open(content_seg_path).convert("RGB").resize(content_shape, resample=Image.BILINEAR), dtype=np.float32) / 255.0
	# style_seg = np.array(Image.open(style_seg_path).convert("RGB").resize(style_shape, resample=Image.BILINEAR), dtype=np.float32) / 255.0
	content_seg = np.array(Image.open(content_seg_path).convert("RGB"), dtype=np.float32) / 255.0
	style_seg = np.array(Image.open(style_seg_path).convert("RGB"), dtype=np.float32) / 255.0

	color_content_masks = []
	color_style_masks = []
	for i in xrange(len(color_codes)):
		color_content_masks.append(np.expand_dims(np.expand_dims(_extract_mask(content_seg, color_codes[i]),0),0))
		color_style_masks.append(np.expand_dims(np.expand_dims(_extract_mask(style_seg, color_codes[i]),0),0))
	# print color_content_masks[0].shape, color_style_masks[0].shape
	return color_content_masks, color_style_masks

MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

def getlaplacian1(i_arr, consts, epsilon=1e-5, win_rad=1):
	neb_size = (win_rad * 2 + 1) ** 2
	h, w, c = i_arr.shape
	img_size = w * h
	consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_rad * 2 + 1, win_rad * 2 + 1)))
	indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
	tlen = int((-consts[win_rad:-win_rad, win_rad:-win_rad] + 1).sum() * (neb_size ** 2))
	row_inds = np.zeros(tlen)
	col_inds = np.zeros(tlen)
	vals = np.zeros(tlen)
	l = 0
	for j in range(win_rad, w - win_rad):
		for i in range(win_rad, h - win_rad):
			if consts[i, j]:
				continue
			win_inds = indsM[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1]
			win_inds = win_inds.ravel(order='F')
			win_i = i_arr[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1, :]
			win_i = win_i.reshape((neb_size, c), order='F')
			win_mu = np.mean(win_i, axis=0).reshape(c, 1)
			win_var = np.linalg.inv(np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu, win_mu.T) + epsilon / neb_size * np.identity(c))
			win_i2 = win_i - np.repeat(win_mu.transpose(), neb_size, 0)
			tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size
			ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
			row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
			col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
			vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
			l += neb_size ** 2
	vals = vals.ravel(order='F')[0: l]
	row_inds = row_inds.ravel(order='F')[0: l]
	col_inds = col_inds.ravel(order='F')[0: l]
	a_sparse = sps.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))
	sum_a = a_sparse.sum(axis=1).T.tolist()[0]
	a_sparse = sps.diags([sum_a], [0], shape=(img_size, img_size)) - a_sparse
	return a_sparse

def im2double(im):
	min_val = np.min(im.ravel())
	max_val = np.max(im.ravel())
	return (im.astype('float') - min_val) / (max_val - min_val)

def reshape_img(in_img):
	im = in_img
	if len(im.shape) == 2:
		im = im[:, :, np.newaxis]
		im = np.repeat(im, 3, axis=2)
	# h, w, _ = im.shape
	# if h < w:
	# 	im = skimage.transform.resize(im, (IMAGE_W1, w*IMAGE_W2/h))*255
	# else:
	# 	im = skimage.transform.resize(im, (h*IMAGE_W1/w, IMAGE_W2))*255
	# h, w, _ = im.shape
	# im = im[h//2-IMAGE_W//2:h//2+IMAGE_W//2, w//2-IMAGE_W//2:w//2+IMAGE_W//2]
	return im

def prep_image(im):
	if len(im.shape) == 2:
		im = im[:, :, np.newaxis]
		im = np.repeat(im, 3, axis=2)
	# h, w, _ = im.shape
	# if h < w:
	# 	im = skimage.transform.resize(im, (IMAGE_W1, w*IMAGE_W2/h))*255
	# else:
	# 	im = skimage.transform.resize(im, (h*IMAGE_W1/w, IMAGE_W2))*255
	# h, w, _ = im.shape
	# im = im[h//2-IMAGE_W1//2:h//2+IMAGE_W1//2, w//2-IMAGE_W2//2:w//2+IMAGE_W2//2]
	rawim = np.copy(im).astype('uint8')
	im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
	im = im[::-1, :, :]
	im = im - MEAN_VALUES
	return rawim, floatX(im[np.newaxis])

def prep_image_affine(im):
	if len(im.shape) == 2:
		im = im[:, :, np.newaxis]
		im = np.repeat(im, 3, axis=2)
	# h, w, _ = im.shape
	# if h < w:
	# 	im = skimage.transform.resize(im, (IMAGE_W1, w*IMAGE_W2/h))*255
	# else:
	# 	im = skimage.transform.resize(im, (h*IMAGE_W1/w, IMAGE_W2))*255
	# h, w, _ = im.shape
	# im = im[h//2-IMAGE_W1//2:h//2+IMAGE_W1//2, w//2-IMAGE_W2//2:w//2+IMAGE_W2//2]
	rawim = np.copy(im).astype('uint8')
	im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
	im = im[::-1, :, :]
	return rawim, floatX(im)

def gram_matrix(x):
	x = x.flatten(ndim=3)
	g = T.tensordot(x, x, axes=([2], [2]))
	return g


def content_loss(P, X, layer):
	p = P[layer]
	x = X[layer]
	# print type(p), type(x)
	loss = 1./2 * ((x - p)**2).sum()
	return loss


# def style_loss(A, X, layer):
# 	a = A[layer]
# 	x = X[layer]
# 	A = gram_matrix(a)
# 	G = gram_matrix(x)
# 	N = a.shape[1]
# 	M = a.shape[2] * a.shape[3]
# 	loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
# 	return loss

def style_loss(A, X, Am, Xm, layer):
	a = A[layer]
	x = X[layer]
	N = a.shape[1]
	M = a.shape[2] * a.shape[3]
	loss = []
	if layer == 'conv1_1':
		repeats = 64
	elif layer == 'conv2_1':
		repeats= 128
	elif layer == 'conv3_1':
		repeats = 256
	elif layer == 'conv4_1':
		repeats = 512
	elif layer == 'conv5_1':
		repeats = 512

	for i in xrange(len(Am)):
		
		A_ = Am[i][layer].repeat(repeats, axis=1)*a
		G_ = Xm[i][layer].repeat(repeats, axis=1)*x
		A = gram_matrix(A_)
		G = gram_matrix(G_)
		loss.append(1./(4 * N**2 * M**2) * ((G - A)**2).sum())
	return sum(loss)

def total_variation_loss(x):
	return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()

def matting_loss(x,matting):
	x = (x.reshape((3,IMAGE_W1,IMAGE_W2))+MEAN_VALUES)/255.
	v1 = x[0].transpose()
	v2 = x[1].transpose()
	v3 = x[2].transpose()
	w1 = v1.reshape([-1])
	w2 = v2.reshape([-1])
	w3 = v3.reshape([-1])
	y1 = theano.sparse.basic.structured_dot(matting, w1.reshape((-1,1)))
	y2 = theano.sparse.basic.structured_dot(matting, w2.reshape((-1,1)))
	y3 = theano.sparse.basic.structured_dot(matting, w3.reshape((-1,1)))
	z1 = T.dot(w1.reshape((1,-1)),y1)
	z2 = T.dot(w2.reshape((1,-1)),y2)
	z3 = T.dot(w3.reshape((1,-1)),y3)
	loss = z1+z2+z3
	return T.sum(loss)

def eval_loss(x0):
	x0 = floatX(x0.reshape((1, 3, IMAGE_W1, IMAGE_W2)))
	generated_image.set_value(x0)
	return f_loss().astype('float64')

def eval_grad(x0):
	x0 = floatX(x0.reshape((1, 3, IMAGE_W1, IMAGE_W2)))
	generated_image.set_value(x0)
	return np.array(f_grad()).flatten().astype('float64')

def deprocess(x):
	x = np.copy(x[0])
	x += MEAN_VALUES

	x = x[::-1]
	x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
	
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def deprocess_matting(x):
	x = np.copy(x[0])
	x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
	return x

if __name__ == '__main__':
	start_time = time.time()
	
	print ("--- %s seconds ---" % (time.time() - start_time)), "network built. Compiling network."
	net = build_model()
	print ("--- %s seconds ---" % (time.time() - start_time)), "network compiled. loading weights."
	values = pickle.load(open('vgg19_normalized.pkl'))['param values']
	lasagne.layers.set_all_param_values(net['pool5'], values)
	print ("--- %s seconds ---" % (time.time() - start_time)), "weights loaded. preprocessing content and style image."

	photo = imread('in11.png')
	rawim, photo = prep_image(photo)
	# print photo.shape
	art = imread('tar11.png')
	rawim, art = prep_image(art)
	# print art.shape
	print ("--- %s seconds ---" % (time.time() - start_time)), "images and masks preprocessed. extracting content and style features."
	
	layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
	layers = {k: net[k] for k in layers}
	input_im_theano = T.tensor4()
	outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
	photo_features = {k: theano.shared(output.eval({input_im_theano: photo})) for k, output in zip(layers.keys(), outputs)}
	art_features = {k: theano.shared(output.eval({input_im_theano: art})) for k, output in zip(layers.keys(), outputs)}
	print ("--- %s seconds ---" % (time.time() - start_time)), "style and content extracted. generating image."

	seg_net = build_model_segs()
	print ("--- %s seconds ---" % (time.time() - start_time)), "seg network built."
	content_masks, style_masks = load_seg('in11_mask.png', 'tar11_mask.png', (IMAGE_W2,IMAGE_W1), (IMAGE_W2,IMAGE_W1))
	print ("--- %s seconds ---" % (time.time() - start_time)), "segmented images prepared."
	layers_seg = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
	layers_seg = {k: seg_net[k] for k in layers_seg}
	input_im_theano_seg = T.tensor4()
	outputs_seg = lasagne.layers.get_output(layers_seg.values(), input_im_theano_seg)
	content_seg_features = []
	for content_mask in content_masks:
		photo_seg_features = {k: theano.shared(output.eval({input_im_theano_seg: content_mask})) for k, output in zip(layers_seg.keys(), outputs_seg)}
		content_seg_features.append(photo_seg_features)
	style_seg_features = []
	for style_mask in style_masks:
		art_seg_features = {k: theano.shared(output.eval({input_im_theano_seg: style_mask})) for k, output in zip(layers_seg.keys(), outputs_seg)}
		style_seg_features.append(art_seg_features)
	print ("--- %s seconds ---" % (time.time() - start_time)), "segmented features graphed."

	generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W1, IMAGE_W2))))
	gen_features = lasagne.layers.get_output(layers.values(), generated_image)
	gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}
	print ("--- %s seconds ---" % (time.time() - start_time)), "image generated. calculating losses."

	losses = []
	losses.append(0.0065* content_loss(photo_features, gen_features, 'conv4_2'))
	losses.append(0.2e6 * style_loss(art_features, gen_features, style_seg_features, content_seg_features, 'conv1_1'))
	losses.append(0.2e6 * style_loss(art_features, gen_features, style_seg_features, content_seg_features, 'conv2_1'))
	losses.append(0.2e6 * style_loss(art_features, gen_features, style_seg_features, content_seg_features, 'conv3_1'))
	losses.append(0.2e6 * style_loss(art_features, gen_features, style_seg_features, content_seg_features, 'conv4_1'))
	losses.append(0.2e6 * style_loss(art_features, gen_features, style_seg_features, content_seg_features, 'conv5_1'))
	losses.append(0.1e-7 * total_variation_loss(generated_image))
	total_loss = sum(losses)

	grad = T.grad(total_loss, generated_image)
	f_loss = theano.function([], total_loss)
	f_grad = theano.function([], grad)
	print ("--- %s seconds ---" % (time.time() - start_time)), "losses calculated. generating inital image."

	generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W1, IMAGE_W2))))
	print ("--- %s seconds ---" % (time.time() - start_time)), "image initialized. Optimizing."

	x0 = generated_image.get_value().astype('float64')
	# for i in range(8):
	# 	print ("--- %s seconds ---" % (time.time() - start_time)), i
	# 	scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40)
	# 	x0 = generated_image.get_value().astype('float64')
	scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=2, disp=1)
	x0 = generated_image.get_value().astype('float64')
	print ("--- %s seconds ---" % (time.time() - start_time)), "optimised. deprocessing."

	op = copy.deepcopy(deprocess(x0))
	imsave('tmp_result.png',op)
	print ("--- %s seconds ---" % (time.time() - start_time)), "deprocessed. outputs ssved"
	del net, values, photo, art, rawim, layers, input_im_theano, outputs, photo_features, art_features, generated_image, gen_features, losses, grad, f_loss, f_grad, x0
	print '--------------------------------------------------------------------------------------------'

	print ("--- %s seconds ---" % (time.time() - start_time)), "network built. Compiling network."
	net = build_model()
	print ("--- %s seconds ---" % (time.time() - start_time)), "network compiled. loading weights."
	values = pickle.load(open('vgg19_normalized.pkl'))['param values']
	lasagne.layers.set_all_param_values(net['pool5'], values)
	print ("--- %s seconds ---" % (time.time() - start_time)), "weights loaded. preprocessing content and style image."

	photo = imread('in11.png')
	photo_mat = reshape_img(photo)
	# print photo_mat.shape
	# photo_mat = im2double(photo_mat)
	h, w, c = photo_mat.shape
	coo = getlaplacian1(photo_mat/255., np.zeros(shape=(h, w)), 1e-5, 1).tocoo()
	indices = np.mat([coo.row, coo.col]).transpose()
	csr = coo.tocsr()
	matting = theano.sparse.basic.as_sparse(csr,name=None)
	print ("--- %s seconds ---" % (time.time() - start_time)), "matting laplacian created."
	rawim, photo = prep_image(photo)
	# print photo.shape
	art = imread('tar11.png')
	rawim, art = prep_image(art)
	# print art.shape
	print ("--- %s seconds ---" % (time.time() - start_time)), "images and masks preprocessed. extracting content and style features."
	
	layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
	layers = {k: net[k] for k in layers}
	input_im_theano = T.tensor4()
	outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
	photo_features = {k: theano.shared(output.eval({input_im_theano: photo})) for k, output in zip(layers.keys(), outputs)}
	art_features = {k: theano.shared(output.eval({input_im_theano: art})) for k, output in zip(layers.keys(), outputs)}
	print ("--- %s seconds ---" % (time.time() - start_time)), "style and content extracted. generating image."

	seg_net = build_model_segs()
	print ("--- %s seconds ---" % (time.time() - start_time)), "seg network built."
	content_masks, style_masks = load_seg('in11_mask.png', 'tar11_mask.png', (IMAGE_W2,IMAGE_W1), (IMAGE_W2,IMAGE_W1))
	print ("--- %s seconds ---" % (time.time() - start_time)), "segmented images prepared."
	layers_seg = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
	layers_seg = {k: seg_net[k] for k in layers_seg}
	input_im_theano_seg = T.tensor4()
	outputs_seg = lasagne.layers.get_output(layers_seg.values(), input_im_theano_seg)
	content_seg_features = []
	for content_mask in content_masks:
		photo_seg_features = {k: theano.shared(output.eval({input_im_theano_seg: content_mask})) for k, output in zip(layers_seg.keys(), outputs_seg)}
		content_seg_features.append(photo_seg_features)
	style_seg_features = []
	for style_mask in style_masks:
		art_seg_features = {k: theano.shared(output.eval({input_im_theano_seg: style_mask})) for k, output in zip(layers_seg.keys(), outputs_seg)}
		style_seg_features.append(art_seg_features)
	print ("--- %s seconds ---" % (time.time() - start_time)), "segmented features graphed."

	generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W1, IMAGE_W2))))
	gen_features = lasagne.layers.get_output(layers.values(), generated_image)
	gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}
	print ("--- %s seconds ---" % (time.time() - start_time)), "image generated. calculating losses."

	losses = []
	losses.append(0.0065 * content_loss(photo_features, gen_features, 'conv4_2'))
	losses.append(0.2e6 * style_loss(art_features, gen_features, style_seg_features, content_seg_features, 'conv1_1'))
	losses.append(0.2e6 * style_loss(art_features, gen_features, style_seg_features, content_seg_features, 'conv2_1'))
	losses.append(0.2e6 * style_loss(art_features, gen_features, style_seg_features, content_seg_features, 'conv3_1'))
	losses.append(0.2e6 * style_loss(art_features, gen_features, style_seg_features, content_seg_features, 'conv4_1'))
	losses.append(0.2e6 * style_loss(art_features, gen_features, style_seg_features, content_seg_features, 'conv5_1'))
	losses.append(0.1e-7 * total_variation_loss(generated_image))
	losses.append(0.135 * matting_loss(generated_image, matting))
	total_loss = sum(losses)

	grad = T.grad(total_loss, generated_image)
	f_loss = theano.function([], total_loss)
	f_grad = theano.function([], grad)
	print ("--- %s seconds ---" % (time.time() - start_time)), "losses calculated. generating inital image."

	rawim, init_image = prep_image(imread('tmp_result.png'))#rawim, init_image = prep_image(op)
	generated_image.set_value(floatX(init_image))
	print ("--- %s seconds ---" % (time.time() - start_time)), "image initialized. Optimizing."

	x0 = generated_image.get_value().astype('float64')
	# for i in range(8):
	# 	print ("--- %s seconds ---" % (time.time() - start_time)), i
	# 	scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40)
	# 	x0 = generated_image.get_value().astype('float64')
	scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=2, disp=1)
	x0 = generated_image.get_value().astype('float64')
	print ("--- %s seconds ---" % (time.time() - start_time)), "optimised. deprocessing."

	np.save('x0.npy', x0)
	imsave('best_image_rgb.png',deprocess(x0))

	x0 = np.load('x0.npy')
	x0 = x0[0]
	x0 += MEAN_VALUES
	content_input = imread('in11.png')
	rawim, content_input = prep_image_affine(content_input)
	input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.
	_, H, W = np.shape(input_)
	output_ = np.ascontiguousarray(x0, dtype=np.float32) / 255.
	from smooth_local_affine import smooth_local_affine
	best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, 15, 1e-1).transpose(1, 2, 0)
	print ("--- %s seconds ---" % (time.time() - start_time)), "smooth local affine calculated."
	result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
	result.save('final_output.png')
	print ("--- %s seconds ---" % (time.time() - start_time)), "final result generated"