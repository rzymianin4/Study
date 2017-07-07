#!/usr/bin/env python3

if True:	#imports
	import matplotlib
	matplotlib.use('Agg')
	import signal, os, sys
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	#print("__________________________")
	import tensorflow as tf
	#from tensorflow.examples.tutorials.mnist import input_data
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy.misc import toimage
	from skimage import transform
	import pickle
	from tqdm import tqdm, trange
	from loader import CifarLoader, MnistLoader
	from math import ceil

def sighandler(sig, frame):
	print("\b\bInterrupted", file=sys.stderr)
	exit(-1)
signal.signal(signal.SIGINT, sighandler)

#-----------------------------------------------------------------------------
BATCH_SIZE = 128
IMG_SIZES = [24,24,3]
#IMG_SIZES = [28,28,1]
INIT_SCALE = 0.05
DROPOUT = 0.3
#-----------------------------------------------------------------------------

class ResCell(object):
	def __init__(self, nol, img_shp, f_shp, in_sc=0.05, keep_prob=1, epsilon=1e-8,
	stride=1, residual=False, activation=tf.nn.relu):
		if stride>1 and (nol>1 or residual):
			raise ValueError("bad set of arguments")
		if img_shp[-1]!=f_shp[-1] and residual:
			raise ValueError("bad set of arguments")
		self.nol = nol
		self.keep_prob = keep_prob
		self.epsilon = epsilon
		self.stride = stride
		self.residual = residual
		self.activation = activation
		f_shp = f_shp[:2] + img_shp[-1:] + f_shp[2:]

		self.alphas = [None]*nol
		self.betas = [None]*nol
		self.filters = [None]*nol
		for i in range(nol):
			self.alphas[i] = tf.Variable(tf.random_uniform(img_shp[-1:], 1-in_sc/100, 1+in_sc/100))
			self.betas[i] =  tf.Variable(tf.random_uniform(img_shp[-1:], -in_sc/100, in_sc/100))
			self.filters[i] = tf.Variable(tf.random_uniform(f_shp, -in_sc, in_sc))		
			img_shp[-1] = f_shp[-1]
			f_shp[-2] = img_shp[-1]
			if stride>1:
				img_shp[:2] = [ceil(n/stride) for n in img_shp[:2]]

	def normalize(self, x, alpha, beta):
		mean, var = tf.nn.moments(x,[0,1,2])
		x = tf.nn.batch_normalization(x, mean, var, beta, alpha, self.epsilon)
		return x

	def __call__(self, x, train=False):
		if self.residual:
			start_state = x
		for a, b, f in zip(self.alphas, self.betas, self.filters):
		#for f in self.filters:
			x = self.normalize(x, a, b)
			if train and self.keep_prob < 1:
				tf.nn.dropout(x, keep_prob = self.keep_prob) / self.keep_prob
			x = self.activation(x)
			print(x.get_shape().as_list(), ":", f.get_shape().as_list(), file=sys.stderr)
			x = tf.nn.conv2d(x, f, strides=[1,self.stride,self.stride,1], padding="SAME")
		if self.residual:
			x = (x + start_state)/2
		return x		


class PoolCell(object):
	def __init__(self, pool_fun, stride=2, ks=2, padding="SAME"):
		self.pool_fun = pool_fun
		self.stride = stride
		self.ks = ks
		self.padding = padding

	def __call__(self, x, train=False):
		return self.pool_fun(x, ksize=[1,self.ks,self.ks,1],
								strides=[1,self.stride,self.stride,1],
								padding=self.padding)


class FullyCell(object):
	def __init__(self, in_size, out_size, in_sc=0.05, keep_prob=1, epsilon=1e-8,
	activation=tf.nn.relu):
		self.keep_prob = keep_prob
		self.epsilon = epsilon
		self.activation = activation

		self.alpha = tf.Variable(tf.random_uniform([in_size], 1-in_sc, 1+in_sc))
		self.beta =  tf.Variable(tf.random_uniform([in_size], -in_sc, in_sc))
		self.weights =  tf.Variable(tf.random_uniform([in_size, out_size], -in_sc, in_sc)) 

	def normalize(self, x, alpha, beta):
		mean, var = tf.nn.moments(x,[0])
		x = tf.nn.batch_normalization(x, mean, var, beta, alpha, self.epsilon)
		return x

	def __call__(self, x, train):
		x = self.normalize(x, self.alpha, self.beta)
		if train and self.keep_prob < 1:
			tf.nn.dropout(x, keep_prob = self.keep_prob) / self.keep_prob
		x = self.activation(x)
		x = x @ self.weights
		return x

class Discriminator(object):
	def layerFactory(self, l_list, params, img_shp):
		for data in params:
			if data[0] == "pool":
				pool_fun, kwargs = data[1:]
				l_list.append( PoolCell(pool_fun, **kwargs) )
				s = 2 if 'stride' not in kwargs else kwargs['stride']
				img_shp[:2] = [ceil(n/s) for n in img_shp[:2]]
			if data[0] == "conv":
				nol, f_shp, kwargs = data[1:]
				if 'residual' not in kwargs:
					kwargs = dict(kwargs, residual=(img_shp[-1]==f_shp[-1] and 'stride' not in kwargs))
				l_list.append( ResCell(nol, img_shp, f_shp, **kwargs) )
		return img_shp

	def __init__(self, n_cat, sizes, learning_rate=0.001, in_sc=0.05,
	dropout=0, epsilon=1e-8, GPU=0):
		self.GPU = GPU
		self.n_cat = n_cat
		self.in_sc = in_sc
		self.learning_rate = learning_rate
		self.keep_prob = 1 - dropout
		self.epsilon = epsilon
		self.sizes = sizes
		self.activation = tf.nn.relu 

		self.config = tf.ConfigProto( device_count = {'GPU': GPU} )
		self.sess = tf.InteractiveSession(config=self.config)

		self.imgs = tf.placeholder(tf.float32, [None]+list(sizes))
		self.lbls = tf.placeholder(tf.float32, [None, self.n_cat])

		# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		img_shp = list(sizes)
		
		self.conv_layer_params = [ ("conv", 1, [3,3,32], {'in_sc': self.in_sc}),
								   ("conv", 2, [3,3,32], {'in_sc': self.in_sc}),
								   ("conv", 2, [3,3,32], {'in_sc': self.in_sc}),
								   ("conv", 2, [3,3,32], {'in_sc': self.in_sc}),
								   ("conv", 2, [3,3,32], {'in_sc': self.in_sc}),
								   ("conv", 1, [3,3,64], {'in_sc': self.in_sc}), #'stride': 2}),
								   ("pool", tf.nn.max_pool, {}),
								   ("conv", 2, [3,3,64], {'in_sc': self.in_sc}),
								   ("conv", 2, [3,3,64], {'in_sc': self.in_sc}),
								   ("conv", 1, [3,3,128], {'in_sc': self.in_sc}), #'stride': 2}),
								   ("pool", tf.nn.max_pool, {}),
								   ("conv", 2, [3,3,128], {'in_sc': self.in_sc}),
								   ("conv", 2, [3,3,128], {'in_sc': self.in_sc}), ]
		self.convLayers = []
		self.layerFactory(self.convLayers, self.conv_layer_params, img_shp)
		self.fullyLayers = []
		self.fullyLayers.append( FullyCell(np.prod(img_shp), 1024, activation=tf.nn.tanh,
																   keep_prob=self.keep_prob) )
		self.fullyLayers.append( FullyCell(1024, self.n_cat, activation=tf.nn.tanh,
															 keep_prob=self.keep_prob) )

		# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

		self.t_ans = self.network(self.imgs, train=True)
		self.t_acc = tf.equal(tf.argmax(self.t_ans, 1), tf.argmax(self.lbls, 1))
		self.t_acc = tf.reduce_mean(tf.cast(self.t_acc, tf.float32))
		
		#self.ans= self.t_ans
		self.ans = self.network(self.imgs, train=False)
		self.acc = tf.equal(tf.argmax(self.ans, 1), tf.argmax(self.lbls, 1))
		self.acc = tf.reduce_mean(tf.cast(self.acc, tf.float32))

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.t_ans, labels=self.lbls))
		self.gradloss = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)
		self.savior = tf.train.Saver()
		self.sess.graph.finalize()

	#-----------------------------------------------------------------------------

	def normalize(self, batch, alpha, beta):
		mean, var = tf.nn.moments(batch,[0])
		return tf.nn.batch_normalization(batch, mean, var, alpha, beta, self.epsilon)

	def network(self, x, train=False):
		#x = self.normalize(x, self.in_alpha, self.in_beta)
		for cell in self.convLayers:
			x = cell(x, train)
		x = tf.reshape(x, [-1, np.prod(np.array(x.get_shape().as_list()[-3:]))])
		for cell in self.fullyLayers:
			x = cell(x, train)
		return x

	#-----------------------------------------------------------------------------

	def epoch(self, dataSet):
		images, labels = dataSet.get_batches(BATCH_SIZE, sizes=IMG_SIZES)
		res = np.float32(0.0)
		for x, y in tqdm(zip(images, labels), total=len(images)):
			gl, acc = self.sess.run((self.gradloss, self.t_acc),
									feed_dict={self.imgs: x, self.lbls: y})
			res += acc
		return res/np.float32(len(labels))
	
	def test(self, dataSet):
		images, labels = dataSet.get_batches(BATCH_SIZE, mode="test", sizes=IMG_SIZES)
		res = np.float32(0.0)
		for x, y in zip(images, labels):
			acc = self.sess.run((self.acc),
								feed_dict={self.imgs: x, self.lbls: y})
			res += acc
		return res/np.float32(len(labels))
	
	def dem_test(self, dataSet):
		images, labels = dataSet.get_batches(BATCH_SIZE, mode="test")
		res = np.float32(0.0)
		for ims, y in tqdm(zip(images, labels), total=len(labels), desc="test"):
			ans = 0
			for i in [-4, 0, 4]:
				for j in [-4, 0, 4]:
					x = dataSet.cut(ims, sizes=(24,24), pos=(i,j))
					ans = self.sess.run(self.ans, feed_dict={self.imgs: x}) + ans
			ans = np.argmax(ans, axis=-1)
			y = np.argmax(y, axis=-1)
			ans = (ans==y).mean()
			res += ans
		return res/np.float32(len(labels))


#-----------------------------------------------------------------------------

dataSet = CifarLoader("../cifar-10-batches-py/")
#dataSet = MnistLoader("../MNIST_data/")

dataSet.train.images = np.load("gen_images.npy")
dataSet.train.labels = np.load("gen_labels.npy")

nn = Discriminator(10, sizes=IMG_SIZES, in_sc=INIT_SCALE, dropout=DROPOUT, GPU=1)
print("__________________", file=sys.stderr)

res = nn.test(dataSet)
print(res)
print(res, file=sys.stderr)

while True:
	train_res = nn.epoch(dataSet)
	test_res = nn.test(dataSet)
	dem_test_res = nn.dem_test(dataSet)
	print(train_res, test_res, dem_test_res)
	print(train_res, test_res, dem_test_res, file=sys.stderr)
	sys.stdout.flush()
	
