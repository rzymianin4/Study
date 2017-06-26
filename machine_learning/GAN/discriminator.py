#!/usr/bin/env python3

# Dawid Tracz 
 # oświadzcam iż poniższy kod został napisany samodzielnie.
 # 

if True:	#imports
	import os, sys
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	import tensorflow as tf
	import numpy as np
	from tqdm import tqdm, trange
	from math import ceil

#-------------------------------------------------------------------------------------------------

def LeakyReLU(x, a=0.01):
	return tf.maximum(x, a*x)

class ResCell(object):
	def __init__(self, nol, n_cat, img_shp, f_shp, in_sc=0.05, keep_prob=1, epsilon=1e-8,
	stride=1, residual=False, activation=LeakyReLU):
		if stride>1 and (nol>1 or residual):
			raise ValueError("bad set of arguments")
		if img_shp[-1]!=f_shp[-1] and residual:
			raise ValueError("bad set of arguments")
		self.nol = nol
		self.n_cat = n_cat
		self.keep_prob = keep_prob
		self.epsilon = epsilon
		self.stride = stride
		self.residual = residual
		self.activation = activation
		self.img_shp = img_shp[:]
		f_shp = f_shp[:2] + [img_shp[-1]+n_cat] + f_shp[2:]

		self.alpha = [None]*nol
		self.beta = [None]*nol
		self.weights = [None]*nol
		for i in range(nol):
			self.alpha[i] = tf.Variable(tf.random_uniform(img_shp[-1:], 1-in_sc/100, 1+in_sc/100))
			self.beta[i] =  tf.Variable(tf.random_uniform(img_shp[-1:], -in_sc/100, in_sc/100))
			self.weights[i] = tf.Variable(tf.random_uniform(f_shp, -in_sc, in_sc))
			print(img_shp, "+", img_shp[:-1]+[n_cat], " ::", f_shp, "(s", stride, ")", sep='', file=sys.stderr)	
			img_shp[-1] = f_shp[-1]
			f_shp[-2] = img_shp[-1] + n_cat
			if stride>1:
				img_shp[:2] = [ceil(n/stride) for n in img_shp[:2]]

	def app_lbls(self, x, lbls):
		shp = x.get_shape().as_list()[1:3] + lbls.get_shape().as_list()[-1:]
		lbls = tf.tile(lbls, shp[:2])
		shp = shp[:1] + [-1] + shp[1:]
		lbls = tf.reshape(lbls, shp)
		lbls = tf.transpose(lbls, (1,0,2,3))
		x = tf.concat([x, lbls], axis=-1)
		return x

	def normalize(self, x, alpha, beta):
		mean, var = tf.nn.moments(x,[0,1,2])
		x = tf.nn.batch_normalization(x, mean, var, beta, alpha, self.epsilon)
		return x

	def __call__(self, x, y, train=False):
		if self.residual:
			start_state = x
		for a, b, f in zip(self.alpha, self.beta, self.weights):
			x = self.normalize(x, a, b)
			if train and self.keep_prob < 1:
				tf.nn.dropout(x, keep_prob = self.keep_prob)
			x = self.activation(x)
			x = self.app_lbls(x, y)
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
	def __init__(self, n_cat, in_size, out_size, in_sc=0.05, keep_prob=1, epsilon=1e-8, activation=tf.nn.relu):
		self.keep_prob = keep_prob
		self.epsilon = epsilon
		self.activation = activation

		self.alpha = tf.Variable(tf.random_uniform([in_size], 1-in_sc, 1+in_sc))
		self.beta =  tf.Variable(tf.random_uniform([in_size], -in_sc, in_sc))
		self.weights =  tf.Variable(tf.random_uniform([in_size+n_cat, out_size], -in_sc, in_sc))
		print([None, in_size], "+", [None, n_cat], " ::", [in_size+n_cat, out_size], sep='', file=sys.stderr)

	def normalize(self, x, alpha, beta):
		mean, var = tf.nn.moments(x,[0])
		x = tf.nn.batch_normalization(x, mean, var, beta, alpha, self.epsilon)
		return x

	def __call__(self, x, y, train):
		x = self.normalize(x, self.alpha, self.beta)
		if train and self.keep_prob < 1:
			tf.nn.dropout(x, keep_prob = self.keep_prob)
		x = self.activation(x)
		x = tf.concat([x, y], axis=-1)
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
				l_list.append( ResCell(nol, self.n_cat, img_shp, f_shp, **kwargs) )
		return img_shp

	def __init__(self, n_cat, sizes, learning_rate=0.001, in_sc=0.05, dropout=0, epsilon=1e-8):
		self.n_cat = n_cat
		self.in_sc = in_sc
		self.learning_rate = learning_rate
		self.keep_prob = 1 - dropout
		self.epsilon = epsilon
		self.img_shp = list(sizes)
		self.activation = tf.nn.relu

		img_shp = self.img_shp[:]
		
		self.conv_layer_params = [ ("conv", 1, [3,3,32], {'in_sc': self.in_sc}),
								   #("conv", 1, [3,3,32], {'in_sc': self.in_sc}),
								   #("conv", 2, [3,3,32], {'in_sc': self.in_sc}),
								   #("conv", 2, [3,3,32], {'in_sc': self.in_sc}),
								   #("conv", 2, [3,3,32], {'in_sc': self.in_sc}),
								   ("conv", 1, [3,3,64], {'in_sc': self.in_sc, 'stride': 2}),
								   #("pool", tf.nn.max_pool, {}),
								   #("conv", 2, [3,3,64], {'in_sc': self.in_sc}),
								   #("conv", 2, [3,3,64], {'in_sc': self.in_sc}),
								   ("conv", 1, [3,3,128], {'in_sc': self.in_sc, 'stride': 2}),
								   #("pool", tf.nn.max_pool, {}),
								   #("conv", 2, [3,3,128], {'in_sc': self.in_sc}),
								   #("conv", 2, [3,3,128], {'in_sc': self.in_sc}),
								   ("conv", 1, [3,3,256], {'in_sc': self.in_sc, 'stride': 2}) ]
		self.convLayers = []
		self.layerFactory(self.convLayers, self.conv_layer_params, img_shp)

		self.fullyLayers = []
		self.fully_layer_params = [np.prod(img_shp)] + [1024] + [1]	
		inp = self.fully_layer_params[0]
		for out in self.fully_layer_params[1:]:
			self.fullyLayers.append( FullyCell(self.n_cat, inp, out, activation=tf.nn.tanh,
														 			 keep_prob=self.keep_prob) )
			inp = out

		self.trainables = self.get_trainables()

	def get_trainables(self):
		trainables = []
		for layer in self.convLayers + self.fullyLayers:
			trainables.append(layer.alpha)
			trainables.append(layer.beta)
			trainables.append(layer.weights)
		return trainables

	def __call__(self, x, y, train=False):
		print("D::CALL", file=sys.stderr)
		for cell in self.convLayers:
			x = cell(x, y, train)
		x = tf.reshape(x, [-1, np.prod(np.array(x.get_shape().as_list()[-3:]))])
		for cell in self.fullyLayers:
			x = cell(x, y, train)
		return tf.nn.sigmoid(x)
