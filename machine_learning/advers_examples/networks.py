#!/usr/bin/env python3

if True:	#imports
	import os, sys, signal
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	import tensorflow as tf
	import numpy as np
	from numpy import prod
	from tqdm import tqdm, trange
	from math import ceil
	from layers import *
	from loader import *

#------------------------------------------------------------------------------------------------

class NeuralNetwork(object):

	def _layerFactory(self, data_shp, architecture):
		layers = [None]*len(architecture)
		for i, specif in enumerate(architecture):
			if type(specif) is tuple:
				print(">>> ", data_shp, end='\n')
				LayerClass = specif[0]
				layer_shp = specif[1]
				kwargs = {**self._defargs, **specif[-1]}
				layers[i] = LayerClass(data_shp, layer_shp, **kwargs)
			elif type(specif) is list:
				resLayers = self._layerFactory(data_shp, specif)
				layers[i] = ResCell(resLayers)
		print(">>> ", data_shp)
		return layers

	def __init__(self, data_shp, architecture, defargs):
		self._defargs = defargs
		self._layers = self._layerFactory(data_shp, architecture)	

	def __call__(self, x, y, trainmode):
		for layer in self._layers:
			x = layer(x, y, trainmode)
		return x

	def getTrainables(self):
		trainables = []
		for layer in self._layers:
			trainables += layer.getTrainables()
		return trainables

#------------------------------------------------------------------------------------------------

CIFAR_INP_SHP = [32,32,3]
MNIST_INP_SHP = [28,28,1]
N_CAT = 10
INIT_SCALE = 0.05
DROPOUT = 0.0
BATCH_SIZE = 128
CLASS_LEARNING_RATE = 0.001

#------------------------------------------------------------------------------------------------

std_bn_params  = {'class': BatchNormalization, 'dims': [0], 'epsilon': 1e-8, 'in_sc': 0.0005}
std_vbn_params = {'class': VirtualBatchNormalization, 'dims': -1, 'epsilon': 1e-8, 'in_sc': 0.0005}
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

DEFARGS = { 'keep_prob': 1.0 - DROPOUT,
			'activ_fun': tf.nn.relu,
			'stride': 1,
			'padding': "SAME",
			'in_sc': INIT_SCALE,
			'bias': False,
			'batch_norm': False,
			'n_cat': 0, # none to append to layers
		  }	

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

CIFAR_ARCHITECTURE = [
					  (ConvLayer,   [3,3,32], { }),
					  [
					   (ConvLayer,   [3,3,32], { }),
					   (ConvLayer,   [3,3,32], { }),
					  ],
					  [
					   (ConvLayer,   [3,3,32], { }),
					   (ConvLayer,   [3,3,32], { }),
					  ],
					  [
					   (ConvLayer,   [3,3,32], { }),
					   (ConvLayer,   [3,3,32], { }),
					  ],
					  [
					   (ConvLayer,   [3,3,32], { }),
					   (ConvLayer,   [3,3,32], { }),
					  ],
					  (ConvLayer,   [3,3,64], { }),
					  (Pooling, tf.nn.max_pool, {'stride': 2, 'ksize': 2 }),
					  [
					   (ConvLayer,   [3,3,64], { }),
					   (ConvLayer,   [3,3,64], { }),
					  ],
					  [
					   (ConvLayer,   [3,3,64], { }),
					   (ConvLayer,   [3,3,64], { }),
					  ],
					  (ConvLayer,   [3,3,128], { }),
					  (Pooling, tf.nn.max_pool, {'stride': 2, 'ksize': 2 }),
					  [
					   (ConvLayer,   [3,3,128], { }),
					   (ConvLayer,   [3,3,128], { }),
					  ],
					  [
					   (ConvLayer,   [3,3,128], { }),
					   (ConvLayer,   [3,3,128], { }),
					  ],
				#	  (ConvLayer,   [3,3,256], { }),
					  (Pooling, tf.nn.max_pool, {'stride': 2, 'ksize': 2 }),
					  (Reshape,     [2048], { }),
					  (LinearLayer, [1024], { }),
					  (LinearLayer, [10],   {'bias': [1], 'activ_fun': lambda x: x }),
				     ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

MNIST_ARCHITECTURE = [
					  (ConvLayer,   [3,3,32], { }),
					  (ConvLayer,   [3,3,64], { }),
					  (Pooling, tf.nn.max_pool, {'stride': 2, 'ksize': 2 }),
					  (ConvLayer,   [3,3,128], { }),
					  (Pooling, tf.nn.max_pool, {'stride': 2, 'ksize': 2 }),
				#	  (ConvLayer,   [3,3,256], { }),
					  (Pooling, tf.nn.max_pool, {'stride': 2, 'ksize': 2 }),
					  (Reshape,     [2048], { }),
					  (LinearLayer, [1024], { }),
					  (LinearLayer, [10],   {'bias': [1], 'activ_fun': lambda x: x }),
				     ]

#------------------------------------------------------------------------------------------------

class Classifier(object):
	def __init__(self, sess, data_shp, n_cat, architecture, defargs):
		self.sess = sess

		self.imgs = tf.placeholder(tf.float32, [None]+data_shp)
		self.lbls = tf.placeholder(tf.float32, [None, n_cat])

		self.model = NeuralNetwork(data_shp[:], architecture, defargs)
		# print("len", len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

		self.t_ans = self.model(self.imgs, y=None, trainmode=True)
		
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.t_ans, labels=self.lbls))
		self.gradloss = tf.train.AdamOptimizer(CLASS_LEARNING_RATE).minimize(self.loss, var_list=self.model.getTrainables())
		# print(len(self.model.getTrainables()))

		self.t_acc = tf.equal(tf.argmax(self.t_ans, 1), tf.argmax(self.lbls, 1))
		self.t_acc = tf.reduce_mean(tf.cast(self.t_acc, tf.float32))
		
		self.v_ans = self.model(self.imgs, y=None, trainmode=False)
		self.v_acc = tf.equal(tf.argmax(self.v_ans, 1), tf.argmax(self.lbls, 1))
		self.v_acc = tf.reduce_mean(tf.cast(self.v_acc, tf.float32))

		self.savior = tf.train.Saver(self.model.getTrainables(), max_to_keep=1)

	def epoch(self, dataSet, batch_size):
		images, labels = dataSet.get_batches(batch_size)
		res = np.float32(0.0)
		for x, y in tqdm(zip(images, labels), total=len(images)):
			gl, acc = self.sess.run((self.gradloss, self.t_acc), feed_dict={self.imgs: x, self.lbls: y})
			res += acc
		return res/np.float32(len(labels))
	
	def test(self, dataSet, batch_size):
		images, labels = dataSet.get_batches(batch_size, mode="test")
		res = np.float32(0.0)
		for x, y in zip(images, labels):
			acc = self.sess.run((self.v_acc), feed_dict={self.imgs: x, self.lbls: y})
			res += acc
		return res/np.float32(len(labels))

	def save(self, filename):
		self.savior.save(self.sess, filename)

	def restore(self, filename):
		self.savior.restore(self.sess, filename)

#------------------------------------------------------------------------------------------------
