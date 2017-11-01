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

def sighandler(sig, frame):
	print("\b\bInterrupted", file=sys.stderr)
	exit(-1)
signal.signal(signal.SIGINT, sighandler)

#------------------------------------------------------------------------------------------------

DATA_SHAPE = [24, 24, 3]
N_CAT = 10
DROPOUT = 0
INIT_SCALE = 0.05
LEARNING_RATE = 0.001
BATCH_SIZE = 128
GPU = int(sys.argv[1])

#------------------------------------------------------------------------------------------------

class NeuralNetwork(object):

	def _layerFactory(self, data_shp, architecture):
		layers = [None]*len(architecture)
		for i, specif in enumerate(architecture):
			if type(specif) is tuple:
				print(">>> ", data_shp)
				LayerClass = specif[0]
				layer_shp = specif[1]
				kwargs = {**self._defargs, **specif[-1]}
				layers[i] = LayerClass(data_shp, layer_shp, **kwargs)
			elif type(specif) is list:
				resLayers = self._layerFactory(data_shp, specif)
				layers[i] = ResCell(resLayers)
		return layers

	def __init__(self, data_shp, architecture, defargs):
		self._defargs = defargs
		self._layers = self._layerFactory(data_shp, architecture)	

	def __call__(self, x, trainmode):
		for layer in self._layers:
			x = layer(x, 0, trainmode)
		return x

	def getTrainables(self):
		trainables = []
		for layer in self._layers:
			trainables += layer.getTrainables()
		return trainables

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

std_bn_params = {'dims': [0], 'epsilon': 1e-8, 'in_sc': 0.0005}

DEFARGS = { 'keep_prob': 1.0 - DROPOUT,
			'activ_fun': tf.nn.relu,
			'stride': 1,
			'padding': "SAME",
			'in_sc': INIT_SCALE,
			'bias': False,
			'batch_norm': False,
			'n_cat': False, # don't append labels
		  }	

ARCHITECTURE = [
				(ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				[
				 (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				 (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				],
				[
				 (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				 (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				],
				[
				 (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				 (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				],
				[
				 (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				 (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				],
				(ConvLayer,   [3,3,64], {'batch_norm': std_bn_params }),
				(Pooling, tf.nn.max_pool, {'stride': 2, 'ksize': 2 }),
				[
				 (ConvLayer,   [3,3,64], {'batch_norm': std_bn_params }),
				 (ConvLayer,   [3,3,64], {'batch_norm': std_bn_params }),
				],
				[
				 (ConvLayer,   [3,3,64], {'batch_norm': std_bn_params }),
				 (ConvLayer,   [3,3,64], {'batch_norm': std_bn_params }),
				],
				(ConvLayer,   [3,3,128], {'batch_norm': std_bn_params }),
				(Pooling, tf.nn.max_pool, {'stride': 2, 'ksize': 2 }),
				[
				 (ConvLayer,   [3,3,128], {'batch_norm': std_bn_params }),
				 (ConvLayer,   [3,3,128], {'batch_norm': std_bn_params }),
				],
				[
				 (ConvLayer,   [3,3,128], {'batch_norm': std_bn_params }),
				 (ConvLayer,   [3,3,128], {'batch_norm': std_bn_params }),
				],
				#(ConvLayer,   [3,3,256], {'batch_norm': std_bn_params }),
				#(Pooling, tf.nn.max_pool, {'stride': 2, 'ksize': 2 }),
				(Reshape,     [4608], { }),
				(LinearLayer, [1024], {'batch_norm': std_bn_params }),
				(LinearLayer, [10],   {'bias': [1], 'activ_fun': lambda x: x }),
			   ]

#------------------------------------------------------------------------------------------------

class Classifier(object):
	def __init__(self, data_shp, n_cat, architecture, defargs, GPU):
		self.config = tf.ConfigProto( device_count = {'GPU': GPU} )
		self.sess = tf.InteractiveSession(config=self.config)

		self.imgs = tf.placeholder(tf.float32, [None]+data_shp)
		self.lbls = tf.placeholder(tf.float32, [None, n_cat])

		self.network = NeuralNetwork(data_shp[:], architecture, defargs)
		print("len ", len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
		#exit(-1)

		self.t_ans = self.network(self.imgs, trainmode=True)
		self.t_acc = tf.equal(tf.argmax(self.t_ans, 1), tf.argmax(self.lbls, 1))
		self.t_acc = tf.reduce_mean(tf.cast(self.t_acc, tf.float32))
		
		self.ans = self.network(self.imgs, trainmode=False)
		self.acc = tf.equal(tf.argmax(self.ans, 1), tf.argmax(self.lbls, 1))
		self.acc = tf.reduce_mean(tf.cast(self.acc, tf.float32))

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.t_ans, labels=self.lbls))
		self.gradloss = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss, var_list=self.network.getTrainables())
		print(len(self.network.getTrainables()))

		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)
		self.savior = tf.train.Saver()
		self.sess.graph.finalize()

	def epoch(self, dataSet):
		images, labels = dataSet.get_batches(BATCH_SIZE, sizes=DATA_SHAPE)
		res = np.float32(0.0)
		for x, y in tqdm(zip(images, labels), total=len(images)):
			gl, acc = self.sess.run((self.gradloss, self.t_acc),
									feed_dict={self.imgs: x, self.lbls: y})
			res += acc
		return res/np.float32(len(labels))
	
	def test(self, dataSet):
		images, labels = dataSet.get_batches(BATCH_SIZE, mode="test", sizes=DATA_SHAPE)
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
			ans = np.float32(0.0)
			for i in [-4, 0, 4]:
				for j in [-4, 0, 4]:
					x = dataSet.cut(ims, sizes=(24,24), pos=(i,j))
					ans = self.sess.run(self.ans, feed_dict={self.imgs: x}) + ans #(1 + (i==0 and j==0))*
			ans = np.argmax(ans, axis=-1)
			y = np.argmax(y, axis=-1)
			ans = (ans==y).mean()
			res += ans
		return res/np.float32(len(labels))


#------------------------------------------------------------------------------------------------

dataSet = CifarLoader("../cifar-10-batches-py/")

#dataSet.train.images = np.load("gen_images.npy")
#dataSet.train.labels = np.load("gen_labels.npy")

classifier = Classifier(DATA_SHAPE, N_CAT, ARCHITECTURE, DEFARGS, GPU=1)
print("__________________", file=sys.stderr)

res = classifier.test(dataSet)
i = 0
print(res)
print(res, file=sys.stderr)
while i<100:
	i += 1
	train_res = classifier.epoch(dataSet)
	test_res = classifier.test(dataSet)
	print("EPOCH: ", i, ", TRAIN RES: ", train_res, ", TEST RES: ", test_res, sep='', end='')
	if test_res > 0.85:
		dem_test_res = classifier.dem_test(dataSet)
		print(", VOTE TEST RES: ", dem_test_res, sep='', end='')
	print()
	sys.stdout.flush()
