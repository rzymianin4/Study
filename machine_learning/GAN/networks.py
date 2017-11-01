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

G_INP_SHP  = [100]
D_INP_SHP  = [32,32,3]
INIT_SCALE = 0.05
G_DROPOUT  = 0.4
D_DROPOUT  = 0.0
N_CAT      = 10
#GPU = int(sys.argv[1])

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

std_bn_params = {'dims': [0,1,2], 'epsilon': 1e-8, 'in_sc': 0.0005}

G_DEFARGS = { 'keep_prob': 1.0 - G_DROPOUT,
			  'activ_fun': tf.nn.relu,
			  'stride': 1,
			  'padding': "SAME",
			  'in_sc': INIT_SCALE,
			  'bias': False,
			  'batch_norm': False,
			  'n_cat': N_CAT,
			}	

G_ARCHITECTURE = [
				  (LinearLayer, [1024],    {'batch_norm': False }),
				  (LinearLayer, [4*4*128], {'batch_norm': {'dims': [0], 'epsilon': 1e-8, 'in_sc': 0.0005} }),
				  (Reshape,     [4,4,128], { }),
				  (DeconvLayer, [3,3,128], {'stride': 2, 'outXY': [ 8, 8], 'batch_norm': False }),
				  (DeconvLayer, [3,3, 64], {'stride': 2, 'outXY': [16,16], 'batch_norm': False }),
				  (DeconvLayer, [3,3, 32], {'stride': 2, 'outXY': [32,32], 'batch_norm': std_bn_params }),
				  (DeconvLayer, [3,3,  3], {'outXY': [32,32], 'keep_prob': 1.0, 'activ_fun': tf.nn.tanh, }),
				 ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

std_bn_params = {'dims': [0], 'epsilon': 1e-8, 'in_sc': 0.0005}

D_DEFARGS = { 'keep_prob': 1.0 - D_DROPOUT,
			  'activ_fun': leakyReLU,
			  'stride': 1,
			  'padding': "SAME",
			  'in_sc': INIT_SCALE,
			  'bias': False,
			  'batch_norm': False,
			  'n_cat': N_CAT,
			}	

D_ARCHITECTURE = [
				  (BatchNormalization, [0,1,2],{'epsilon': 1e-8, 'in_sc': 0.0005}),
				  (ConvLayer,   [3,3, 32], {'batch_norm': std_bn_params }),
				  [
				   (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				   (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				  ],
				  [
				   (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				   (ConvLayer,   [3,3,32], {'batch_norm': std_bn_params }),
				  ],
				  (ConvLayer,   [3,3, 64], {'stride': 2, 'batch_norm': std_bn_params }),
				  [
				   (ConvLayer,   [3,3,64], {'batch_norm': std_bn_params }),
				   (ConvLayer,   [3,3,64], {'batch_norm': std_bn_params }),
				  ],
				  (ConvLayer,   [3,3,128], {'stride': 2, 'batch_norm': std_bn_params }),
				  [
				   (ConvLayer,   [3,3,128], {'batch_norm': std_bn_params }),
				   (ConvLayer,   [3,3,128], {'batch_norm': std_bn_params }),
				  ],
				  (ConvLayer,   [3,3,256], {'stride': 2, 'batch_norm': std_bn_params }),
				  (Reshape,     [4096], { }),
				  (LinearLayer, [1024], {'batch_norm': std_bn_params }),
				  (LinearLayer, [1],   {'bias': [1], 'activ_fun': lambda x: x }),
				 ]

#------------------------------------------------------------------------------------------------
