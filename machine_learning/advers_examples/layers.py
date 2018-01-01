if True:	#imports
	import os, sys
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	import tensorflow as tf
	from tensorflow.python.layers.normalization import BatchNormalization as tf_BatchNormalization
	from tensorflow.python.training import moving_averages
	import numpy as np
	from numpy import prod
	from math import ceil

#------------------------------------------------------------------------------------------------

def leakyReLU(x, a=0.15):
	return tf.maximum(x, a*x)

def drawlDims(data_shp, activ_dims):
	data_shp = [None] + data_shp
	cuted = [None]*len(activ_dims)
	for i, dim in enumerate(activ_dims):
		cuted[i] = data_shp[dim]
	return cuted

def cutDims(data_shp, useless_dims):
	data_shp = [None] + data_shp
	useless_dims = sorted(useless_dims, reverse=True)
	for i in useless_dims:
		del data_shp[i]
	return data_shp

#------------------------------------------------------------------------------------------------

class Layer(object):
	"""docstring for NullLayer"""
	def __init__(self, **ignored):
		pass

	def __call__(self, x, y=None, trainmode=False):
		return x

	def getTrainables(self):
		return []

nullLayer = Layer();

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Reshape(object):
	"""docstring for Reshape"""
	def __init__(self, data_shp, out_shp, **ignored):
		if(prod(data_shp) != prod(out_shp)):
			raise ValueError("Invalid reshape:\n\tinput:  " +str(data_shp)+ " (prod=" +str(prod(data_shp))+ ")\n\toutput: " + str(out_shp)+ " (prod=" +str(prod(out_shp))+ ")")
		self._out_shp = data_shp[:] = out_shp

	def __call__(self, x, y=None, trainmode=False):
		return tf.reshape(x, [-1] + self._out_shp)

	def getTrainables(self):
		return []

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class BatchNormalization(object):
	def __init__(self, data_shp, dims, epsilon, in_sc, **ignored):
		if 0 not in dims:
			raise ValueError("batch normalization must be executed over batch dimension (0)")

		self._dims = dims
		self._epsilon = epsilon
		data_shp = cutDims(data_shp, dims)
		self._gamma = tf.Variable(tf.random_uniform(data_shp, 1-in_sc, 1+in_sc))
		self._beta  = tf.Variable(tf.random_uniform(data_shp,  -in_sc,   in_sc))

	def __call__(self, x, y, trainmode):
		mean, var = tf.nn.moments(x, self._dims)
		x = tf.nn.batch_normalization(x, mean, var, self._beta, self._gamma, self._epsilon)
		return x

	def getTrainables(self):
		trainables = [self._gamma, self._beta]
		#print(trainables[0].get_shape().as_list())
		return trainables
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
'''
class VirtualBatchNormalization(object):
	def __init__(self, data_shp, dims, epsilon, in_sc, momentum=0.99, **ignored):
		if 0 not in dims:
			raise ValueError("batch normalization must be executed over batch dimension (0)")

		self._dims = dims
		self._epsilon = epsilon
		self._momentum = momentum
		data_shp = cutDims(data_shp, dims)
		self._gamma = tf.Variable(tf.random_uniform(data_shp, 1-in_sc, 1+in_sc))
		self._beta  = tf.Variable(tf.random_uniform(data_shp,  -in_sc,   in_sc))
		self._var = tf.Variable(tf.random_uniform(data_shp, -in_sc, in_sc), trainable=False)
		self._mean  = tf.Variable(tf.random_uniform(data_shp,  -in_sc,   in_sc), trainable=False)

	def __call__(self, x, y, trainmode):
		print(trainmode)
		if trainmode:
			mean, var = tf.nn.moments(x, self._dims)
			mean = moving_averages.assign_moving_average(self._mean, mean, self._momentum, zero_debias=True)
			var = moving_averages.assign_moving_average(self._var , var , self._momentum, zero_debias=True)
			#print(">>>",mean, var)
		else:
			mean = self._mean
			var = self._var
		x = tf.nn.batch_normalization(x, self._mean, self._var, self._beta, self._gamma, self._epsilon)
		return x

	def getTrainables(self):
		trainables = [self._gamma, self._beta]
		print(trainables[0].get_shape().as_list())
		return trainables
'''
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

class VirtualBatchNormalization(object):
	def __init__(self, data_shp, dims, epsilon, in_sc, momentum=0.99, **ignored):
		#gamma_init = tf.random_uniform(data_shp, 1-in_sc, 1+in_sc)
		#beta_init  = tf.random_uniform(data_shp,  -in_sc,   in_sc)
		self._tf_BN = tf_BatchNormalization(axis=dims, momentum=momentum, epsilon=epsilon)#,
											#beta_initializer=beta_init, gamma_initializer=gamma_init)

	def __call__(self, x, y, trainmode):
		print(trainmode)
		x = self._tf_BN(x, training=trainmode)
		return x

	def getTrainables(self):
		trainables = self._tf_BN.trainable_variables
		#print(trainables[0].get_shape().as_list())
		return trainables
		

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Pooling(object):
	def __init__(self, data_shp, pool_fun, ksize=2, stride=2, padding="SAME", **ignored):
		self._pool_fun = pool_fun
		self._ksize = [1, ksize, ksize, 1]
		self._strides = [1, stride, stride, 1]
		self._padding = padding
		data_shp[:2] = [ceil(i/stride) for i in data_shp[:2]]

	def __call__(self, x, y=None, train=False):
		x = self._pool_fun(x, ksize=self._ksize, strides=self._strides, padding=self._padding)
		return x

	def getTrainables(self):
		return []

#------------------------------------------------------------------------------------------------

class LinearLayer(object):
	"""docstring for LinearLayer"""
	def __init__(self, data_shp, layer_shp, n_cat, bias, batch_norm, keep_prob, activ_fun, in_sc, **ignored):
		data_shp[-1] += n_cat
		self._weights = tf.Variable(tf.random_uniform(data_shp+layer_shp, -in_sc, in_sc))
		data_shp[0] = layer_shp[0]

		self._n_cat = n_cat
		BatchNormalizationClass = batch_norm['class'] if batch_norm else Layer
		self._batch_norm = BatchNormalizationClass(data_shp, **batch_norm) if batch_norm else nullLayer
		self._bias = tf.Variable(tf.random_uniform(drawlDims(data_shp, bias), -in_sc, in_sc)) if bias else 0
		self._keep_prob = keep_prob
		self._activ_fun = activ_fun

	def _append_labels(self, x, y):
		if self._n_cat:
			x = tf.concat([x, y], axis=-1)
		return x

	def __call__(self, x, y, trainmode):
		x = self._append_labels(x, y)
		x = x @ self._weights
		x = self._batch_norm(x, y, trainmode)
		x = x + self._bias
		if trainmode and self._keep_prob < 1:
			tf.nn.dropout(x, keep_prob = self._keep_prob)
		x = self._activ_fun(x)
		return x

	def getTrainables(self):
		trainables = [self._weights]
		if self._bias:
			trainables.append(self._bias)
		trainables += self._batch_norm.getTrainables()
		return trainables

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class ConvLayer(object):
	"""docstring for ConvLayer"""
	def __init__(self, data_shp, filter_shp, stride, padding, n_cat, bias, batch_norm, keep_prob, activ_fun, in_sc, **ignored):
		self._strides = [1, stride, stride, 1]
		self._padding = padding
		data_shp[-1] += n_cat
		filter_shp = filter_shp[:2] + data_shp[-1:] + filter_shp[-1:]
		self._filter = tf.Variable(tf.random_uniform(filter_shp, -in_sc, in_sc))
		data_shp[:] = [ceil(i/stride) for i in data_shp[:2]] + filter_shp[-1:]

		self._n_cat = n_cat
		BatchNormalizationClass = batch_norm['class'] if batch_norm else Layer
		self._batch_norm = BatchNormalizationClass(data_shp, **batch_norm) if batch_norm else nullLayer
		self._bias = tf.Variable(tf.random_uniform(drawlDims(data_shp, bias), -in_sc, in_sc)) if bias else 0
		self._keep_prob = keep_prob
		self._activ_fun = activ_fun

	def _append_labels(self, x, y):
		if self._n_cat:
			shp = x.get_shape().as_list()[1:3] + y.get_shape().as_list()[-1:]
			y = tf.tile(y, shp[:2])
			shp = shp[:1] + [-1] + shp[1:]
			y = tf.reshape(y, shp)
			y = tf.transpose(y, (1,0,2,3))
			x = tf.concat([x, y], axis=-1)
		return x

	def __call__(self, x, y, trainmode):
		x = self._append_labels(x, y)
		x = tf.nn.conv2d(x, self._filter, strides=self._strides, padding=self._padding)
		x = self._batch_norm(x, y, trainmode)
		x = x + self._bias
		if trainmode and self._keep_prob < 1:
			tf.nn.dropout(x, keep_prob = self._keep_prob)
		x = self._activ_fun(x)
		return x

	def getTrainables(self):
		trainables = [self._filter]
		if self._bias:
			trainables.append(self._bias)
		trainables += self._batch_norm.getTrainables()
		return trainables

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class DeconvLayer(object):
	"""docstring for DeconvLayer"""
	def __init__(self, data_shp, filter_shp, outXY, stride, padding, n_cat, bias, batch_norm, keep_prob, activ_fun, in_sc, **ignored):
		if data_shp[:2] != [ceil(n/stride) for n in outXY]:
			raise ValueError("ERR: deconv bad shapes! input: " +str(data_shp[:2])+ ", stride: " +str(stride)+ ", output:" +str(outXY))
		
		self._strides = [1, stride, stride, 1]
		self._padding = padding
		data_shp[-1] += n_cat
		filter_shp = filter_shp + data_shp[-1:]		# deconv: [height, width, output_channels, input_channels]
		self._filter = tf.Variable(tf.random_uniform(filter_shp, -in_sc, in_sc))
		self._data_shp = data_shp[:] = outXY + [filter_shp[-2]]

		self._n_cat = n_cat
		BatchNormalizationClass = batch_norm['class'] if batch_norm else Layer
		self._batch_norm = BatchNormalizationClass(data_shp, **batch_norm) if batch_norm else nullLayer
		self._bias = tf.Variable(tf.random_uniform(drawlDims(data_shp, bias), -in_sc, in_sc)) if bias else 0
		self._keep_prob = keep_prob
		self._activ_fun = activ_fun

	def _append_labels(self, x, y):
		if self._n_cat:
			shp = x.get_shape().as_list()[1:3] + y.get_shape().as_list()[-1:]
			y = tf.tile(y, shp[:2])
			shp = shp[:1] + [-1] + shp[1:]
			y = tf.reshape(y, shp)
			y = tf.transpose(y, (1,0,2,3))
			x = tf.concat([x, y], axis=-1)
		return x

	def __call__(self, x, y, trainmode):
		x = self._append_labels(x, y)
		out_shp = [tf.shape(x)[0]]+self._data_shp
		x = tf.nn.conv2d_transpose(x, self._filter, out_shp, strides=self._strides, padding=self._padding)
		x = tf.reshape(x, [-1]+out_shp[1:])
		x = self._batch_norm(x, y, trainmode)
		x = x + self._bias
		if trainmode and self._keep_prob < 1:
			tf.nn.dropout(x, keep_prob = self._keep_prob)
		x = self._activ_fun(x)
		return x

	def getTrainables(self):
		trainables = [self._filter]
		if self._bias:
			trainables.append(self._bias)
		trainables += self._batch_norm.getTrainables()
		return trainables

#------------------------------------------------------------------------------------------------

class ResCell(object):
	"""docstring for ResCell"""
	def __init__(self, layers):
		self._layers = layers

	def __call__(self, x, y, trainmode):
		_x = x
		for layer in self._layers:
			_x = layer(_x, y, trainmode)
		#x = (_x + x)/2
		return _x + x

	def getTrainables(self):
		trainables = []
		for layer in self._layers:
			trainables += layer.getTrainables()
		return trainables

#------------------------------------------------------------------------------------------------	
