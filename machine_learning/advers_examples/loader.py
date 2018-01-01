#!/usr/bin/env python3

if True:	#imports
	import matplotlib
	#matplotlib.use('TkAgg')
	matplotlib.use('Agg')
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	from tensorflow.examples.tutorials.mnist import input_data
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy.misc import toimage
	from skimage import transform
	import pickle

#-----------------------------------------------------------------------------------------------------

def rand(n, a=1.5):
	r = np.random.randint(round(a*n+1))
	if r > n or not a:
		r = np.random.binomial(n, 0.5)
	return r

class Data(object):
	def __init__(self):
		self.images = None
		self.labels = None

class CifarLoader(object):
	def subload(self, filename):
		filename = self.directory+filename
		FILE = open(filename, "rb")
		data = pickle.load(FILE, encoding="latin1")
		imgs = data["data"]
		imgs = imgs.reshape(10000, 3, 32, 32).astype("float32")/255
		imgs = imgs*2 - 1
		imgs = imgs.transpose(0,2,3,1)
		lbls = np.zeros((len(data["labels"]), 10))
		lbls[np.arange(len(data["labels"])), data["labels"]] = 1
		return np.array(imgs), np.array(lbls)

	def __init__(self, directory):
		self.directory = directory
		self.name = "CIFAR10"
		self.train = Data()
		self.test = Data()
		cti = []
		ctl = []
		for i in range(5):
			imgs, lbls = self.subload("data_batch_%d"%(i+1))
			cti.append(imgs)
			ctl.append(lbls)
		cti = np.array(cti)
		ctl = np.array(ctl)
		self.train.images = cti.reshape((cti.shape[0]*cti.shape[1],) + cti.shape[2:])
		self.train.labels = ctl.reshape((ctl.shape[0]*ctl.shape[1],) + ctl.shape[2:])
		self.test.images, self.test.labels = self.subload("test_batch")
		self.dict = {0: "airplane",
					 1: "automobile",
					 2: "bird",
					 3: "cat",
					 4: "deer",
					 5: "dog",
					 6: "frog",
					 7: "horse",
					 8: "ship",
					 9: "truck"}

	def scale_and_rotate_image(self, images, angle_range=15.0, scale_range=0.1):
		shp = images.shape
		images = images.reshape(shp[0], shp[1], np.prod(shp[2:])).transpose(1,2,0)
		angle = 2 * angle_range * np.random.random() - angle_range
		scale = 1 + 2 * scale_range * np.random.random() - scale_range
		tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(angle))
		tf_scale = transform.SimilarityTransform(scale=scale)
		tf_shift = transform.SimilarityTransform(translation=[-14, -14])
		tf_shift_inv = transform.SimilarityTransform(translation=[14, 14])
		images = transform.warp(images, (tf_shift + tf_scale + tf_rotate + tf_shift_inv).inverse)
		return images.transpose(2,0,1).reshape(shp)

	def cut(self, batch, sizes, pos="rand"):
		try:
			shape = batch[0].get_shape().as_list()[:-1]
		except:
			shape = batch[0].shape
		if shape == sizes:
			return batch
		xl, yl = shape[0]-sizes[0], shape[1]-sizes[1]
		if pos is "rand":
			xr = np.random.randint(xl+1) if xl>0 else 0
			xl -= xr
			yr = np.random.randint(yl+1) if yl>0 else 0
			yl -= yr
			xr, yr = shape[0]-xr, shape[1]-yr
		else:
			xr = int(xl/2)
			xl -= xr
			yr = int(yl/2)
			yl -= yr
			xl, xr, yl, yr = xl+pos[0], shape[0]-xr+pos[0], yl+pos[1], shape[1]-yr+pos[1]
		batch = batch[:,xl:xr,yl:yr,:]
		return batch

	def get_batches(self, batch_size, mode="train", sizes=(32,32), pos=False, rotations=False):
		data_set = self.test if mode=="test" else self.train
		if not pos:
			pos = "rand" if mode=="train" else (0,0)
		rand = np.arange(len(data_set.images))
		np.random.shuffle(rand)
		images, labels = data_set.images[rand], data_set.labels[rand]
		if rotations:
			images = self.scale_and_rotate_image(images)
		spt = [batch_size*(i+1) for i in range(int(len(labels)/batch_size))]
		#images = self.cut(images, sizes, pos)
		images = np.split(images, spt)
		labels = np.split(labels, spt)
		for i in range(len(images)-1): # we want every batch cuted different
			images[i] = self.cut(images[i], sizes)
		return images[:-1], labels[:-1]

	def display(self, imgs, lbls=None, x=2, y=2, directory="./", filename=None):
		imgs = (imgs+1)/2
		if lbls is not None:
			lbls = np.argmax(lbls, axis=1)
		fig, axes1 = plt.subplots(x,y,figsize=imgs.shape[1:-1])
		for j in range(x):
			for k in range(y):
				axes1[j][k].set_axis_off()
				axes1[j][k].imshow(toimage(imgs[j*y+k]), interpolation='none')
				if lbls is not None:
					axes1[j][k].set_title(self.dict[lbls[j*y+k]], fontsize=100)
		if not filename:
			plt.show()
		else:
			plt.savefig(directory+filename+".png")
			plt.close('all')

#-----------------------------------------------------------------------------------------------------

class MnistLoader(object):
	def subload(self, dat_set):
		data = Data()
		data.images = dat_set.images.reshape([-1,1,28,28]).transpose(0,2,3,1)
		#data.images = data.images*2 - 1  # [0,1] --> [-1,1]
		data.labels = dat_set.labels
		return data

	def __init__(self, directory):
		self.directory = directory
		self.name = "MNIST"
		data = input_data.read_data_sets(self.directory, one_hot=True)
		self.train = self.subload(data.train)
		self.validation = self.subload(data.validation)
		self.test = self.subload(data.test)

	def scale_and_rotate_image(self, images, angle_range=15.0, scale_range=0.1):
		shp = images.shape
		images = images.reshape(shp[0], shp[1], np.prod(shp[2:])).transpose(1,2,0)
		angle = 2 * angle_range * np.random.random() - angle_range
		scale = 1 + 2 * scale_range * np.random.random() - scale_range
		tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(angle))
		tf_scale = transform.SimilarityTransform(scale=scale)
		tf_shift = transform.SimilarityTransform(translation=[-14, -14])
		tf_shift_inv = transform.SimilarityTransform(translation=[14, 14])
		images = transform.warp(images, (tf_shift + tf_scale + tf_rotate + tf_shift_inv).inverse)
		return images.transpose(2,0,1).reshape(shp)

	def cut(self, batch, sizes, pos="rand"):
		try:
			shape = batch[0].get_shape().as_list()[:-1]
		except:
			shape = batch[0].shape
		if shape == sizes:
			return batch
		xl, yl = shape[0]-sizes[0], shape[1]-sizes[1]
		if pos is "rand":
			xr = np.random.randint(xl+1) if xl>0 else 0
			xl -= xr
			yr = np.random.randint(yl+1) if yl>0 else 0
			yl -= yr
			xr, yr = shape[0]-xr, shape[1]-yr
		else:
			xr = int(xl/2)
			xl -= xr
			yr = int(yl/2)
			yl -= yr
			xl, xr, yl, yr = xl+pos[0], shape[0]-xr+pos[0], yl+pos[1], shape[1]-yr+pos[1]
		batch = batch[:,xl:xr,yl:yr,:]
		return batch

	def get_batches(self, batch_size, mode="train", sizes=(28,28), rotations=False):
		data_set = self.validation if mode=="valid" else self.test if mode=="test" else self.train
		rand = np.arange(len(data_set.images))
		np.random.shuffle(rand)
		images, labels = data_set.images[rand], data_set.labels[rand]
		if rotations:
			images = self.scale_and_rotate_image(images)
		spt = [batch_size*(i+1) for i in range(int(len(labels)/batch_size))]
		images = np.split(images, spt)
		labels = np.split(labels, spt)
		for i in range(len(images)-1):
			images[i] = self.cut(images[i], sizes)
		return images[:-1], labels[:-1]

	def display(self, imgs, lbls=None, x=2, y=2, directory="./", filename="konv_MNIST"):
		#imgs = (imgs+1)/2  # [0,1] <-- [-1,1]
		imgs = 1 - imgs
		imgs = imgs.reshape(-1, 28, 28)
		if lbls is not None:
			lbls = np.argmax(lbls, axis=1)
		fig, axes1 = plt.subplots(x,y,figsize=(28,28))
		for j in range(x):
			if j*y >= imgs.shape[0]:
				break
			for k in range(y):
				if j*y+k >= imgs.shape[0]:
					break
				axes1[j][k].set_axis_off()
				axes1[j][k].imshow(imgs[j*y+k], cmap='gray', interpolation='none')
				if lbls is not None:
					axes1[j][k].set_title(lbls[j*y+k], fontsize=100)
		fig.savefig(directory+filename+".png")
		plt.close('all')

#-----------------------------------------------------------------------------------------------------
