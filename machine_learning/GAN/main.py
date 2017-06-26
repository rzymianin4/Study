#!/usr/bin/env python3

# Dawid Tracz 
 # oświadzcam iż poniższy kod został napisany samodzielnie.
 # 

if True:	#imports
	import signal, sys
	from tqdm import tqdm, trange
	import tensorflow as tf
	import numpy as np
	from loader import CifarLoader, MnistLoader
	from discriminator import Discriminator
	from GAN import Generator

def sighandler(sig, frame):
	print("\b\bInterrupted", file=sys.stderr)
	exit(-1)
signal.signal(signal.SIGINT, sighandler)

#-------------------------------------------------------------------------------------------------

DATA_NAME = sys.argv[1]
GPU = int(sys.argv[2])
if DATA_NAME == "MNIST":
	IMG_SIZES = (28,28,1)
elif DATA_NAME == "CIFAR10":
	IMG_SIZES = (32,32,3)
else:
	raise ValueError(DATA_NAME+": unknown data set")
BATCH_SIZE = 128
N_CAT = 10
EPOCH_LEN = 128
NOISE_SIZE = 128
INIT_SCALE = 0.002
DROPOUT = 0.0

#-------------------------------------------------------------------------------------------------

class Trainer(object):
	def __init__(self, G, D, sess, dataSet):
		self.dataSet = dataSet
		self.sess = sess

		self.noise = tf.placeholder(tf.float32, [None, NOISE_SIZE])
		self.glbs = tf.placeholder(tf.float32, [None, N_CAT])

		self.imgs = tf.placeholder(tf.float32, [None]+list(IMG_SIZES)) #########
		self.lbls = tf.placeholder(tf.float32, [None, N_CAT])

		self.gims = G(self.noise, self.glbs)

		self.preds_R = D(self.imgs, self.tf_blur_labels(self.lbls))
		self.preds_F = D(self.gims, self.tf_blur_labels(self.glbs))
		
		self.loss_D = -tf.reduce_mean(0.9*tf.log(self.preds_R) + 1.0*tf.log(1-self.preds_F))
		self.loss_G = -tf.reduce_mean(tf.log(self.preds_F))

		#loss_DR = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.preds_R, labels=0.9*tf.ones_like(self.preds_R)))
		#loss_DF = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.preds_F, labels=tf.zeros_like(self.preds_F)))
		#self.loss_D = loss_DR + loss_DF

		self.gradloss_D = tf.train.GradientDescentOptimizer(D.learning_rate).minimize(self.loss_D, var_list=D.trainables)
		self.gradloss_G = tf.train.AdamOptimizer(G.learning_rate).minimize(self.loss_G, var_list=G.trainables)

		self.ans_R = tf.reduce_mean(self.preds_R)
		self.ans_F = tf.reduce_mean(self.preds_F)

	def tf_blur_labels(self, lbls, mean=0, var=0.015):
		ns = tf.random_normal(tf.shape(lbls), mean=mean, stddev=var, dtype=tf.float32)
		ns = tf.abs(ns)
		lbls += ns
		lbls = tf.transpose(tf.transpose(lbls)/tf.reduce_sum(lbls, axis=-1))
		return lbls

	#---------------------------------------------------------------------------------------------	

	def get_noise(self, batch_size=BATCH_SIZE, noise_size=NOISE_SIZE, mean=0, var=1):
		#noise = np.random.normal(mean, var, size=(batch_size, noise_size))
		noise = np.random.uniform(mean-var, mean+var, size=(batch_size, noise_size))
		return noise
		
	def get_labels(self, batch_size=BATCH_SIZE):
		glbs = np.zeros((batch_size, N_CAT))
		ones = np.random.randint(N_CAT, size=(batch_size))
		glbs[np.arange(batch_size), ones] = 1	
		return glbs

	def blur_labels(self, lbls, mean=0, var=0.015):
		ns = np.random.normal(mean, var, size=lbls.shape)
		ns = np.abs(ns)
		lbls += ns
		lbls = np.transpose(lbls.transpose()/np.sum(lbls, axis=-1))
		return lbls

	def train(self, batch_size=BATCH_SIZE, noise_size=NOISE_SIZE, mean=0, var=1):
		imgs, lbls = self.dataSet.get_batches(batch_size)
		loss_D, loss_G, ans_R, ans_F = 10, 10, 0.5, 0.5
		for i in trange(len(lbls)):
			noise = self.get_noise(batch_size, noise_size, mean, var)
			glbs = self.get_labels(batch_size)
			if loss_D > 4*loss_G:
				glD, loss_D, ans_R, ans_F = self.sess.run((self.gradloss_D, self.loss_D, self.ans_R, self.ans_F), 
												feed_dict={self.noise: noise, self.glbs: glbs,
														   self.imgs: imgs[i], self.lbls: lbls[i]})
				i+=1
			elif loss_G > 4*loss_D:
				glG, loss_G, ans_F = self.sess.run((self.gradloss_G, self.loss_G, self.ans_F), 
										 feed_dict={self.noise: noise, self.glbs: glbs})
			else:
				glD, glG, loss_D, loss_G, ans_R, ans_F = self.sess.run((self.gradloss_D, self.gradloss_G,
																		self.loss_D, self.loss_G,
																		self.ans_R, self.ans_F), 
															 feed_dict={self.noise: noise,
															 			self.glbs: glbs,
															 			self.imgs: imgs[i],
															 			self.lbls: lbls[i]})
				i+=1
		return loss_D, loss_G, ans_R, ans_F

	def generate(self, batch_size=BATCH_SIZE, noise_size=NOISE_SIZE, mean=0, var=1):
		noise = self.get_noise(batch_size, noise_size, mean, var)
		glbs = self.get_labels(batch_size)
		gims = self.sess.run(self.gims, feed_dict={self.noise: noise, self.glbs: glbs})
		return gims, glbs

	def make_sample(self, filename, directory="./gims/", imgs_num_XY=(3,3)):
		imgs_num = imgs_num_XY[0] * imgs_num_XY[1]
		gims, glbs = self.generate(batch_size=imgs_num)
		self.dataSet.display(gims, glbs, x=imgs_num_XY[0], y=imgs_num_XY[1], directory="./gims/", filename=filename)

	def make_transition(self, lbl1, lbl2, noise_size=NOISE_SIZE, mean=0, var=1):
		noise = self.get_noise(1, noise_size, mean, var)
		imgs = []
		i=0
		while i<=1:
			lbl = lbl1*i + lbl2*(1-i)
			lbl = lbl.reshape(1,10)
			gim = self.sess.run(self.gims, feed_dict={self.noise: noise, self.glbs: lbl})
			imgs.append(gim.reshape(gim.shape[1:]))
			i+=0.01
		imgs = np.array(imgs)
		self.dataSet.display(imgs, x=10, y=10, directory="./gims/", filename="mechanic_horse")

	def create_set(self, set_size=50000, batch_size=100, noise_size=NOISE_SIZE, mean=0, var=1):
		imgs = [None]*int(set_size/batch_size)
		lbls = [None]*int(set_size/batch_size)
		for i in trange(500):
			imgs[i], lbls[i] = self.generate(batch_size=batch_size)
		imgs = np.array(imgs)
		imgs = np.reshape(imgs, tuple([set_size]+list(imgs.shape[2:])))
		lbls = np.array(lbls)
		lbls = np.reshape(lbls, tuple([set_size]+list(lbls.shape[2:])))
		np.save("gen_images", imgs)
		np.save("gen_labels", lbls)

#-------------------------------------------------------------------------------------------------

config = tf.ConfigProto( device_count = {'GPU': GPU} )
sess = tf.InteractiveSession(config=config)

if DATA_NAME == "MNIST":
	dataSet = MnistLoader("../MNIST_data/")
elif DATA_NAME == "CIFAR10":
	dataSet = CifarLoader("../cifar-10-batches-py/")
print("\n__________loaded__________\n", file=sys.stderr)
dis = Discriminator(N_CAT, sizes=IMG_SIZES, in_sc=INIT_SCALE, dropout=DROPOUT, learning_rate=2e-3)
print("\n_________dis_DONE_________\n", file=sys.stderr)
gan = Generator(N_CAT, NOISE_SIZE, dis, sizes=IMG_SIZES, in_sc=INIT_SCALE, dropout=0.4, learning_rate=2e-4)
print("\n_________gan_DONE_________\n", file=sys.stderr)
trainer = Trainer(gan, dis, sess, dataSet)
init = tf.global_variables_initializer()
sess.run(init)
savior = tf.train.Saver()
sess.graph.finalize()
print("\n_______trainer_DONE_______\n", file=sys.stderr)

#-------------------------------------------------------------------------------------------------

#savior.restore(sess, "./model_data/GANnDIS_100.ckpt")

i=0
best_loss_G = 1
while True:
	trainer.make_sample(DATA_NAME+'_'+str(i))
	loss_D, loss_G, ans_R, ans_F = trainer.train()
	i+=1
	print(i, "\nlosses::D,G:", loss_D, loss_G, "\nanswers::R,F:", ans_R, ans_F, file=sys.stderr)
	print(i, loss_D, loss_G, ans_R, ans_F)
	sys.stdout.flush()
	if i<80 and i%10==0:
		savior.save(sess, "./model_data/GANnDIS.ckpt")
		print("saved after ", i, ". epoch", sep='', file=sys.stderr)
	if i>=80 and i%10==0:
		savior.save(sess, "./model_data/GANnDIS_"+str(i)+".ckpt")
	if i>120:
		break
#'''
trainer.make_sample(DATA_NAME+"_FIN")

lbl1 = np.array([0,1,0,0,0,0,0,0,0,0])
lbl2 = np.array([0,0,0,0,0,0,0,1,0,0])
trainer.make_transition(lbl1, lbl2)

trainer.create_set()
#'''
