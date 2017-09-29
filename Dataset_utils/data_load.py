import subprocess
import os, sys
import numpy as np

import time
import shutil

import random
import numpy as np
from scipy import misc, ndimage



def read_image(name):
	img=misc.imread(name)
	img_Resize= misc.imresize(img, (640, 480))

	return img_Resize

def load_data(path,num_img):
	filename = path
	images =[]
	labels=[]
	with open(filename) as f:
		for line in f[0:num_img]:
			prova =line.strip().split(' ')
			images.append(read_image(prova[0]))
			labels.append(prova[1])
	images=np.array(image.astype('float32'))
	return images, labels

def create_mean(path):
	filename = path
	images =[]
	labels=[]
	with open(filename) as f:
		for line in f:
			prova =line.strip().split(' ')

			images.append(calc_mean(read_image(prova[0])))

	print (sum(images) / float(len(images)))
	return sum(images) / float(len(images))


def calc_mean(image):

	return np.mean(image, axis=(0, 1))




[x_train,x_labels]=load_data('/imatge/jmorera/PSPNet-Keras-tensorflow/train.txt',10)

print (x_train.shape)