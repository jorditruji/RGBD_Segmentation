import subprocess
import os, sys
import numpy as np

import time
import shutil
import _pickle as cPickle
import random
import numpy as np
from scipy import misc, ndimage



def read_image(name):
	img=misc.imread(name)
	img_Resize= misc.imresize(img, (640, 480))

	return img_Resize

def load_data(path):
	filename = path
	images =[]
	labels=[]
	with open(filename) as f:
		for line in f:
			prova =line.strip().split(' ')
			images.append(read_image(prova[0]))


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




create_mean('/imatge/jmorera/PSPNet-Keras-tensorflow/train.txt')

