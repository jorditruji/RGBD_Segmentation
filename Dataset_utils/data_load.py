import subprocess
import os, sys
import numpy as np
from itertools import islice
import time
import shutil

import random
import numpy as np
from scipy import misc, ndimage, io

DATA_MEAN = np.array([[[126.92261499, 114.11585906, 99.15394194]]])  # RGB order

def read_image(name):
	img=misc.imread(name)
	img_Resize= misc.imresize(img, (640, 480))

	return img_Resize

def read_label(name):
	label=io.loadmat(name)
	print (label)
	return label


def load_data(path,num_img):
	filename = path
	images =[]
	labels=[]
	cont=0
	with open(filename) as f:
		head = list(islice(f, num_img))
		for line in head:
			print (line)
			if cont<num_img:
				prova =line.strip().split(' ')
				img=read_image(prova[0])
				float_img = img.astype('float16')
        		centered_image = float_img - DATA_MEAN
        		bgr_image = centered_image[:, :, ::-1]  # RGB => BGR
        		input_data = bgr_image[np.newaxis, :, :, :] 
				images.append(input_data)
				labels.append(prova[1])
	images=np.array(images)
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


read_label('/projects/world3d/2017-06-scannet/scene0292_00/frame-000175.depth.pgm.mat')