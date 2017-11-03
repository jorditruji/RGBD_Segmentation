import subprocess
import os, sys
import numpy as np
from itertools import islice
import time
import shutil
from keras.utils import np_utils
import random
import cv2
import numpy as np
from scipy import misc, ndimage, io
import re
from itertools import product
DATA_MEAN = np.array([[[126.92261499, 114.11585906, 99.15394194]]])  # RGB order


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|'):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total:
		print()


def read_image(name):
	img=misc.imread(name)
	img_Resize= misc.imresize(img, (640, 480))

	return img_Resize

def read_label(name):
	label=io.loadmat(name)[name[:-4]]
	#label = label.ravel()
	#label = np_utils.to_categorical(label, 16)
	#print (label.shape)
	return label


def load_data(path,num_img):
	while True:
		filename = path
		images =[]
		labels=[]
		cont=0
		i=0
		with open(filename) as f:
			head = list(islice(f, 2000))
		
			for line in head:
					#printProgressBar(i + 1, len(head), prefix='Progress:', suffix='Complete', length=50)
				i += 1
				#print (line)
				if i<9:
					prova =line.strip().split(' ')
					img=read_image(prova[0])
					float_img = img.astype('float16')
					centered_image = float_img - DATA_MEAN
					bgr_image = centered_image[:, :, ::-1]  # RGB => BGR
					input_data = bgr_image[np.newaxis, :, :, :] 
					images.append(input_data)
					labels.append(read_label(prova[1]))
		images=np.array(images)
		labels=np.array(labels)
		images= np.squeeze(images)
			#print (images.shape)
			#print (labels.shape)
			#labels = labels.reshape(num_img, 307200)
		#y_test = y_test.reshape(100, 307200)
		#y_train = y_train.reshape(100, 307200)
		#y_test = y_test.reshape(100, 307200)
		yield images, labels


def equalize_hist(img):
    equ = cv2.equalizeHist(img)
    #res = np.hstack((img,equ)) #stacking images side-by-side
    return equ
    #cv2.imwrite('res.png',res)


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def img2int(img):
    zmax=np.max(img)
    norm_img=np.zeros(img.shape,dtype=np.uint8)
    mask=np.zeros(img.shape,dtype=np.uint8)
    mask_std=np.zeros(img.shape,dtype=np.uint8)
    cont=0
    h,w = img.shape
    for pos in product(range(h), range(w)):
    #for idx in img:
        pixel =  img.item(pos[0],pos[1])
        if pixel>0:
            new_pix=np.multiply((((1.0/float(pixel))-(1.0/float(zmax)))/((1.0-(1.0/float(zmax))))),1.0)
            new_pix2=(float(pixel)/float(zmax))*254.0
           # print new_pix
      #  print new_pix
            norm_img[pos]=255-new_pix2
            mask[pos]=0
            mask_std[pos]=255
        else:
            norm_img[pos]=0
            if (pos[1]>20 and pos [0]>20):
                mask[pos]=255
                mask_std[pos]=0
            else:
                mask[pos]=255
                mask_std[pos]=0
        cont+=1
    #print (np.unique(norm_img))

    dst_TELEA = cv2.inpaint(norm_img,mask,3,cv2.INPAINT_TELEA)
    dst_TELEA=equalize_hist(dst_TELEA)
    return dst_TELEA


def load_data_V2(path,num_img):
	while True:
		filename = path
		images =[]
		labels=[]
		cont=0
		i=0
		with open(filename) as f:
			head = list(islice(f, 1000))
		
			for line in head:
					#printProgressBar(i + 1, len(head), prefix='Progress:', suffix='Complete', length=50)
				i += 1
				#print (line)
				
				prova =line.strip().split(' ')
				img=read_image(prova[0])
				float_img = img.astype('float16')
				centered_image = float_img - DATA_MEAN
				bgr_image = centered_image[:, :, ::-1]  # RGB => BGR
				input_data = bgr_image[np.newaxis, :, :, :] 
				images.append(input_data)
				a=prova[1]
		#		print (a[:-4])
				labels.append(img2int(read_pgm(a[:-4],'>')))

				if i%3==0:
					images=np.array(images)
					labels=np.array(labels)
					images= np.squeeze(images)
					#images= np.expand_dims(images,0)
					yield images, labels
					images = []
					labels = []


					#labels.append(read_label(prova[1]))

			#print (images.shape)
			#print (labels.shape)
			#labels = labels.reshape(num_img, 307200)
		#y_test = y_test.reshape(100, 307200)
		#y_train = y_train.reshape(100, 307200)
		#y_test = y_test.reshape(100, 307200)
		#yield images, labels


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


