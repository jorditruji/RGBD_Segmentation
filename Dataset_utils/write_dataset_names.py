import subprocess
import os, sys
import numpy as np

import time
import shutil
import _pickle as cPickle
import random


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def load_names_from_folders(path):
    names = os.listdir(path)
    return names
images, labels = [], []
path_or='/projects/world3d/2017-06-scannet/'
outpath = '/imatge/jmorera/PSPNet-Keras-tensorflow'

names = load_names_from_folders(path_or)
filelist = []  # list of lists

for folders in names:

	if os.path.isdir(path_or+folders):
		path_to_folder= path_or+folders+'/'

		names =load_names_from_folders(path_to_folder)
		n_files=len(names)
	#print n_files
		for files in names:
			fileName, fileExtension = os.path.splitext(files)
			
		#	if fileExtension == '.jpg' or fileExtension == '.JPG':
		#		path_to_file=path_to_folder+files
		#		images.append(path_to_file)
			if fileExtension == '.mat':
				
				path_to_file=path_to_folder+files+'.mat'
				path_to_jpg = path_to_folder+files[0:18]+'.jpg'
				labels.append(path_to_file)
				images.append(path_to_jpg)


images = np.array(images)
labels = np.array(labels)
images, labels = shuffle_in_unison(images, labels)
percentage = 0.5
X_train = images[0:round(len(images) * percentage)]
y_train = labels[0:round(len(labels) * percentage)]

X_test = images[round(len(images) * percentage):]
y_test = labels[round(len(labels) * percentage):]

os.chdir(outpath)

trainfile = open("train.txt", "w")
print("writting train file")
print (len(X_train))
for i, l in zip(X_train, y_train):
	trainfile.write(i + " " + str(l) + "\n")
	#print ('train: '+str(i))
print("writting test file")
print (len(X_test))
testfile = open("val.txt", "w")
for i, l in zip(X_test, y_test):
	testfile.write(i + " " + str(l) + "\n")


trainfile.close()
testfile.close()