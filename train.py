from __future__ import print_function
from __future__ import division
from os.path import splitext, join, isfile
from os import environ
from math import ceil
from keras.optimizers import SGD, Adam
import argparse
from keras.utils import np_utils
import numpy as np
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
import layers_builder as layers
import utils
import matplotlib
import cv2
import scipy.io
matplotlib.use('Agg')
import os
from keras.layers import Activation, Dense, Flatten, Conv2D, Lambda, Input, Reshape
from keras.models import Model,Sequential
from pspnet import PSPNet50
from data_load import *

pspnet_ini = PSPNet50(nb_classes=150, input_shape=(640, 480),
                              weights='pspnet50_ade20k')

pspnet_ini.model.summary()

# DELETE SOFTMAX LAYER
pspnet_ini.model.layers.pop()

# DELETE INTERPOLATION LAYER:
#pspnet_ini.model.layers.pop()

# ADD REGRESSION LAYER
out = Flatten()(pspnet_ini.model.layers[-1].output)
out =Dense(1, name='my_dense')(pspnet_ini.model.layers[-1].output)


inp = pspnet_ini.model.input
model2 = Model(inp, out)
model2.compile(loss='mean_absolute_error', optimizer='sgd')
model2.summary()
# TRAINING
history= model2.fit_generator(
	load_data_V2('/imatge/jmorera/PSPNet-Keras-tensorflow/train.txt', 4),
	steps_per_epoch = 800,
	nb_epoch = 20,
	verbose=1,
	validation_data=load_data_V2('/imatge/jmorera/PSPNet-Keras-tensorflow/val.txt', 4),
	validation_steps=200)



