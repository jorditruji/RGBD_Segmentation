#!/usr/bin/env python

'''
This sample demonstrates SEEDS Superpixels segmentation
Use [space] to toggle output mode
Usage:
  seeds.py [<video source>]
'''
from __future__ import division
from labeling import make_labels
import re
import numpy as np
from skimage.segmentation import slic
from itertools import product
import cv2
import time
# built-in module
import sys
import os
from matplotlib import pyplot
import scipy.io as sio
import glob
def load_names_from_folders(path):
    names = os.listdir(path)
    return names

def img2int(img):
    zmax=np.max(img)
    norm_img=np.zeros(img.shape,dtype=np.uint8)
    mask=np.zeros(img.shape,dtype=np.uint8)
    mask_std=np.zeros(img.shape,dtype=np.uint8)
    cont=0
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

# relative module


def segment_image(img, filename):
    converted_img=img
    seeds = None
    display_mode = 0
    num_superpixels = 400
    prior = 2
    num_levels = 6
    num_histogram_bins = 5
    n_iterations=200
    height,width,channels = converted_img.shape
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels,
    num_superpixels, num_levels, prior, num_histogram_bins)
    color_img = np.zeros((height,width,3), np.uint8)
    color_img[:] = (0, 0, 255)
    seeds.iterate(converted_img, n_iterations)

# retrieve the segmentation result
    labels = seeds.getLabels()
    labels_remember=labels
    #print(np.unique(labels_remember))
    labels_remeber=make_labels(labels_remember)
    print (labels_remeber)
    sio.savemat(filename, {filename:labels_remember})
#print(np.unique(labels))

# labels output: use the last x bits to determine the color
    num_label_bits = 2
    labels &= (1<<num_label_bits)-1
    labels *= 1<<(16-num_label_bits)


    mask = seeds.getLabelContourMask(False)

    # stitch foreground & background together
    mask_inv = cv2.bitwise_not(mask)
    result_bg = cv2.bitwise_and(converted_img1, converted_img1, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
    result = cv2.add(result_bg, result_fg)


#cv2.namedWindow('SEEDS')

path_or='/projects/world3d/2017-06-scannet/'
output = '/projects/world3d/2017-06-scannet/'


names = load_names_from_folders(path_or)

filelist = []  # list of lists

for folders in names:
    path_to_file=path_or+folders

    files= load_names_from_folders(path_or)
    print ("processing folder:"+folders)
    for filename in files:
        if filename.endswith('.pgm'):
            image=path_to_file+'/'+filename
            img = read_pgm(image, byteorder='>')
            h,w = img.shape
            dst_TELEA=img2int(img)
            converted_img1 = cv2.cvtColor(dst_TELEA, cv2.COLOR_GRAY2BGR)
            #print(np.unique(converted_img1))
            converted_img = cv2.cvtColor(converted_img1, cv2.COLOR_BGR2HSV)

            segment_image(converted_img,image)
            #print(image)



'''
img = read_pgm(image, byteorder='>')

h,w = img.shape


dst_TELEA=img2int(img)
#print (np.unique(dst_TELEA))
converted_img1 = cv2.cvtColor(dst_TELEA, cv2.COLOR_GRAY2BGR)
#print(np.unique(converted_img1))
converted_img = cv2.cvtColor(converted_img1, cv2.COLOR_BGR2HSV)
filename='/home/Jordi/Desktopframe-000000.depth.pgm'
segment_image(converted_img,filename)
#sio.savemat('seeds_seg.mat', {'frame-000000.depth.pgm':labels_remember})
aa=sio.loadmat('seeds_seg')
'''