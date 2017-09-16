import scipy.io
import matplotlib.pyplot as plt
#from PIL import Image
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import cv2

def make_labels(img):
	padded_data = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
	data = img
	h, w = padded_data.shape
	labels = np.zeros(data.shape, dtype=int)
	#ITERATE AND LABEL EVERY PIXEL
	for pos in product(range(h-2), range(w-2)):
		pixel = padded_data.item(pos[0]+1,pos[1]+1)+1
		if pixel>0:
			p_left = padded_data.item(pos[0]+1,pos[1])+1
			p_right = padded_data.item(pos[0]+1,pos[1]+2)+1
			p_up = padded_data.item(pos[0],pos[1]+1)+1
			p_down = padded_data.item(pos[0]+2,pos[1]+1)+1		
			#NETEJA: PIXELS VEINS DE VALOR DIFERENT AL CENTRE POSATS A VALOR_CENTRE+1
			const = pixel-1
			if padded_data.item(pos[0]+1,pos[1])+1 != pixel:
				p_left = const

			if padded_data.item(pos[0]+1,pos[1]+2)+1 != pixel:
				p_right = const

			if padded_data.item(pos[0],pos[1]+1)+1 != pixel:
				p_up = const

			if padded_data.item(pos[0]+2,pos[1]+1)+1 != pixel:
				p_down = const
		#RESTEM PIXEL A ELL MATEIX I ELS SEUS VEINS ACABANT AMB LA MATEIXA SITUACIO QUE EN LA SEGMENTACIO ANTERIOR:
			pixel = 1
			p_left=p_left-const
			p_right=p_right-const
			p_up=p_up-const
			p_down=p_down-const
			non_zeros = np.count_nonzero([p_left, p_right, p_up, p_down])
			#3 uns i 1 zero de veins
			if non_zeros == 3:
				#CLASSE 10: RIGHT
				if p_right==10:
					labels[pos]=0
				#CLASSE 1 :BAIX
				if p_down==0:
					labels[pos]=1
				#CLASSE 2: LEFT
				if p_left==0:
					labels[pos]=2
				#CLASSE 3 :UP
				if p_up==0:
					labels[pos]=3
			
			#2 uns i 2 zeros de veins					
			if non_zeros == 2:
				if p_up ==1:
					#CLASSE 4:RIGHT-DOWN
					if p_left == 1:
						labels[pos]=4
					#CLASSE 5:LEFT-DOWN
					if p_right == 1:
						labels[pos]=5
					#CLASSE 8:VERTICAL
					if p_down == 1:
						labels[pos]=8
						
					#else if p_down == 1:
					#CLASSE 6:LFET-UP
					if p_right == 1:
						labels[pos]=6
					#CLASSE 7:RIGHT-UP
					if p_left == 1:
						labels[pos]=7
				else:
					#CLASSE 9:HORITZONTAL
					labels[pos]=9
			#1 ZERO 3 UNS
			if non_zeros == 1:
				if p_up ==1:
					#CLASSE 12:SINGLE_UP
					labels[pos]=12
				if p_left == 1:
					#CLASSE 13: SINGLE_LEFT
					labels[pos]=13
				if p_right == 1:
					#CLASSE 14: SINGLE_RIGHT
					labels[pos]=14
				if p_down == 1:
					#CLASSE 15: SINGLE_DOWN
					labels[pos]=15

			if non_zeros == 4:
				#CLASSE 0: INTERSECTION
				labels[pos]=0
	return labels
