import numpy as np
import cv2 as cv
import sys
from tifffile import TiffFile

try:
	path_file = open("path.txt", "r")
	sys.path.append(path_file.read())
	path_file.close()
except:
	print("Can't import, path.txt doesn't exist")
	pass

import wholeBrain as wb
import fileManager as fm

import matplotlib.pyplot as plt
import math
from hdf5manager import hdf5manager
import os

def toNumpy(tiffObject):
	pages = 0
	for frame in tiffObject:
		pages += 1

	ar = np.empty((pages,) + tiffObject[0].shape, dtype="uint8")
	for i, frame in enumerate(tiffObject):
		ar[i] = frame.asarray()

	return ar

vid_name = "180720_05_under"

fps = 30
output = vid_name + "_shape.avi"

fourcc = cv.VideoWriter_fourcc('M','J','P','G') 
x1 = 0
x2 = 100
x3 = 100
x4 = 100

directory = "Assets"
videofiles = fm.findFiles(directory, '(\d{6}_\d{2})\D+([@-](\d{4}))?\.tiff?', regex=True)
experiments = fm.movieSorter(videofiles)
print(experiments)
experimentName = "180713_12"
out = cv.VideoWriter(output, fourcc, fps, (640,480), isColor=True)
frame = (np.random.random((480,640,3))*255).astype("uint8")


def writeFrame(frame, out, cmap = None, fmin = 0, fmax = 255):
		
	# rescale and convert to uint8
	fslope = 255 / fmax
	frame = frame - fmin
	frame = frame * fslope

	# cap min and max to prevent incorrect cmap application
	frame[frame > 255] = 255
	frame[frame < 0] = 0

	frame = frame.astype('uint8')

	# apply colormap, write frame to .avi
	if cmap is not None: 
		frame = cv.applyColorMap(frame.astype('uint8'), cmap)
	else:
		if (len(frame.shape) == 2):
			frame = np.repeat(frame[:,:,None], 3, axis=2)

	out.write(frame)


for i in range(300):
	print(i)
	writeFrame(frame, out)

out.release()