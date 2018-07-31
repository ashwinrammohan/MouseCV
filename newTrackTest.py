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

vid_name = "180720_05_under"

fps = 30
output = "Outputs/" + vid_name + "_shape.avi"

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

def toNumpy(tiffObject):
	pages = 0
	for frame in tiffObject:
		pages += 1

	ar = np.empty((pages,) + tiffObject[0].shape, dtype="uint8")
	for i, frame in enumerate(tiffObject):
		ar[i] = frame.asarray()

	return ar

pathlist = []
for expname in sorted(experiments):
	if experimentName is not None:
		if expname != experimentName:
			print(expname, 'does not match experiment key:', 
				experimentName +'.  skipping..')
			continue
		else:
			print('found match:', expname)
	# Make output filenames based on name
	pathlist.extend(experiments[expname])
	mouse_vid = toNumpy(TiffFile(pathlist[0]).pages).astype("float")
	mouse_vid_shape = mouse_vid.shape
	print("Found experiment!!")

print("Loaded movie part 1")
hdf5FilePath = "mouse_vectors.hdf5"
hdf5File = hdf5manager(hdf5FilePath)

hdf5Dict = {}
clicks = {"footFL":0, "footFR":0, "footBL":0, "footBR":0, "tail":0, "head":0}
lightest = {"footFL":0, "footFR":0, "footBL":0, "footBR":0, "tail":0, "head":0}
darkest = {"footFL":255, "footFR":255, "footBL":255, "footBR":255, "tail":255, "head":255}
selected = {"footFL":False, "footFR":False, "footBL":False, "footBR":False, "tail":False, "head":False}
pos = {"footFL":(0,0), "footFR":(0,0), "footBL":(0,0), "footBR":(0,0), "tail":(0,0), "head":(0,0)}
currentLabel = ""
crop_area = (0,0,100,100)
stage = 0

verbose = True

w = mouse_vid_shape[1]
h = mouse_vid_shape[2]
out = cv.VideoWriter(output, fourcc, fps, (h,w), isColor=True)

print("\n\n\n\n\n\n")
print(mouse_vid_shape)
print("Getting dif")
dif_vid = mouse_vid[1:] - mouse_vid[:-1]

print("Getting abs")
dif_vid = np.abs(dif_vid)

print("Thresholding")
stds = np.std(dif_vid, axis=(1,2))
avgs = np.mean(dif_vid, axis=(1,2))

print("Finding gravities")
frame_n = 0
for frame, std, avg in  zip(dif_vid, stds, avgs):
	print("Frame:", frame_n)
	frame_n += 1

	points = np.where(frame > avg + 3*std)
	xs = points[0]
	ys = points[1]

	for x in range(frame.shape[0]):
		for y in range(frame.shape[1]):
			rs_squared = np.divide(1, (np.square(xs - x) + np.square(ys - y)))
			frame[x,y] = np.nansum(rs_squared)


print("Finding bounds")
dif_max = np.max(dif_vid)
dif_min = np.min(dif_vid)

print("Adjusting bounds")
dif_vid -= dif_min
dif_vid /= dif_max - dif_min
dif_vid *= 255

print("Converting")
dif_vid = dif_vid.astype("uint8")
	
print("Writing to mp4")
for frame in dif_vid:
	out.write(frame)

out.release()