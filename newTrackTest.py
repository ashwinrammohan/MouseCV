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
from hdf5manager import hdf5manager
import os
from multiprocessing import Process, Array, cpu_count, Manager
import ctypes as c

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

	obj_shape = (tiffObject[0].shape[0], tiffObject[0].shape[1])
	ar = np.empty((pages,) + obj_shape, dtype="uint8")
	for i, frame in enumerate(tiffObject):
		ar[i] = frame.asarray()[:,:,0]

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

print("Finding gravities...")

print("Making inverse square array")
inv_square = np.empty((dif_vid.shape[1]*2-1, dif_vid.shape[2]*2-1))
center_x = dif_vid.shape[1]-1
center_y = dif_vid.shape[2]-1

wanted_half_radius = 15
val = wanted_half_radius * wanted_half_radius

for x in range(inv_square.shape[0]):
	for y in range(inv_square.shape[1]):
		dx = x - center_x
		dy = y - center_y
		inv_square[x,y] = val / (dx*dx + dy*dy + val)


def _process(output_vid, vid_start, dif_vid, inv_square, output_vid_shape):
	print("Thresholding")
	stds = np.std(dif_vid, axis=(1,2))
	avgs = np.mean(dif_vid, axis=(1,2))

	output_vid_np = np.frombuffer(output_vid.get_obj()).reshape(output_vid_shape) # extract numpy matrix from special objects

	frame_n = vid_start
	for i in range(dif_vid.shape[0]):
		frame = dif_vid[i]
		std = stds[i]
		avg = avgs[i]

		points = np.where(frame > avg + 3*std)
		xs = points[0]
		ys = points[1]

		for x in range(frame.shape[0]):
			sub_x = center_x - x
			for y in range(frame.shape[1]):
				sub_y = center_y - y
				sub_section = inv_square[sub_x : sub_x+frame.shape[0], sub_y : sub_y+frame.shape[1]]
				output_vid_np[i+vid_start,x,y] = np.nansum(inv_square[xs, ys])

		print("Frame done:", frame_n)
		frame_n += 1


threads = []
wanted_threads = cpu_count()
print("Creating " + str(wanted_threads) + " threads...")
output_vid = Array(c.c_double, dif_vid.shape[0]*dif_vid.shape[1]*dif_vid.shape[2]) # create inter-process variables
lower = 0
dataPer = int(dif_vid.shape[0] // wanted_threads)

for i in range(wanted_threads): # create all threads

	if i == wanted_threads-1: # for the last thread, give it any leftover rows of data, for example 23 rows, 5 threads, this process will do #17-23
		_process(output_vid, upper, dif_vid[upper:], inv_square, dif_vid.shape)
	else: # otherwise just divide up the rows into each process normally
		lower = i*dataPer
		upper = (i+1)*dataPer
		p = Process(target=_process, args=(output_vid, lower, dif_vid[lower:upper], inv_square, dif_vid.shape))
	
	p.start()
	threads.append(p)

for thread in threads:
	thread.join()

dif_vid = np.frombuffer(output_vid.get_obj()).reshape(dif_vid.shape)

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
new_frame = np.empty((dif_vid.shape[1], dif_vid.shape[2], 3), dtype="uint8")
for frame in dif_vid:
	new_frame[:,:,0] = frame
	new_frame[:,:,1] = frame
	new_frame[:,:,2] = frame
	out.write(new_frame)

out.release()