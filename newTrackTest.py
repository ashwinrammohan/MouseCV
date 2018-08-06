import numpy as np
import cv2 as cv
import sys
import math
from tifffile import TiffFile
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

try:
	path_file = open("path.txt", "r")
	sys.path.append(path_file.read())
	path_file.close()
except:
	print("Can't import, path.txt doesn't exist")
	pass

import wholeBrain as wb
import fileManager as fm
import time
import matplotlib.pyplot as plt
from hdf5manager import hdf5manager
import os
from multiprocessing import Process, Array, cpu_count, Manager
import ctypes as c

experimentName = "180713_12"
vid_name = experimentName + "_under"

fps = 30
output = "Outputs/" + vid_name + "_shape.avi"
fourcc = cv.VideoWriter_fourcc('M','J','P','G') 

directory = "Assets"
videofiles = fm.findFiles(directory, '(\d{6}_\d{2})\D+([@-](\d{4}))?\.tiff?', regex=True)
experiments = fm.movieSorter(videofiles)
print(experiments)

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

	pathlist.extend(experiments[expname])
	print("Found experiment")

hdf5FilePath = "mouse_vectors.hdf5"
hdf5File = hdf5manager(hdf5FilePath)
n_frames_total = 19000
max_contours = 50
frame_n = 0
hdf5Dict = {"contour_data":np.empty((n_frames_total, max_contours, 2), dtype='uint32'), "n_contours":np.empty((n_frames_total), dtype='uint32'), "stds":np.empty((n_frames_total))}

fl = TiffFile(pathlist[0])
w = fl.pages[0].shape[0]
h = fl.pages[0].shape[1]
out = cv.VideoWriter(output, fourcc, fps, (h,w), isColor=True)

print("\n\n\n")

def track_vid(mouse_vid, mouse_vid_shape, out, hdf5Dict):
	global frame_n

	contour_data = hdf5Dict["contour_data"]
	n_contours = hdf5Dict["n_contours"]
	stds_hdf5 = hdf5Dict["stds"]

	print(mouse_vid_shape)
	print("Getting dif")
	dif_vid = mouse_vid[1:] - mouse_vid[:-1]

	print("Getting abs")
	dif_vid = np.abs(dif_vid)

	print("Blurring")
	for i in range(dif_vid.shape[0]):
		dif_vid[i] = cv.GaussianBlur(dif_vid[i], (31, 31), 0)

	new_dif_vid = np.zeros_like(dif_vid, dtype='uint8')

	print("Thresholding")
	stds = np.std(dif_vid, axis=(1,2))[...,None,None]
	max_std = np.max(stds)
	min_std = np.min(stds)
	avgs = np.mean(dif_vid, axis=(1,2))[...,None,None]
	thresh = avgs + stds * 3

	new_dif_vid[dif_vid < thresh] = 0
	new_dif_vid[dif_vid >= thresh] = 255

	mouse_vid[1:,-20:,-20:] = (255 * (stds - min_std) / (max_std - min_std)).astype('uint8')

	print("Writing to mp4")
	new_frame = np.empty((dif_vid.shape[1], dif_vid.shape[2], 3), dtype="uint8")
	for frame, bin_frame, std in zip(mouse_vid[1:], new_dif_vid, stds):
		stds_hdf5[frame_n] = std

		new_frame[:,:,0] = frame
		new_frame[:,:,1] = frame
		new_frame[:,:,2] = frame

		blah, contours, hierarchy = cv.findContours(bin_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		areas = np.empty(len(contours))
		cnt_count = 0
		for i, cnt in enumerate(contours):
			m = cv.moments(cnt)
			areas[i] = m["m00"]

			if (areas[i] > 0):
				contour_data[frame_n, cnt_count, 0] = m['m10']/m['m00']
				contour_data[frame_n, cnt_count, 1] = m['m01']/m['m00']
				cnt_count += 1

		n_contours[frame_n] = cnt_count

		sorted_order = np.argsort(areas)
		mx = np.max(areas) / 100
		areas /= mx
		# area_std = int(np.std(areas))
		area_mean = int(np.mean(areas))
		areas = areas.astype('int32')
		areas[areas == 0] = 1

		new_frame[-area_mean:-area_mean+3, :sorted_order.shape[0]*20-10, 2] = 255

		for i, index in enumerate(sorted_order):
			new_frame[-areas[index]:, i*20:i*20+10, 1] = 255

		cv.drawContours(new_frame, contours, -1, (0,0,255), 2)
		out.write(new_frame)
		frame_n += 1


for path in pathlist:
	mouse_vid = toNumpy(TiffFile(path).pages).astype("float")
	mouse_vid_shape = mouse_vid.shape
	track_vid(mouse_vid, mouse_vid_shape, out, hdf5Dict)


hdf5Dict["contour_data"] = hdf5Dict["contour_data"][:frame_n]
hdf5Dict["n_contours"] = hdf5Dict["n_contours"][:frame_n]
hdf5Dict["stds"] = hdf5Dict["stds"][:frame_n]
hdf5File.save(hdf5Dict)
out.release()