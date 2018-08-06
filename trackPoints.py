from hdf5manager import hdf5manager as h5
import numpy as np
import cv2 as cv
import time
import math
from tifffile import TiffFile

f = h5("mouse_vectors.hdf5").load()
contour_data = f["contour_data"].astype("int32")
n_contours = f["n_contours"]	

f_bl = (0,0)
f_br = (0,0)
f_fl = (0,0)
f_fr = (0,0)
fs = [f_bl, f_br, f_fl, f_fr]

experimentName = "180713_12"
vid_name = experimentName + "_under"

directory = "Assets"
videofiles = fm.findFiles(directory, '(\d{6}_\d{2})\D+([@-](\d{4}))?\.tiff?', regex=True)
experiments = fm.movieSorter(videofiles)

def toNumpy(tiffObject):
	pages = 0
	for frame in tiffObject:
		pages += 1

	obj_shape = (tiffObject[0].shape[0], tiffObject[0].shape[1])
	ar = np.empty((pages,) + obj_shape, dtype="uint8")
	for i, frame in enumerate(tiffObject):
		ar[i] = frame.asarray()[:,:,0]

	return ar

def limb_track():
	for frame, n in zip(contour_data, n_contours):
		dsts = np.repeat(-1, 4)
		for x, y in frame[:n]:


def limb_click(event, x, y, flags, param):
	global f_bl, f_br, f_fl, f_fr, index
	if index == 0:
		f_bl = (x, y)
		index += 1
		print("Click back right...")
	elif index == 1:
		f_br = (x, y)
		index += 1
		print("Click front left...")
	elif index == 2:
		f_fl = (x, y)
		index += 1
		print("Click front right...")
	elif index == 3:
		f_fr = (x, y)
		limb_track()


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

mv_frame = toNumpy(TiffFile(pathlist[0]).pages).astype("uint8")[200]
index = 0

print("Click back left...")
cv.namedWindow("Original")
cv.setMouseCallback("Original", limb_click)
cv.imshow("Original", mv_frame)