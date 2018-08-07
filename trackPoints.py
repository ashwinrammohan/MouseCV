from hdf5manager import hdf5manager as h5
import numpy as np
import cv2 as cv
from tifffile import TiffFile

import time, math, sys

try:
	path_file = open("path.txt", "r")
	sys.path.append(path_file.read())
	path_file.close()
except:
	print("Can't import, path.txt doesn't exist")
	pass

import fileManager as fm

experimentName = "180807_01"
vid_name = experimentName + "_under"

f = h5("mouse_vectors_"+experimentName+".hdf5").load()
f_write = h5("new_mouse_vectors_"+experimentName+".hdf5")
contour_data = f["contour_data"].astype("int32")
n_contours = f["n_contours"]	

names = ["bl", "br", "fl", "fr"]
write_dict = {"bl":np.empty((contour_data.shape[0], 2)), "br":np.empty((contour_data.shape[0], 2)), "fl":np.empty((contour_data.shape[0], 2)), "fr":np.empty((contour_data.shape[0], 2))}


stds = f["stds"]
max_std = np.max(stds)
min_std = np.min(stds)
stds -= min_std
stds /= max_std - min_std

wanted_min = 15
wanted_max = 100
stds *= wanted_max - wanted_min
stds += wanted_min

print("Min dist:", np.min(stds), "Max dist:", np.max(stds))

fps = 30
output = "Outputs/" + vid_name + "_shape.avi"
fourcc = cv.VideoWriter_fourcc('M','J','P','G') 
cap = cv.VideoCapture("Assets/"+vid_name+".mp4")

f_bl = (0,0)
f_br = (0,0)
f_fl = (0,0)
f_fr = (0,0)
fs = [f_bl, f_br, f_fl, f_fr]
num_limbs = 4

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

frame_n = 0
def limb_track():
	global frame_n

	cv.namedWindow("Dots")
	fps = 30
	frame_dt = 0 #1.0 / fps
	mv_i = 0
	pause = False

	while True:
		print(mv_i)
		if frame_n >= contour_data.shape[0]:
			mv_i = 0

		t = time.clock()
		ret, im = cap.read()


		frame = contour_data[frame_n]
		n = n_contours[frame_n]
		dists = np.repeat(-1, num_limbs)
		new_pos = np.empty((num_limbs, 2), dtype=contour_data.dtype)
		max_dist = stds[frame_n]*stds[frame_n]
		if frame_n == 0:
			max_dist = 50**2

		for x, y in frame[:n]:
			for i in range(num_limbs):
				fx, fy = fs[i]
				dx = fx - x
				dy = fy - y
				dist = dx*dx + dy*dy

				if dist < dists[i] or dists[i] == -1:
					dists[i] = dist
					new_pos[i][0] = x
					new_pos[i][1] = y

		avg_sz = 5
		d_clr = 30
		im = cv.GaussianBlur(im, (5, 5), 0)
		for i in range(num_limbs):
			if dists[i] < max_dist:
				fs[i] = (new_pos[i][0], new_pos[i][1])

			x, y = fs[i]

			avg_clr = np.mean(im[y-avg_sz:y+avg_sz, x-avg_sz:x+avg_sz, 0])
			min_clr = avg_clr - d_clr
			max_clr = avg_clr + d_clr

			retval, result = cv.threshold(im[:,:,0], max_clr, 255, cv.THRESH_TOZERO_INV);
			retval, result = cv.threshold(result, min_clr, 255, cv.THRESH_BINARY);
			a, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

			min_dist = -1
			wanted_contour_i = 0
			cnt_pos = (0,0)
			for j, cnt in enumerate(contours):
				m = cv.moments(cnt)
				if m['m00'] <= 100:
					continue

				cnt_x = int(m['m10']/m['m00'])
				cnt_y = int(m['m01']/m['m00'])
				area = m["m00"]
				dx = cnt_x - x
				dy = cnt_y - y
				dist = dx*dx + dy*dy
				if min_dist == -1 or dist < min_dist:
					min_dist = dist
					wanted_contour_i = j
					cnt_pos = (cnt_x, cnt_y)

			cv.drawContours(im, contours, wanted_contour_i, (0,255,0), 1)
			im[y:y+3, x:x+3, 1:3] = 0
			im[y:y+3, x:x+3, 0] = 255
			x, y = cnt_pos
			im[y:y+3, x:x+3, 0:2] = 0
			im[y:y+3, x:x+3, 2] = 255

			write_dict[names[i]][frame_n,0] = x
			write_dict[names[i]][frame_n,1] = y


		#cv.putText(im, str(frame_n), (5,25), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
		cv.imshow("Dots", im)

		if pause:
			k = cv.waitKey(0)
		else:
			dt = frame_dt - (time.clock() - t)
			dt_mili = int(dt * 1000)

			if (dt_mili < 1):
				dt_mili = 1

			k = cv.waitKey(dt_mili)
			mv_i += 1
			frame_n += 1

		if k == 27: # esc key
			print("Frames completed:", frame_n)
			f_write.save(write_dict)
			break
		elif k == 32: # space key
			pause = not(pause)
		elif k == 63235 and pause: # right arrow
			mv_i += 1
			frame_n += 1
			print(stds[frame_n])
		elif k == 63234 and pause: # left arrow		
			mv_i -= 1
			frame_n -= 1
			print(stds[frame_n])


def limb_click(event, x, y, flags, param):
	global index
	if event == cv.EVENT_LBUTTONUP:
		if index == 0:
			fs[index] = (x, y)
			index += 1
			print("Click back right...")
		elif index == 1:
			fs[index] = (x, y)
			index += 1
			print("Click front left...")
		elif index == 2:
			fs[index] = (x, y)
			index += 1
			print("Click front right...")
		elif index == 3:
			fs[index] = (x, y)
			cv.destroyAllWindows()

			# for path in pathlist:
			# 	mouse_vid = toNumpy(TiffFile(path).pages).astype("float")
			# 	limb_track(mouse_vid)

			limb_track()

			print("All Done!")


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

mv_frame = toNumpy(TiffFile(pathlist[0]).pages).astype("uint8")[0]
w = mv_frame.shape[0]
h = mv_frame.shape[1]
out = cv.VideoWriter(output, fourcc, fps, (h,w), isColor = True)
index = 0

print("Click back left...")
cv.namedWindow("Original")
cv.setMouseCallback("Original", limb_click)
cv.imshow("Original", mv_frame)

out.release()

cv.waitKey(0)
cv.destroyAllWindows()