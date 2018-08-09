from hdf5manager import hdf5manager as h5
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from tifffile import TiffFile
from scipy.cluster.vq import vq, kmeans, whiten

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
vid_name = experimentName + "_result"

f = h5("mouse_vectors_"+experimentName+".hdf5").load()
f_write = h5("new_mouse_vectors_"+experimentName+".hdf5")
contour_data = f["contour_data"].astype("int32")
n_contours = f["n_contours"]	

names = ["bl", "br", "fl", "fr", "t"]
write_dict = {"bl":np.empty((contour_data.shape[0], 2)), 
				"br":np.empty((contour_data.shape[0], 2)), 
				"fl":np.empty((contour_data.shape[0], 2)), 
				"fr":np.empty((contour_data.shape[0], 2)),
				"t":np.empty((contour_data.shape[0], 2))}


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
output = "Outputs/" + vid_name + "_under_result.avi"
fourcc = cv.VideoWriter_fourcc('M','J','P','G') 
cap = cv.VideoCapture("Outputs/"+vid_name+".avi")

f_bl = (0,0)
f_br = (0,0)
f_fl = (0,0)
f_fr = (0,0)
f_t = (0,0)
fs = [f_bl, f_br, f_fl, f_fr, f_t]
num_limbs = len(write_dict.keys())

directory = "Assets"
videofiles = fm.findFiles(directory, '(\d{6}_\d{2})\D+([@-](\d{4}))?\.tiff?', regex=True)
experiments = fm.movieSorter(videofiles)

def cost(kmean_points, needed_limb_indices):
	limb_count = needed_limb_indices.shape[0]
	distances = np.empty((limb_count, kmean_points.shape[0]))
	for i in range(limb_count):
		limb_x, limb_y = fs[needed_limb_indices[i]]
		for j in range(kmean_points.shape[0]):
			x, y = kmean_points[j]
			dx = limb_x - x
			dy = limb_y - y
			distance = dx*dx + dy*dy
			distances[i, j] = distance

		distances[i] = np.sort(distances[i])

	avg_dist = np.mean(distances[:,0])
	return avg_dist + kmean_points.shape[0] * 200

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
		print("Frame:", mv_i)
		if frame_n >= contour_data.shape[0]:
			#mv_i = 0
			print("Frames completed:", frame_n)
			f_write.save(write_dict)
			break

		t = time.clock()
		ret, im = cap.read()

		for x, y in fs:
			cv.circle(im, (x, y), 2, (255, 0, 0), -1)

		n = n_contours[mv_i]

		if (n > 0):

			c_points = contour_data[mv_i, :n]
			
			limb_distances = np.empty((num_limbs, n))
			for i in range(num_limbs):
				limb_x, limb_y = fs[i]
				for j in range(n):
					x, y = c_points[j]
					dx = limb_x - x
					dy = limb_y - y
					distance = dx*dx + dy*dy
					limb_distances[i, j] = distance

				limb_distances[i] = np.sort(limb_distances[i])

			threshold = 1500
			needed_limbs = np.where(limb_distances[:,0] < threshold)[0]
			
			whitened = whiten(c_points)
			x_scale = c_points[0,0] / whitened[0,0]
			y_scale = c_points[0,1] / whitened[0,1]

			if (needed_limbs.shape[0] > 0):
				max_k = 6
				costs = np.empty(max_k - needed_limbs.shape[0])
				all_kmean_points = []
				for k in range(needed_limbs.shape[0], max_k):
					points, distortion = kmeans(whitened, k)
					points[:,0] *= x_scale
					points[:,1] *= y_scale
					points = points.astype('int32')
					all_kmean_points.append(points)
					costs[k - needed_limbs.shape[0]] = cost(points, needed_limbs)

				best_ind = np.argmin(costs)
				best_points = all_kmean_points[best_ind]

				for i, (x, y) in enumerate(best_points):
					cv.circle(im, (x, y), 2, (0, 0, 255), -1)
				
				distances = np.empty((needed_limbs.shape[0], best_points.shape[0]))
				indices = np.empty((needed_limbs.shape[0], best_points.shape[0], 2), dtype='uint8')
				for i in range(needed_limbs.shape[0]):
					limb_x, limb_y = fs[needed_limbs[i]]
					for j in range(best_points.shape[0]):
						x, y = best_points[j]
						dx = x - limb_x
						dy = y - limb_y
						distance = dx*dx + dy*dy
						distances[i,j] = distance
						indices[i,j,0] = needed_limbs[i]
						indices[i,j,1] = j

				for i in range(needed_limbs.shape[0]):
					i, j = np.unravel_index(np.nanargmin(distances), distances.shape)
					limb_ind = indices[i,j,0]
					point_ind = indices[i,j,1]
					new_limb_pos = (best_points[point_ind,0], best_points[point_ind,1])
					cv.line(im, fs[limb_ind], new_limb_pos, (255, 255, 255), 1)
					fs[limb_ind] = new_limb_pos
					distances[i] = np.NaN
					distances[:,j] = np.NaN


		for i in range(num_limbs):
			name = names[i]
			x, y = fs[i]
			write_dict[name][mv_i,0] = x
			write_dict[name][mv_i,1] = y

		cv.putText(im, str(frame_n), (5,25), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
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
			index += 1
			print("Click tail...")
		elif index == 4:
			fs[index] = (x,y)
			cv.destroyAllWindows()
			limb_track()

			# for path in pathlist:
			# 	mouse_vid = toNumpy(TiffFile(path).pages).astype("float")
			# 	limb_track(mouse_vid)

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