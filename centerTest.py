from hdf5manager import hdf5manager as h5
import cv2 as cv2
import numpy as np

domain_map = h5("P2_timecourses.hdf5").load()["domainmap"]

def interpAxis(contour, interp_x):
	contour = contour[:,0]

	if interp_x:
		xs = contour[:,0]
		ys = contour[:,1]
	else:
		xs = contour[:,1]
		ys = contour[:,0]

	xs_interp = xs.tolist()
	ys_interp = ys.tolist()

	sz = xs.shape[0]
	i = 1
	while i < sz:
		dif = xs_interp[i] - xs_interp[i-1]
		new_x = 0

		if dif < -1:
			new_x = xs_interp[i-1] - 1
		elif dif > 1:
			new_x = xs_interp[i-1] + 1
		else:
			i += 1
			continue

		t_n = (new_x - xs_interp[i-1]) / dif
		new_y = (ys_interp[i] - ys_interp[i-1]) * t_n + ys_interp[i-1]

		xs_interp.insert(i, new_x)
		ys_interp.insert(i, new_y)

		sz += 1
		i += 1

	if interp_x:
		xs_interp = np.asarray(xs_interp, dtype="int32")
		ys_interp = np.asarray(ys_interp, dtype="int32")
	else:
		xs_interp_temp = np.asarray(ys_interp, dtype="int32")
		ys_interp = np.asarray(xs_interp, dtype="int32")
		xs_interp = xs_interp_temp

	return xs_interp, ys_interp

def centerOnContour(contour):
	xs, ys = interpAxis(contour, True)
	order = np.argsort(xs)
	xs = xs[order]
	ys = ys[order]

	dif_x = xs[1:] - xs[:-1]
	dif_y = ys[1:] - ys[:-1]
	dif_y[dif_x > 0] = 0
	dif_y = np.abs(dif_y)

	maxX = xs[np.argmax(dif_y)]
	maxYs = ys[xs == maxX]
	maxY = np.max(maxYs)
	minY = np.min(maxYs)

	xs, ys = interpAxis(contour, False)
	xs = xs[(ys >= minY) * (ys <= maxY)]
	ys = ys[(ys >= minY) * (ys <= maxY)]
	order = np.argsort(ys)
	xs = xs[order]
	ys = ys[order]

	dif_x = xs[1:] - xs[:-1]
	dif_y = ys[1:] - ys[:-1]
	dif_x[dif_y > 0] = 0
	dif_x = np.abs(dif_x)
	maxY = ys[np.argmax(dif_x)]

	return maxX, maxY
	

'''for i in range(domain_map.shape[0]):
	img = np.zeros_like(overlapped_domain_map, dtype="uint8")
	img[overlapped_domain_map == i+1] = 255

	#blah is there cause there's apparently supposed to be 3 outputs
	blah, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	for i, contour in enumerate(contours):
		contourImage = np.zeros_like(img)
		cont_area = cv.contourArea(contour)
		if (cont_area > 100):
			cent_x, cent_y = centerOnContour(contour)'''


