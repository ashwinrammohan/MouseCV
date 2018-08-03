from hdf5manager import hdf5manager as h5
import numpy as np
import cv2 as cv
import time
import math

f = h5("mouse_vectors.hdf5").load()
contour_data = f["contour_data"].astype("int32")
n_contours = f["n_contours"]

mv = np.empty((contour_data.shape[0], 480, 640, 3), dtype='uint8')
prev_frame = contour_data[0, :n_contours[0]]

for i in range(contour_data.shape[0]):
	n = n_contours[i]

	for x, y in contour_data[i, :n]:
		min_v = (0, 0)
		min_dist = (1000**2) * 2
		min_pos = (0,0)
		for x2, y2 in prev_frame:
			dx = x2 - x
			dy = y2 - y
			dist = dx*dx + dy*dy
			if (dist < min_dist):
				min_dist = dist
				min_v = (dx, dy)
				min_pos = (x2, y2)

		cv.line(mv[i], (x, y), (x + min_v[0], y + min_v[1]), (0, 255, 0))
		cv.circle(mv[i], min_pos, 3, (0, 0, 255), -1)
		cv.circle(mv[i], (x, y), 3, (255, 0, 0), -1)


	prev_frame = contour_data[i, :n]

print("Playing Movie")
print("Blue means new, Red means old")
cv.namedWindow("Dots")
cv.namedWindow("Movie")
fps = 30
frame_dt = 1.0 / fps
mv_i = 0
pause = False

while True:
	if mv_i >= mv.shape[0]:
		mv_i = 0

	t = time.clock()
	im = mv[mv_i]
	cv.putText(im, str(mv_i), (5,25), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
	cv.imshow("Dots", im)

	if pause:
		k = cv.waitKey(0)
	else:
		dt = frame_dt - (time.clock() - t)
		dt_mili = int(dt * 1000)

		if (dt_mili < 1):
			print(dt)
			dt_mili = 1

		k = cv.waitKey(dt_mili)
		mv_i += 1

	if k == 27: # esc key
		break
	elif k == 32: # space key
		pause = not(pause)
	elif k == 63235 and pause: # right arrow
		mv_i += 1
	elif k == 63234 and pause: # left arrow		
		mv_i -= 1
		
cap.release()
cv.destroyAllWindows()