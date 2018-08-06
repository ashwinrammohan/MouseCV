from hdf5manager import hdf5manager as h5
import numpy as np
import cv2 as cv
import time
import math

f = h5("new_mouse_vectors.hdf5").load()
f_bl = f["bl"].astype("int64")
f_br = f["br"].astype("int64")
f_fl = f["fl"].astype("int64")
f_fr = f["fr"].astype("int64")
feet = [f_bl, f_br, f_fl, f_fr]

mv = np.empty((f_bl.shape[0], 480, 640, 3), dtype='uint8')

for f in feet:
	for i in range(f.shape[0]):
		x, y = f[i]
		cv.circle(mv[i], (x, y), 3, (255, 0, 0), -1)

print("Playing Movie")
print("Blue means new, Red means old")
cv.namedWindow("Dots")
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