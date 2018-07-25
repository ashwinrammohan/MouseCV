import cv2 as cv
import numpy as np
from hdf5manager import hdf5manager as h5
import matplotlib.cm as cm

pMatrix = h5("Outputs/171018_03_MatrixData_full_interp.hdf5").load()['pMatrix']
cv.namedWindow('image')

def callback(x):
	cutoff = x / 1000
	print("Cutoff:", cutoff)

	img = np.zeros((pMatrix.shape[0], pMatrix.shape[1], 3), dtype="uint8")
	mult = 2

	# preMatrix = np.zeros_like(pMatrix, dtype=bool)
	# preMatrix[pMatrix < cutoff] = True
	# preMatrix[pMatrix > 1 - cutoff] = True

	# fullImg = np.empty((preMatrix.shape[0] * 2, preMatrix.shape[1] * 3), dtype='uint8')
	# btmX = 0
	# btmY = 0
	# topY = preMatrix.shape[0]
	# for i in range(1, 4):
	# 	topX = btmX + preMatrix.shape[1]
	# 	fullImg[btmY:topY, btmX:topX] = preMatrix[:,:,i]
	# 	btmX = topX

	# btmY = topY
	# topY = btmY + preMatrix.shape[0]
	# btmX = 0

	# for i in range(4, 7):
	# 	topX = btmX + preMatrix.shape[1]
	# 	fullImg[btmY:topY, btmX:topX] = preMatrix[:,:,i]
	# 	btmX = topX

	# cv.imshow('image', fullImg)

	for i in range(6,0,-1):
		cmClr = cm.jet(i / 6)
		clr = np.empty(3)

		clr[0] = cmClr[2] * 255
		clr[1] = cmClr[1] * 255
		clr[2] = cmClr[0] * 255

		img[pMatrix[:,:,i] > 1 - cutoff] = clr

	cv.imshow('image', cv.resize(img, (pMatrix.shape[0]*mult, pMatrix.shape[1]*mult), interpolation=cv.INTER_NEAREST))

# create trackbars for color change
cv.createTrackbar('Cutoff','image', 0, 100, callback)

cv.waitKey(0)
cv.destroyAllWindows()