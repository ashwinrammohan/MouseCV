import numpy as np
import cv2 as cv

img = cv.imread("ahh.jpg")
hsvImg = cv.cvtColor(cv.imread("ahh.jpg"), cv.COLOR_BGR2HSV)
foreground = np.zeros_like(img)

for i, row in enumerate(hsvImg):
	for j, pixel in enumerate(row):
		if pixel[0] < 50 or pixel[0] > 70:
			foreground[i][j][0] = 255;
			foreground[i][j][1] = 255;
			foreground[i][j][2] = 255;

img = cv.bitwise_and(img, foreground)
img = cv.inRange(img, (10, 10, 10), (255, 255, 255), cv.THRESH_BINARY_INV);
cv.imshow("Original", img)

cv.waitKey(0)
cv.destroyAllWindows()