import numpy as np
import cv2 as cv
import wholeBrain as wb
import matplotlib.pyplot as plt

clicks = 0
lightest = 0
darkest = 255

mouse_vid = cv.VideoCapture("mouse_vid.tif")

def mouse_click(event,x,y,flags,param):
	global lightest, darkest, clicks
	if event == cv.EVENT_LBUTTONUP:
		print(img.item(y,x))
		if lightest < img.item(y,x):
			lightest = img.item(y,x)
		if darkest > img.item(y,x):
			darkest = img.item(y,x)

		clicks += 1
		if clicks >= 6:
			print(darkest, lightest)
			retval, result = cv.threshold(img, lightest+10, 255, cv.THRESH_TOZERO_INV);
			retval, result = cv.threshold(result, darkest-10, 255, cv.THRESH_BINARY);

			#blah is there cause there's apparently supposed to be 3 outputs

			blah, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

			index = 0
			for i, contour in enumerate(contours):
				if cv.pointPolygonTest(contour, (x,y), False) > 0:
					print("found it!")
					index = i
					break;

			cv.drawContours(contors_img, contours, index, (255,255,255), 2)
			cv.imshow("Contours", contors_img)

img = cv.imread("realmouse.png", 0)
contors_img = img.copy()

cv.namedWindow("Original")
cv.setMouseCallback("Original", mouse_click)

#wb.playMovie(img, )

cv.imshow("Original", img)
cv.waitKey(0)
cv.destroyAllWindows()