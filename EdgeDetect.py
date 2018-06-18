import numpy as np
import cv2 as cv

img = cv.imread("bottom_sc.png")

def edgeSelectImg(img, radius, window_name, callback):
	data = {"mousedown":False, "lastx":0, "lasty":0}
	drawableImg = img.copy()
	cv.imshow(window_name, drawableImg)
	mask = np.zeros_like(img)

	def mouseEvent(event, x, y, flags, param):
		if event == cv.EVENT_LBUTTONDOWN:
			data["mousedown"] = True
			data["lastx"] = x
			data["lasty"] = y
		elif event == cv.EVENT_LBUTTONUP:
			data["mousedown"] = False
			maskedImg = cv.bitwise_and(img, mask)
			edges = cv.Canny(maskedImg, 100, 200)
			cv.imshow("Result", maskedImg)
			cv.imshow("Edges", edges)
			if (callback != None):
				callback(edges)
		elif event == cv.EVENT_MOUSEMOVE:
			if data["mousedown"]:
				cv.line(mask, (data["lastx"],data["lasty"]), (x,y), (255,255,255), radius)
				cv.line(drawableImg, (data["lastx"],data["lasty"]), (x,y), (255,255,255), radius)
				data["lastx"] = x
				data["lasty"] = y
				cv.imshow(window_name, drawableImg)

	cv.setMouseCallback(window_name, mouseEvent)


cv.namedWindow("Original")
edgeSelectImg(img, 10, "Original", None)

cv.waitKey(0)
cv.destroyAllWindows()