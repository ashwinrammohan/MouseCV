import numpy as np
import cv2 as cv
import wholeBrain as wb
import matplotlib.pyplot as plt

clicks = 0
lightest = 0
darkest = 255
verbose = True

mouse_vid = wb.loadMovie("mouse_vid.tif").astype("uint8")
mouse_frame = mouse_vid[0]

def mouse_click(event,x,y,flags,param):
	global lightest, darkest, clicks
	if event == cv.EVENT_LBUTTONUP:
		if lightest < mouse_frame.item(y,x):
			lightest = mouse_frame.item(y,x)
		if darkest > mouse_frame.item(y,x):
			darkest = mouse_frame.item(y,x)

		clicks += 1
		if clicks == 6:
			print(darkest, lightest)

			retval, result = cv.threshold(mouse_frame, lightest+10, 255, cv.THRESH_TOZERO_INV);
			retval, result = cv.threshold(result, darkest-10, 255, cv.THRESH_BINARY);

			#blah is there cause there's apparently supposed to be 3 outputs
			blah, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

			index = 0
			for i, contour in enumerate(contours):
				if cv.pointPolygonTest(contour, (x,y), False) > 0:
					index = i
					M = cv.moments(contour)
					contour_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
					print("Correct contour found at index ", index, ". Position: ", contour_center[0], ", ", contour_center[1])
					break;

			j = 0
			for frame in mouse_vid:
				retval, result = cv.threshold(frame, lightest+10, 255, cv.THRESH_TOZERO_INV);
				retval, result = cv.threshold(result, darkest-10, 255, cv.THRESH_BINARY);

				#blah is there cause there's apparently supposed to be 3 outputs
				blah, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

				min_dist = 1000000
				closest_index = 0
				new_c_x = 0
				new_c_y = 0
				for i, contour in enumerate(contours):
					M = cv.moments(contour)

					if M["m00"] == 0:
						continue

					dx = int(M["m10"] / M["m00"]) - contour_center[0]
					dy = int(M["m01"] / M["m00"]) - contour_center[1]
					dist = dx*dx + dy*dy
					if dist < min_dist:
						closest_index = i
						min_dist = dist
						new_c_x = int(M["m10"] / M["m00"])
						new_c_y = int(M["m01"] / M["m00"])

				contour_center = (new_c_x, new_c_y)
				if verbose:
					print("Processing frame #", j)
					print("Searching ", len(contours), " contours...")
					j += 1
					print("Closest contour found at index", closest_index, ". Position: ", contour_center[0], ", ", contour_center[1])

				cv.drawContours(frame, contours, closest_index, (255,255,255), 2)

			j = 0
			for frame in mouse_vid[:40]:
				cv.imshow(str(j), frame)
				j += 1
			#wb.playMovie(mouse_vid, cmap=cv.COLORMAP_BONE)
			

cv.namedWindow("Original")
cv.setMouseCallback("Original", mouse_click)
cv.imshow("Original", mouse_frame)

cv.waitKey(0)
cv.destroyAllWindows()