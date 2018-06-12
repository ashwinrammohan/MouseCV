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
			area = 0
			area_tolerance = 0.3 #% tolerance for varience in area
			bounding_width = 0 #width of the contour bounding box
			width_tolerance = 0.15 #% tolerance for varience in width
			for i, contour in enumerate(contours):
				if cv.pointPolygonTest(contour, (x,y), False) > 0:
					index = i
					area = cv.contourArea(contour)
					bounding_rect = cv.minAreaRect(contour)
					bounding_width = min(bounding_rect[1][0], bounding_rect[1][1])
					print("Correct contour found at index " + str(index) + ". Area = " + str(area) + ", Width = " + str(bounding_width))
					break;

			j = 0
			has_contour = False
			for frame in mouse_vid:
				retval, result = cv.threshold(frame, lightest+10, 255, cv.THRESH_TOZERO_INV);
				retval, result = cv.threshold(result, darkest-10, 255, cv.THRESH_BINARY);

				#blah is there cause there's apparently supposed to be 3 outputs
				blah, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

				closest_area = 0
				closest_width = 0
				closest_index = 0
				for i, contour in enumerate(contours):
					cont_area = cv.contourArea(contour)
					bounding_rect = cv.minAreaRect(contour)
					cont_width = min(bounding_rect[1][0], bounding_rect[1][1])
					if abs(cont_area - area) < abs(closest_area - area) and (cont_width - bounding_width) / bounding_width < width_tolerance:
						closest_index = i
						closest_area = cont_area
						closest_width = cont_width

				percent = abs(closest_area - area) / area
				if percent > area_tolerance:
					if has_contour:
						print("Contour lost!")
					has_contour = False
				else:
					if not(has_contour):
						has_contour = True
						print("Found contour!")
						area = closest_area
						bounding_width = closest_width
					cv.drawContours(frame, contours, closest_index, (175,175,175), 2)
				
				if verbose:
					print("Processing frame #" + str(j))
					#print("Searching " + str(len(contours)) + " contours...")
					j += 1
					print("Closest contour found at index " + str(closest_index) + ". Area = " + str(area) + ", Width = " + str(bounding_width))


			
			wb.playMovie(mouse_vid, cmap=cv.COLORMAP_BONE)
			

cv.namedWindow("Original")
cv.setMouseCallback("Original", mouse_click)
cv.imshow("Original", mouse_frame)

cv.waitKey(0)
cv.destroyAllWindows()