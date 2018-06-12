import numpy as np
import cv2 as cv
import wholeBrain as wb
import matplotlib.pyplot as plt

foot_clicks = 0
tail_clicks = 0
tail_lightest = 0
tail_darkest = 255
foot_lightest = 0
foot_darkest = 255
foot_pos = (0,0)
tail_pos = (0,0)
verbose = True

mouse_vid = wb.loadMovie("mouse_vid.tif").astype("uint8")
mouse_frame = mouse_vid[0]

def foot_track(darkest, lightest, x, y):
	print(darkest, lightest)
	retval, result = cv.threshold(mouse_frame, lightest+10, 255, cv.THRESH_TOZERO_INV);
	retval, result = cv.threshold(result, darkest-10, 255, cv.THRESH_BINARY);

	#blah is there cause there's apparently supposed to be 3 outputs
	blah, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	index = 0
	min_area = 0
	max_area = 0
	foot_pos = (0,0)
	position_tolerance = 30*30 #tolerance for change in position of foot (squared)
	for i, contour in enumerate(contours):
		if cv.pointPolygonTest(contour, (x,y), False) > 0:
			index = i
			m = cv.moments(contour)
			min_area = m["m00"] * 0.25
			max_area = m["m00"] * 4
			foot_pos = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
			print("Correct contour found at index " + str(index) + ". Area = " + str(m['m00']) + ", Position = (" + str(foot_pos[0]) + ", " + str(foot_pos[1]) + ")")
			break;

	j = 0
	has_contour = False
	for frame in mouse_vid:
		retval, result = cv.threshold(frame, lightest+10, 255, cv.THRESH_TOZERO_INV);
		retval, result = cv.threshold(result, darkest-10, 255, cv.THRESH_BINARY);

		#blah is there cause there's apparently supposed to be 3 outputs
		blah, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

		closest_index = 0
		closest_dist = 100000
		curr_centroid = (0,0)
		new_centroid = (0,0)
		motion = new_centroid - curr_centroid

		for i, contour in enumerate(contours):
			m = cv.moments(contour)
			area = m["m00"]

			if area == 0:
				continue

			foot_x = int(m['m10']/m['m00'])
			foot_y = int(m['m01']/m['m00'])
			dx = foot_x - foot_pos[0]
			dy = foot_y - foot_pos[1]
			dist = dx*dx + dy*dy

			if area > min_area and area < max_area and dist < closest_dist:
				closest_dist = dist
				closest_index = i
				closest_pos = (foot_x, foot_y)

		closest_moment = cv.moments(contours[closest_index])
		new_centroid = (int(closest_moment['m10']/closest_moment['m00']),int(closest_moment['m01']/closest_moment['m00']))
		if j != 0:
			motion = new_centroid - curr_centroid
			print("Frame " + j + ": foot moved " + motion)
		curr_centroid = new_centroid

		if closest_dist > position_tolerance:
			if has_contour:
				print("Contour lost!")
			has_contour = False
		else:
			cv.drawContours(frame, contours, closest_index, (175,175,175), 2)
			if not(has_contour):
				has_contour = True
				foot_pos = (closest_pos[0], closest_pos[1])
				print("Found contour!")

		
		if verbose:
			print("Processing frame #" + str(j))
			#print("Searching " + str(len(contours)) + " contours...")
			j += 1
			print("Correct contour found at index " + str(index) + ". Distance = " + str(closest_dist))


def foot_click(event,x,y,flags,param):
	global foot_lightest, foot_darkest, foot_clicks, foot_pos
	if event == cv.EVENT_LBUTTONUP:
		if foot_lightest < mouse_frame.item(y,x):
			foot_lightest = mouse_frame.item(y,x)
		if foot_darkest > mouse_frame.item(y,x):
			foot_darkest = mouse_frame.item(y,x)

		foot_clicks += 1
		if foot_clicks == 6:
			print("Foot color calibrated, playing movie")
			foot_pos = (x, y)
			foot_track(tail_darkest, tail_lightest, tail_pos[0], tail_pos[1])
			foot_track(foot_darkest, foot_lightest, foot_pos[0], foot_pos[1])
			wb.playMovie(mouse_vid, cmap=cv.COLORMAP_BONE)

def tail_track(darkest, lightest, x, y):
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
		curr_centroid = (0,0)
		new_centroid = (0,0)

		for i, contour in enumerate(contours):
			cont_area = cv.contourArea(contour)
			bounding_rect = cv.minAreaRect(contour)
			cont_width = min(bounding_rect[1][0], bounding_rect[1][1])
			if abs(cont_area - area) < abs(closest_area - area) and (cont_width - bounding_width) / bounding_width < width_tolerance:
				closest_index = i
				closest_area = cont_area
				closest_width = cont_width

		closest_moment = cv.moments(contours[closest_index])
		new_centroid = (int(closest_moment['m10']/closest_moment['m00']),int(closest_moment['m01']/closest_moment['m00']))
		if j != 0:
			motion = new_centroid - curr_centroid
			print("Frame " + j + ": tail moved " + motion)
		curr_centroid = new_centroid
		
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
	
def tail_click(event,x,y,flags,param):
	global tail_lightest, tail_darkest, tail_clicks, tail_pos
	if event == cv.EVENT_LBUTTONUP:
		if tail_lightest < mouse_frame.item(y,x):
			tail_lightest = mouse_frame.item(y,x)
		if tail_darkest > mouse_frame.item(y,x):
			tail_darkest = mouse_frame.item(y,x)

		tail_clicks += 1
		if tail_clicks == 6:
			print("Tail color calibrated, now click foot")
			tail_pos = (x, y)
			cv.setMouseCallback("Original", foot_click)
			

print("Click tail to gather color data")

cv.namedWindow("Original")
cv.setMouseCallback("Original", tail_click)
cv.imshow("Original", mouse_frame)

cv.waitKey(0)
cv.destroyAllWindows()