import numpy as np
import cv2 as cv
import sys
from tifffile import TiffFile

try:
	path_file = open("path.txt", "r")
	sys.path.append(path_file.read())
	path_file.close()
except:
	print("Can't import, path.txt doesn't exist")
	pass

import wholeBrain as wb
import fileManager as fm

import matplotlib.pyplot as plt
import math
from hdf5manager import hdf5manager
import os

vid_name = "180720_05_under"

codec = None
fps = 30
output = "Outputs/" + vid_name + "_shape.avi"
if codec is None:
	if output.endswith('.mp4'):
		if os.name is 'posix':
			codec = 'X264'
		elif os.name is 'nt':
			# codec = 'H264'
			codec = 'XVID'
	else:
		if os.name is 'posix':
			codec = 'MP4V'
		elif os.name is 'nt':
			codec = 'XVID'
		else:
			assert os.name is 'nt', 'Unknown os type: {0}'.format(os.name)	

fourcc = cv.VideoWriter_fourcc('M','J','P','G') 
x1 = 0
x2 = 100
x3 = 100
x4 = 100

directory = "Assets"
videofiles = fm.findFiles(directory, '(\d{6}_\d{2})\D+([@-](\d{4}))?\.tiff?', regex=True)
experiments = fm.movieSorter(videofiles)
print(experiments)
experimentName = "180713_12"

def toNumpy(tiffObject):
	pages = 0
	for frame in tiffObject:
		pages += 1

	ar = np.empty((pages,) + tiffObject[0].shape, dtype="uint8")
	for i, frame in enumerate(tiffObject):
		ar[i] = frame.asarray()

	return ar

pathlist = []
for expname in sorted(experiments):
	if experimentName is not None:
		if expname != experimentName:
			print(expname, 'does not match experiment key:', 
				experimentName +'.  skipping..')
			continue
		else:
			print('found match:', expname)
	# Make output filenames based on name
	pathlist.extend(experiments[expname])
	mouse_vid = toNumpy(TiffFile(pathlist[0]).pages)
	mouse_vid_shape = mouse_vid.shape
	print("Found experiment!!")

resize_factor = 1

print("Loaded movie part 1")
hdf5FilePath = "mouse_vectors.hdf5"
hdf5File = hdf5manager(hdf5FilePath)
mouse_frame = mouse_vid[300]

hdf5Dict = {}
clicks = {"footFL":0, "footFR":0, "footBL":0, "footBR":0, "tail":0, "head":0}
lightest = {"footFL":0, "footFR":0, "footBL":0, "footBR":0, "tail":0, "head":0}
darkest = {"footFL":255, "footFR":255, "footBL":255, "footBR":255, "tail":255, "head":255}
selected = {"footFL":False, "footFR":False, "footBL":False, "footBR":False, "tail":False, "head":False}
pos = {"footFL":(0,0), "footFR":(0,0), "footBL":(0,0), "footBR":(0,0), "tail":(0,0), "head":(0,0)}
currentLabel = ""
crop_area = (0,0,100,100)
stage = 0

verbose = True

def track_limbs():
	global mouse_vid

	try:
		os.remove(hdf5FilePath)
	except FileNotFoundError:
		print("Creating new file with name: " + hdf5FilePath)

	for path in pathlist:
		with TiffFile(path) as tif:
			mouse_vid = toNumpy(tif.pages)
			mouse_vid = mouse_vid[:,x1:x2,y1:y2, 0].copy()

			for limbKey in clicks.keys():
				if selected[limbKey]:
					print("Starting vid processing of limb", limbKey)
					area_track(darkest[limbKey], lightest[limbKey], pos[limbKey][0], pos[limbKey][1], limbKey)

			if resize_factor != 1:
				print("Resizing, factor = " + str(resize_factor))
				width = int(mouse_vid_shape[2] // resize_factor)
				height = int(mouse_vid_shape[1] // resize_factor)
				print("Old size: (" + str(mouse_vid_shape[2]) + ", " + str(mouse_vid_shape[1]) + ")")
				print("New size: " + str((width, height)))

				for i, frame in enumerate(mouse_vid):
					print("Writing frame", i)
					writeFrame(cv.resize(frame, (width, height), interpolation = cv.INTER_AREA), out)

			else:
				i = 0
				for frame in mouse_vid:
					print("Writing frame", i)
					i += 1
					writeFrame(frame, out)

	hdf5File.save(hdf5Dict)
	out.release()
	print("Done!")

def area_track(darkest, lightest, x, y, limb_label):
	retval, result = cv.threshold(mouse_frame, 255, 255, cv.THRESH_TOZERO_INV);#lightest+20, 255, cv.THRESH_TOZERO_INV);
	retval, result = cv.threshold(result, darkest-20, 255, cv.THRESH_BINARY);

	#blah is there cause there's apparently supposed to be 3 outputs
	blah, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	closest_pos = (0,0)

	index = 0
	area = 0
	area_tolerance = 0.3 #% tolerance for varience in area
	bounding_width = 0 #width of the contour bounding box
	width_tolerance = 0.15 #% tolerance for varience in width
	for i, contour in enumerate(contours):
		if cv.pointPolygonTest(contour, (x,y), False) > 0:
			index = i

			m = cv.moments(contour)
			area = m["m00"]
			closest_pos = (m['m10']/m['m00'], m['m01']/m['m00'])

			bounding_rect = cv.minAreaRect(contour)
			bounding_width = min(bounding_rect[1][0], bounding_rect[1][1])
			print("Correct contour found at index " + str(index) + ". Area = " + str(area) + ", Width = " + str(bounding_width))
			break;

	j = 0
	has_contour = False
	curr_centroid = (0,0)
	last_pos = (0,0)
	position_threshold = 40*40

	hdf5Dict[limb_label] = {"x":[], "y":[], "dx":[], "dy":[], "magnitude":[], "angle":[]}
	limb = hdf5Dict[limb_label]

	for frame in mouse_vid:
		retval, result = cv.threshold(frame, lightest+20, 255, cv.THRESH_TOZERO_INV);
		retval, result = cv.threshold(result, darkest-20, 255, cv.THRESH_BINARY);

		#blah is there cause there's apparently supposed to be 3 outputs
		blah, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

		closest_area = 0
		closest_width = 0
		closest_index = 0
		closest_dist = 10000
		close_pos = (0,0)

		for i, contour in enumerate(contours):
			m = cv.moments(contour)
			cont_area = m["m00"]
			if (cont_area == 0):
				continue

			tail_x = m['m10']/m['m00']
			tail_y = m['m01']/m['m00']
			dx = tail_x - closest_pos[0]
			dy = tail_y - closest_pos[1]
			dist = dx*dx + dy*dy

			bounding_rect = cv.minAreaRect(contour)
			cont_width = min(bounding_rect[1][0], bounding_rect[1][1])
			#if dist < closest_dist:
			if abs(cont_area - area) < abs(closest_area - area) and dist < position_threshold: #(cont_width - bounding_width) / bounding_width < width_tolerance and
				closest_index = i
				closest_area = cont_area
				closest_width = cont_width
				closest_dist = dist
				close_pos = (tail_x, tail_y)
		
		percent = abs(closest_area - area) / area
		#if closest_dist > position_threshold:
		if percent > area_tolerance:
			motion = (closest_pos[0] - curr_centroid[0], closest_pos[1] - curr_centroid[1])

			limb["x"].append(last_pos[0])
			limb["y"].append(last_pos[1])
			limb["dx"].append(0)
			limb["dy"].append(0)
			limb["magnitude"].append(0)
			limb["angle"].append(0)

			if has_contour:
				print("Contour lost!")
			has_contour = False
		else:
			closest_pos = close_pos
			if j != 0:
				motion = (closest_pos[0] - curr_centroid[0], closest_pos[1] - curr_centroid[1])

				limb["x"].append(curr_centroid[0])
				limb["y"].append(curr_centroid[1])
				limb["dx"].append(motion[0])
				limb["dy"].append(motion[1])
				limb["magnitude"].append(math.sqrt(motion[0] * motion[0] + motion[1] * motion[1]))
				limb["angle"].append(math.atan2(motion[1], motion[0]))
				last_pos = curr_centroid

			if not(has_contour):
				has_contour = True
				print("Found contour!")
				area = closest_area
				bounding_width = closest_width

			cv.drawContours(frame, contours, closest_index, (0,0,0), 2)
		
		curr_centroid = closest_pos

		if verbose:
			print("Processing frame #" + str(j))
			j += 1
			print("Closest contour found at index " + str(closest_index) + ". Area = " + str(area) + ", Width = " + str(bounding_width))


def writeFrame(frame, out, cmap = None, fmin = 0, fmax = 255):
		
	# rescale and convert to uint8
	fslope = 255 / fmax
	frame = frame - fmin
	frame = frame * fslope

	# cap min and max to prevent incorrect cmap application
	frame[frame > 255] = 255
	frame[frame < 0] = 0

	frame = frame.astype('uint8')

	# apply colormap, write frame to .avi
	if cmap is not None: 
		frame = cv.applyColorMap(frame.astype('uint8'), cmap)
	elif (len(frame.shape) == 2):
		frame = np.repeat(frame[:,:,None], 3, axis=2)

	out.write(frame)

	
out = None
def limb_click(event,x,y,flags,param):
	global currentLabel, stage, crop_area, mouse_vid, mouse_frame, x1, x2, y1, y2, out, mouse_vid_shape

	if event == cv.EVENT_MOUSEMOVE and stage == 1:
		new_frame = np.copy(mouse_vid[300])
		cv.rectangle(new_frame, crop_area[:2], (x,y), (255, 255, 255), 2)
		cv.imshow("Original", new_frame)

	elif event == cv.EVENT_LBUTTONUP:
		if stage == 0:
			crop_area = (x, y, 100, 100)
			print("Now click to define lower right corner of region of interest")
			stage += 1
			return
		elif stage == 1:
			crop_area = crop_area[:2] + (x, y)
			stage += 1

			x1 = crop_area[1]
			y1 = crop_area[0]
			x2 = crop_area[3]
			y2 = crop_area[2]
			mouse_vid = mouse_vid[:,x1:x2,y1:y2, 0].copy()
			mouse_vid_shape = mouse_vid.shape

			w = int(mouse_vid_shape[1] // resize_factor)
			h = int(mouse_vid_shape[2] // resize_factor)
			print(w,h)
			out = cv.VideoWriter(output, fourcc, fps, (h,w), isColor=True)

			mouse_frame = mouse_vid[300]
			cv.imshow("Original", mouse_frame)
			cv.waitKey(10)
			printLabels()
			return

		if currentLabel == "":
			return

		if lightest[currentLabel] < mouse_frame.item(y,x):
			lightest[currentLabel] = mouse_frame.item(y,x)
		if darkest[currentLabel] > mouse_frame.item(y,x):
			darkest[currentLabel] = mouse_frame.item(y,x)

		clicks[currentLabel] += 1
		print(currentLabel + " click #" + str(clicks[currentLabel]))
		if clicks[currentLabel] == 6:
			print(currentLabel + " color calibrated, tracking")
			pos[currentLabel] = (x, y)
			selected[currentLabel] = True
			currentLabel = ""
			printLabels()
			

def printLabels():
	global currentLabel
	print("Chose a limb to track (or -1 if done): ")
	keys = list(selected.keys())
	for i, limbKey in enumerate(keys):
		if not(selected[limbKey]):
			print(str(i) + ": " + limbKey)

	index = int(input(">>> "))
	if index == -1:
		track_limbs()
	elif index < len(keys):
		limbKey = keys[index]
		if selected[limbKey]:
			print("You've already tracked that limb.")
			printLabels()
			return
		else:
			currentLabel = limbKey

cv.namedWindow("Original")
cv.setMouseCallback("Original", limb_click)
cv.imshow("Original", mouse_frame)

cv.waitKey(10)
print("Click upper left corner of region of interest")

cv.waitKey(0)
cv.destroyAllWindows()

'''

def foot_track(darkest, lightest, x, y, file_name):
	return

	print(darkest, lightest)
	retval, result = cv.threshold(mouse_frame, lightest+20, 255, cv.THRESH_TOZERO_INV);
	retval, result = cv.threshold(result, darkest-20, 255, cv.THRESH_BINARY);

	#blah is there cause there's apparently supposed to be 3 outputs
	blah, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	index = 0
	min_area = 0
	max_area = 0
	foot_pos = (0,0)
	position_tolerance = 80*80 #tolerance for change in position of foot (squared)
	
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
	curr_centroid = (0,0)
	#f = open(file_name + ".csv","w+")
	hdf5Dict = {"foot1":{"x":[], "y":[], "dx":[], "dy":[], "magnitude":[], "angle":[]}}
	for frame in mouse_vid:
		retval, result = cv.threshold(frame, lightest+20, 255, cv.THRESH_TOZERO_INV);
		retval, result = cv.threshold(result, darkest-20, 255, cv.THRESH_BINARY);

		#blah is there cause there's apparently supposed to be 3 outputs
		blah, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

		closest_index = 0
		closest_dist = 100000
		closest_pos = (0, 0)

		for i, contour in enumerate(contours):
			m = cv.moments(contour)
			area = m["m00"]

			if area == 0:
				continue

			foot_x = m['m10']/m['m00']
			foot_y = m['m01']/m['m00']
			dx = int(foot_x) - foot_pos[0]
			dy = int(foot_y) - foot_pos[1]
			dist = dx*dx + dy*dy

			if area > min_area and area < max_area and dist < closest_dist:
				closest_dist = dist
				closest_index = i
				closest_pos = (foot_x, foot_y)

		if closest_dist > position_tolerance:
			if has_contour:
				print("Contour lost!")
			has_contour = False
		else:
			if j != 0:
				motion = (closest_pos[0] - curr_centroid[0], closest_pos[1] - curr_centroid[1])

				hdf5Dict["foot_x"].append(motion[0])
				hdf5Dict["foot_y"].append(motion[1])
				hdf5Dict["foot_magnitude"].append(math.sqrt(motion[0] * motion[0] + motion[1] * motion[1]))
				hdf5Dict["foot_angle"].append(math.atan2(motion[1], motion[0]))

				#f.write(str(j) + "," + str(motion[0]) + "," + str(motion[1]) + "," + str(math.sqrt(motion[0] * motion[0] + motion[1] * motion[1])) + "\n")
				#print("Frame " + str(j) + ": foot moved " + str(motion))

			cv.drawContours(frame, contours, closest_index, (0,0,0), 2)
			if not(has_contour):
				has_contour = True
				#foot_pos = (int(closest_pos[0]), int(closest_pos[1]))
				print("Found contour!")

		curr_centroid = (closest_pos[0], closest_pos[1])
		
		if verbose:
			print("Processing frame #" + str(j))
			#print("Searching " + str(len(contours)) + " contours...")
			j += 1
			#print("Correct contour found at index " + str(index) + ". Distance = " + str(closest_dist))
	#f.close()
	hdf5File.save(hdf5Dict)

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
			tail_track(tail_darkest, tail_lightest, tail_pos[0], tail_pos[1])#, vid_name+"_tail")
			foot_track(foot_darkest, foot_lightest, foot_pos[0], foot_pos[1], vid_name+"_foot")
			wb.playMovie(mouse_vid, cmap=cv.COLORMAP_BONE)
'''
