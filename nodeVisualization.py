import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib import colors
from hdf5manager import hdf5manager as h5
# import wholeBrain as wb
import cv2 as cv
import time

eventMatrix = None
pMatrix = None
domain_map = None

ratio = 10000000
pValue_thresh = 0.00001
index = 0
whole_brain_map = None
trigger = False

windows = None

if __name__ == '__main__':
	print("Reached main")
	global eventMatrix, pMatrix, domain_map, trigger

	import argparse

	ap = argparse.ArgumentParser()

	ap.add_argument('-c', '--coincidence_filename', type = str, nargs = 1, required = True, help = 'name of hdf5 input file with event coincidence data')
	ap.add_argument('-f', '--data_filename', nargs = 1, required = True, help = 'name of hdf5 input file with ICA-filtered timecourses and domain map')
	ap.add_argument('-t', '--trigger', action = "store_true", required = False, help = 'generate the lookup table')

	args = vars(ap.parse_args())

	f = h5(args['coincidence_filename'][0])
	data = f.load()
	eventMatrix = data['eventMatrix']
	pMatrix = data['pMatrix']

	f2 = h5(args['data_filename'][0])
	data2 = f2.load()
	domain_map = data2['domainmap']

	if (args['trigger']):
		trigger = True

title = "Trigger" if trigger else "Precursor"

overlapped_domain_map = np.empty((domain_map.shape[1], domain_map.shape[2]), dtype="uint8")
for i, region in enumerate(domain_map):
	wr = np.where(region == 1)
	overlapped_domain_map[wr[0], wr[1]] = i+1

'''
Calculates and returns the min window matrix, which contains the first time window at which event coincidence is
"significant" for each timecourse-timecourse coincidence. To obtain the min windows, this method iterates over the 
pMatrix from the hdf5 coincidence file...the windows at which the p value is greater than some threshold (defined by the
global variable 'pValue_thresh') are the "significant" windows, and the first of these windows is the min window.
'''

maxRates = np.ones(eventMatrix.shape[0])
minRates = np.zeros(eventMatrix.shape[0])

def minWindows(fps = 10):
	global windows

	wins = np.zeros((eventMatrix.shape[0], eventMatrix.shape[1], 3))

	for i, results in enumerate(pMatrix):
		min_rate = 1
		max_rate = 0

		for j, region_pvals in enumerate(results):
			if (i != j and not(np.any(np.isnan(eventMatrix[i, j])))):
				inds = np.where(region_pvals > 1 - pValue_thresh)

				if (inds[0].shape[0] > 0):
					minWindow = (inds[0][0] + 1) * (1/fps)

					if minWindow < 2:
						wins[i,j,0] = minWindow/2
						wins[i,j,1] = 1
						wins[i,j,2] = 1
						wins[i,j] = colors.hsv_to_rgb(wins[i,j]) * 255
						min_rate = min(min_rate, eventMatrix[i,j,inds[0][0]])
						max_rate = max(max_rate, eventMatrix[i,j,inds[0][0]])
				else:
					wins[i][j] = 150
			else:
				if (i == j):
					wins[i][j] = 255
				else:
					wins[i][j] = 0


		maxRates[i] = max_rate
		minRates[i] = min_rate

	fullWindows = np.zeros((eventMatrix.shape[0]+1, eventMatrix.shape[1]+1, 3))
	fullWindows[:,0] = np.array([0,0,0])
	fullWindows[0,:] = np.array([0,0,0])
	fullWindows[1:,1:] = wins

	windows = fullWindows.astype('uint8')

'''
For a given node number (an index corresponding to a certain domain), this method applies
the corresponding min window map to the global time_map. In the center of each domain in the
brain, a small white circle is drawn so that the user can better understand where the domains
are located.
'''
def getTimeMap(node_number):
	global time_map

	if trigger:
		minWindow = windows[:,node_number]
	else:
		minWindow = windows[node_number]

	time_map[...] = minWindow[overlapped_domain_map]
	this_centerX = centerXs[node_number-1]
	this_centerY = centerYs[node_number-1]

	index1 = node_number - 1
	min_rate = 1
	max_rate = 0

	for x, y, r in zip(centerXs, centerYs, radii):
		index2 = findRegion(y, x) - 1
		coinRate = 0

		if trigger and not(np.any(np.isnan(eventMatrix[index2, index1]))):
			win_t = np.where(pMatrix[index2, index1] > 1 - pValue_thresh)[0]
			if (win_t.shape[0] > 0):
				minRate = minRates[index2, index1]
				maxRate = maxRates[index2, index1]

				coinRate = (eventMatrix[index2,index1,win_t[0]] - minRate) / (maxRate - minRate)

				color = cm.jet(coinRate)
				color = (color[0] * 255, color[1] * 255, color[2] * 255)
				cv.circle(time_map, (y,x), 4, color, -1)

		elif not(trigger) and not(np.any(np.isnan(eventMatrix[index1, index2]))):
			win_t = np.where(pMatrix[index1, index2] > 1 - pValue_thresh)[0]

			if (win_t.shape[0] > 0):
				minRate = minRates[index1]
				maxRate = maxRates[index1]

				coinRate = (eventMatrix[index1,index2,win_t[0]] - minRate) / (maxRate - minRate)

				color = cm.jet(coinRate)
				color = (color[0] * 255, color[1] * 255, color[2] * 255)
				cv.circle(time_map, (y,x), 4, color, -1)

	# print(min_rate, max_rate)

	# for x, y, r in zip(centerXs, centerYs, radii):
	# 	cv.circle(time_map, (y,x), 2, (255,255,255), -1)

	# 	index2 = findRegion(y, x) - 1
		
	# 	if trigger and not(np.any(np.isnan(eventMatrix[index2, index1]))):
	# 		win_t = np.where(pMatrix[index2, index1] > 1 - pValue_thresh)[0]
	# 		if (win_t.shape[0] > 0):
	# 			coinRate = int(eventMatrix[index2, index1, win_t[0]])
	# 			coinRate -= min_rate
	# 			coinRate *= big_size / max_rate
	# 			cv.line(time_map, (this_centerY, this_centerX), (y,x), (255,255,255), thickness = int(coinRate)+1)

	# 	elif not(trigger) and not(np.any(np.isnan(eventMatrix[index1, index2]))):
	# 		win_t = np.where(pMatrix[index1, index2] > 1 - pValue_thresh)[0]
	# 		if (win_t.shape[0] > 0):
	# 			coinRate = eventMatrix[index1, index2, win_t[0]] 
	# 			coinRate -= min_rate
	# 			coinRate *= big_size / max_rate
	# 			cv.line(time_map, (this_centerY, this_centerX), (y,x), (255,255,255), thickness = int(coinRate)+1)

'''
Given a certain (x,y) location on the image, this method determines which domain covers that point.
'''
def findRegion(x, y):
	return overlapped_domain_map[y,x]

'''
This method controls what happens when a certain area is clicked on the image.

If the user left-clicks on a region, the corresponding domain at that location is found. Then,
that region's time map is retrieved and is applied to the image.

If a region has already been selected and the user right clicks on a different region, then
the coincidence rate of the first region with the second region at the min window is printed. This coincidence
rate will be for when the left-clicked region is precursor or trigger depending on the global 'trigger' boolean.
'''
def region_click(event, x, y, flags, param):
	global time_map, color_map, index

	if event == cv.EVENT_LBUTTONUP:
		index = findRegion(x,y)
		img = np.zeros_like(overlapped_domain_map)
		img[overlapped_domain_map == index] = 255

		for x, y, r in zip(centerXs, centerYs, radii):
			if findRegion(y,x) == index:
				cv.circle(img, (y,x), 3, (150,150,150), -1)

		cv.imshow("Region", img)
		cv.waitKey(0)
		cv.destroyWindow("Region")

		print("Displaying index", index)
		getTimeMap(index)
		cv.imshow("Time Map " + title, time_map)
	elif event == cv.EVENT_RBUTTONUP:
		index2 = findRegion(x,y) - 1
		index1 = index-1
		if trigger:
			win_t = np.where(pMatrix[index2, index1] > 1 - pValue_thresh)[0]
			if win_t.shape[0] > 0:
				print("Coincidence rate of", index1, "against", index2, "=", eventMatrix[index2, index1, win_t[0]])
		else:
			win_t = np.where(pMatrix[index1, index2] > 1 - pValue_thresh)[0]
			if win_t.shape[0] > 0:
				print("Coincidence rate of", index1, "against", index2, "=", eventMatrix[index1, index2, win_t[0]])

'''
When the value on the cv trackbar is changed, this method recalculates the pValue_thresh
and then calls minWindows() to get the new min window matrix. The time map for the region
that was clicked previously is updated and the image is refreshed.
'''
def callback(x):
	global pValue_thresh, time_map, windows

	pValue_thresh = x/ratio
	minWindows()

	getTimeMap(index)
	cv.imshow("Time Map " + title, time_map)

'''
This method finds the centers for each of the domains on the domain_map and the average radii
for circles that encompass each domain. 
'''

def findCentersAndRadii():
	centerXs = np.zeros(domain_map.shape[0], dtype = "uint32")
	centerYs = np.zeros(domain_map.shape[0], dtype = "uint32")
	radii = np.zeros(domain_map.shape[0], dtype = "uint32")

	for i in range(domain_map.shape[0]):
		res = np.where(overlapped_domain_map == i+1)
		if (len(res[0]) >= 1):
			centerXs[i] = np.mean(res[0])
			centerYs[i] = np.mean(res[1])

			xDists = np.square(res[0] - centerXs[i])
			yDists = np.square(res[1] - centerYs[i])
			radii[i] = np.mean(np.sqrt(xDists + yDists))

	return centerXs, centerYs, radii

'''
This method draws the circles around each domain using the coordinates
for the circles and the radii from the previous method.
'''

def drawNodes(centerXs, centerYs, radii):
	img = np.zeros((domain_map.shape[1], domain_map.shape[2]))
	for x, y, r in zip(centerXs, centerYs, radii):
		cv.circle(img, (y,x), r, (255,255,255), 2)
		cv.circle(img, (y,x), 3, (255,255,255), -1)

	cv.imshow("Node Map", img)

def findIndexCallback(event, x, y, flags, param):
	if event == cv.EVENT_LBUTTONUP:
		findIndex(centerXs, centerYs, radii, x, y)

def findIndex(centerXs, centerYs, radii, mX, mY):
	for i, (x, y, r) in enumerate(zip(centerXs, centerYs, radii)):
		dx = (mX - y) ** 2
		dy = (mY - x) ** 2
		if dx + dy < 10*10:
			print("Region found at index", i)
			cv.imshow("Region", domain_map[i] * 255)

minWindows()
cv.namedWindow("Time Map " + title)
cv.createTrackbar("P-value threshold", "Time Map " + title, int(pValue_thresh*ratio), 500, callback)
cv.setMouseCallback("Time Map " + title, region_click)
centerXs, centerYs, radii = findCentersAndRadii()

time_map = np.zeros((domain_map.shape[1], domain_map.shape[2], 3), dtype="uint8")

getTimeMap(1)

cv.imshow("Time Map " + title, time_map)
whole_brain_map = time_map

#cv.namedWindow("Node Map")
#drawNodes(centerXs, centerYs, radii)

#cv.setMouseCallback("Node Map", findIndexCallback)

# print(np.max(time_map))

cv.waitKey(0)
cv.destroyAllWindows()




# plt.plot(eventMatrix[6,50])
# plt.show()
# plt.plot(pMatrix[6,50])
# print(minWindows[6,50])
# #print(minWindows)
# plt.show()