import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from hdf5manager import hdf5manager as h5
import wholeBrain as wb
import cv2 as cv
import time

f = h5("Outputs/171018_03_MatrixData_full.hdf5")
data = f.load()
eventMatrix = data['eventMatrix']
pMatrix = data['pMatrix']

f2 = h5("P2_timecourses.hdf5")
data2 = f2.load()
domain_map = data2['domainmap']
overlapped_domain_map = np.empty((domain_map.shape[1], domain_map.shape[2]), dtype="uint8")
for i, region in enumerate(domain_map):
	wr = np.where(region == 1)
	overlapped_domain_map[wr[0], wr[1]] = i+1

ratio = 10000000
pValue_thresh = 0.00001
index = 0
whole_brain_map = None
trigger = False
title = "Trigger" if trigger else "Precursor"

def minWindows(fps = 10):
	windows = np.zeros((eventMatrix.shape[0], eventMatrix.shape[1], 3))
	for i, results in enumerate(pMatrix):
		for j, region_pvals in enumerate(results):
			if (i != j and not(np.isnan(region_pvals).any())):
				inds = np.where(region_pvals > 1 - pValue_thresh)

				if (len(inds[0]) > 0):
					minWindow = (inds[0][0] + 1) * (1/fps)

					if minWindow < 2:
						color = cm.jet(minWindow/2)
						windows[i][j][0] = color[2] * 255
						windows[i][j][1] = color[1] * 255
						windows[i][j][2] = color[0] * 255
				else:
					windows[i][j] = 150
			else:
				if (i == j):
					windows[i][j] = 255
				else:
					windows[i][j] = 0

	fullWindows = np.zeros((eventMatrix.shape[0]+1, eventMatrix.shape[1]+1, 3))
	fullWindows[:,0] = np.array([0,0,0])
	fullWindows[0,:] = np.array([0,0,0])
	fullWindows[1:,1:] = windows
	return fullWindows.astype('uint8')

windows = minWindows()
print(np.max(windows))

def nodeVisualization(node_number):
	global time_map

	if trigger:
		minWindow = windows[:,node_number]
	else:
		minWindow = windows[node_number]

	time_map[...] = minWindow[overlapped_domain_map]

	for x, y, r in zip(centerXs, centerYs, radii):
		cv.circle(time_map, (y,x), 2, (255,255,255), -1)

def findRegion(x, y):
	return overlapped_domain_map[y,x]


def region_click(event, x, y, flags, param):
	global time_map, color_map, index

	if event == cv.EVENT_LBUTTONUP:
		index = findRegion(x,y)
		print("Displaying index", index)
		nodeVisualization(index)
		cv.imshow("Time Map " + title, time_map)
	elif event == cv.EVENT_RBUTTONUP:
		index2 = findRegion(x,y)
		if trigger:
			win_t = np.where(pMatrix[index2, index] > 1 - pValue_thresh)[0][0]
			print("Coincidence rate of", index, "against", index2, "=", eventMatrix[index2, index, win_t])
		else:
			win_t = np.where(pMatrix[index, index2] > 1 - pValue_thresh)[0][0]
			print("Coincidence rate of", index, "against", index2, "=", eventMatrix[index, index2, win_t])

def callback(x):
	global pValue_thresh, time_mapx

	windows = minWindows()
	pValue_thresh = x/ratio
	nodeVisualization(index)

	
def findCenters():
	centerXs = np.zeros(domain_map.shape[0], dtype = "uint32")
	centerYs = np.zeros(domain_map.shape[0], dtype = "uint32")
	radii = np.zeros(domain_map.shape[0], dtype = "uint32")

	for i, domain in enumerate(domain_map):
		res = np.where(domain == 1)
		if (len(res[0]) >= 1):
			centerXs[i] = np.mean(res[0])
			centerYs[i] = np.mean(res[1])

			xDists = np.square(res[0] - centerXs[i])
			yDists = np.square(res[1] - centerYs[i])
			radii[i] = np.mean(np.sqrt(xDists + yDists))

	return centerXs, centerYs, radii


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


cv.namedWindow("Time Map " + title)
cv.createTrackbar("P-value threshold", "Time Map " + title, int(pValue_thresh*ratio), 500, callback)
cv.setMouseCallback("Time Map " + title, region_click)
centerXs, centerYs, radii = findCenters()

time_map = np.zeros((domain_map.shape[1], domain_map.shape[2], 3), dtype="uint8")

nodeVisualization(1)

cv.imshow("Time Map " + title, time_map)
whole_brain_map = time_map

#cv.namedWindow("Node Map")
#drawNodes(centerXs, centerYs, radii)

#cv.setMouseCallback("Node Map", findIndexCallback)

print(np.max(time_map))

cv.waitKey(0)
cv.destroyAllWindows()




# plt.plot(eventMatrix[6,50])
# plt.show()
# plt.plot(pMatrix[6,50])
# print(minWindows[6,50])
# #print(minWindows)
# plt.show()











