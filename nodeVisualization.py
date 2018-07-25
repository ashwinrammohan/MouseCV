import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from hdf5manager import hdf5manager as h5
import wholeBrain as wb
import cv2 as cv
import time

f = h5("Outputs/P5_MatrixData_full_interp.hdf5")
data = f.load()
eventMatrix = data['eventMatrix']
pMatrix = data['pMatrix']

f2 = h5("P5_timecourses.hdf5")
data2 = f2.load()
domain_map = data2['domainmap']
overlapped_domain_map = np.empty((domain_map.shape[1], domain_map.shape[2]), dtype="uint8")
for i, region in enumerate(domain_map):
	wr = np.where(region == 1)
	overlapped_domain_map[wr[0], wr[1]] = i+1

pValue_thresh = 0.01
index = 0
whole_brain_map = None

def minWindows(fps = 10):
	windows = np.zeros((eventMatrix.shape[0], eventMatrix.shape[1], 3))
	for i, results in enumerate(pMatrix):
		for j, region_pvals in enumerate(results):
			if (i != j and not(np.isnan(region_pvals).any())):
				inds = np.where(region_pvals[1:] > 1 - pValue_thresh)

				if (len(inds[0]) > 0):
					minWindow = (inds[0][0] + 1) * (1/fps)

					if minWindow < 1:
						color = cm.jet(min(1, minWindow))
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

	fullWindows = np.zeros((eventMatrix.shape[0], eventMatrix.shape[1]+1, 3))
	fullWindows[:,0] = np.array([0,0,0])
	fullWindows[:,1:] = windows
	return fullWindows.astype('uint8')

windows = minWindows()
print(np.max(windows))

def nodeVisualization(node_number):
	global time_map

	minWindow = windows[node_number]
	time_map[...] = minWindow[overlapped_domain_map]
	print(minWindow.shape[0] - np.where(minWindow[:,0] == 0)[0].shape[0] - np.where(minWindow[:,0] == 150)[0].shape[0])

	for x, y, r in zip(centerXs, centerYs, radii):
		cv.circle(time_map, (y,x), 2, (255,255,255), -1)

def findRegion(x, y):
	return overlapped_domain_map[y,x]-1


def region_click(event, x, y, flags, param):
	global time_map, color_map, index

	if event == cv.EVENT_LBUTTONUP:
		index = findRegion(x,y)
		print("Displaying index", index)
		nodeVisualization(index)
		cv.imshow("Time Map", time_map)

def callback(x):
	global pValue_thresh, time_map

	windows = minWindows()
	pValue_thresh = x/10000
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


cv.namedWindow("Time Map")
cv.createTrackbar("P-value threshold", "Time Map", int(pValue_thresh*10000), 500, callback)
cv.setMouseCallback("Time Map", region_click)
centerXs, centerYs, radii = findCenters()

time_map = np.zeros((domain_map.shape[1], domain_map.shape[2], 3), dtype="uint8")

nodeVisualization(1)

cv.imshow("Time Map", time_map)
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











