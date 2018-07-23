import numpy as np
from matplotlib import pyplot as plt
from hdf5manager import hdf5manager as h5

def nodeVisualization(eventMatrix, pMatrix, pValue_thresh = 0.001, fps = 10):
	minWindows = np.empty((eventMatrix.shape[0], eventMatrix.shape[1]))
	for i, results in enumerate(pMatrix):
		for j, region in enumerate(results):
			minWindow = (np.where(region < pValue_thresh or region > 1 - pValue_thresh)[0][0] + 1) * (1/fps)
			minWindows[i][j] = minWindow

	return minWindows


f = h5("Outputs/P2_MatrixData_full.hdf5")
data = f.load()
eventMatrix = data['eventMatrix']
pMatrix = data['pMatrix']
minWindows = nodeVisualization(eventMatrix, pMatrix)

print(minWindows)
		
