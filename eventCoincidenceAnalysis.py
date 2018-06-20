import numpy as np
import matplotlib
from matplotlib import pyplot

#Method for detecting spikes in data given certain thresholds
#data is of the form: ((xvals), (yvals))
#delta_threshold - threshold for delta between current value and moving average
#stDev_threshold - if a y-value is greater than the mean by this threshold * stDev, that point is declared a spike
def detectSpike(data, delta_threshold = 5, stDev_threshold = 1.5):
	xvals = data[0]
	yvals = data[1]

	ROC = 0
	movingAVG = 0
	done = False
	spikeIndices = set()

	stDev = np.std(yvals)
	mean = np.mean(yvals)

	for i in range(1,len(xvals)-1):
		if (yvals[i] - mean > stDev_threshold * stDev):
			print("Spike detected at x = " + str(xvals[i]) + " due to standard deviation")
			print("Index: " + str(i) + " y value: " + str(yvals[i]) + ", mean: " + str(mean) + ", std: " + str(stDev) + "\n")
			spikeIndices.add(i)
		
		movingAVG = sum(yvals[0:i+1])/(i+1)

		if (yvals[i]- movingAVG > delta_threshold):
			print("Spike detected at x = " + str(xvals[i]) + " due to moving average change")
			print("Index: " + str(i) + ": Y value: " + str(yvals[i]) + ", moving average: " + str(movingAVG) + "\n")
			spikeIndices.add(i)

	return spikeIndices

xs = [1,2,3,4,5,6,7,8,9,10]
ys = [1,4,9,16,25,36,49,64,81,100]
data = [xs,ys]
spikes = detectSpike(data,15,0.5)

for i in spikes:
	print("(" + str(xs[i]) + "," + str(ys[i]) + ")" + "\n")





