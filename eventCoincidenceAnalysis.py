import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
import cv2 as cv
import sys
sys.path.append("..\\pyWholeBrain")
from timecourseAnalysis import butterworth


class FixedQueue:
	def __init__(self, size, values=[]):
		assert(size > 0), "Queue size must be more than 0"
		self.size = size
		self.head = 0
		self.sum = 0

		if len(values) == size:
			self.values = values
			for val in values:
				self.sum += val
		else:
			self.values = [0.0] * size

	def add_value(self, value):
		self.sum -= self.values[self.head]
		self.sum += value

		self.values[self.head] = value

		self.head += 1
		if self.head >= self.size:
			self.head = 0

#Method for detecting spikes in data given certain thresholds
#data is the y values
#stDev_threshold - if a y-value is greater than the mean by this threshold * stDev, that point is declared a spike
#derivative_threshold - if the derivative at a point is greater than this, then that point is declared a spike
def detectSpike(data, interval = 20, stDev_threshold = 1.5, derivative_threshold = 4, min = 0.05):
	size = len(data)
	spikeIndices = []
	
	pre_interval = interval
	post_interval = 0

	front_avg = np.mean(data[:pre_interval])
	front_dev = np.std(data[:pre_interval])

	print("FRONT AVERAGE: " + str(front_avg))
	print("FRONT STDEV: " + str(front_dev))



	if post_interval != 0:
		back_avg = np.mean(data[-post_interval:])
		back_dev = np.std(data[-post_interval:])

	for i in range(pre_interval):
		if (data[i] - front_avg > front_dev * stDev_threshold):
			print ("Spike detected at x = " + str(i))
			spikeIndices.append(i)

	for i in range(size - post_interval, size):
		if (data[i] - back_avg > back_dev * stDev_threshold):
			print ("Spike detected at x = " + str(i))
			spikeIndices.append(i)

	localAvgs = [0] * pre_interval
	localSTDs = [0] * pre_interval

	meanQueue = FixedQueue(pre_interval, data[:pre_interval])
	oldSTDev = 0
	newSTDev = 0
	for i in range(pre_interval, size - post_interval):

		before = i-pre_interval
		after = i+post_interval

		std_subset = data[before:after]
		localSTDev = np.std(std_subset)
		newSTDev = localSTDev

		meanQueue.add_value(data[i])
		localAVG = meanQueue.sum / meanQueue.size

		localAvgs.append(localAVG)
		localSTDs.append(localSTDev)

		if (i < size -1):
			derivative = (data[i+1]-data[i-1])/2

		init_check = (data[i] - localAVG > localSTDev * stDev_threshold) or (derivative > derivative_threshold)
		derivative_check = data[i] > data[i-1]
		derivative_stdev_check = True#abs(newSTDev - oldSTDev) > min
		if (init_check and derivative_check and derivative_stdev_check):
			print("Spike detected at x = " + str(i) + " y value = " + str(data[i]) + " local mean = " + str(localAVG) + " local stdev = " + str(localSTDev))
			spikeIndices.append(i)
		oldSTDev = newSTDev

	localAvgs.extend([0] * post_interval)
	localSTDs.extend([0] * post_interval)

	return (spikeIndices, localAvgs, localSTDs)

high_limit = 0.5
low_limit = 0.01
data = hdf5manager("P2_timecourses.hdf5").load()
print("brain data below...")
vals = data['brain'][0][5000:10000]



print("Global Average: " + str(np.mean(vals)))
print("Global Stdev: " + str(np.std(vals)))

xs = list(np.linspace(0,len(vals),len(vals)))
stDev_threshold = 0.5
butter_vals = butterworth(vals, high = high_limit, low = None)
legend = ("Butterworth Data", "Avgs", "Stdevs")

while (True):
	spikes, avgs, stdevs = detectSpike(butter_vals,100,stDev_threshold)
	plt.clf()
	#plt.plot(xs,vals)
	plt.plot(xs,butter_vals)
	plt.plot(xs,avgs)
	plt.plot(xs, stdevs)
	plt.legend(legend)
	plt.title("Standard Deviation Factor: " + str(stDev_threshold))
	plt.axvline(x = spikes[0], color = (1,0,0,1)) #red
	plt.axvline(x = spikes[-1], color = (1,0,0,1)) #red
	for i in range(1, len(spikes)-1):
		if (spikes[i-1] + 1 < spikes[i] and spikes[i] + 1 == spikes[i+1]) or (spikes[i-1] + 1 == spikes[i] and spikes[i] + 1 < spikes[i+1]):
			plt.axvline(x = spikes[i], color = (1,0,0,1)) #red
		else:
			plt.axvline(x = spikes[i], color = (1,1,0,0.3)) #yellow

	plt.show()
	stDev_threshold -= 0.25
	if (cv.waitKey(0) == ord("q")):
		break

