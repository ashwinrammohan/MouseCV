import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *

class FixedQueue:
	def __init__(self, size, values=[]):
		assert(size > 0), "Queue size must be more than 0"
		if len(values) == size:
			self.values = values
		else:
			self.values = [0.0] * size

		self.size = size
		self.head = 0
		self.sum = 0

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
def detectSpike(data, interval = 20, stDev_threshold = 1.5, derivative_threshold = 4):
	size = len(data)
	spikeIndices = set()
	
	pre_interval = interval
	post_interval = 0

	front_avg = np.mean(data[:pre_interval])
	front_dev = np.std(data[:pre_interval])

	if post_interval != 0:
		back_avg = np.mean(data[-post_interval:])
		back_dev = np.std(data[-post_interval:])

	for i in range(pre_interval):
		if (data[i] - front_avg > front_dev * stDev_threshold):
			print ("Spike detected at x = " + str(i))
			spikeIndices.add(i)

	for i in range(size - post_interval, size):
		if (data[i] - back_avg > back_dev * stDev_threshold):
			print ("Spike detected at x = " + str(i))
			spikeIndices.add(i)

	localAvgs = [0] * pre_interval
	localSTDs = [0] * pre_interval

	meanQueue = FixedQueue(pre_interval, data[:pre_interval])

	for i in range(pre_interval, size - post_interval):

		before = i-pre_interval
		after = i+post_interval

		std_subset = data[before:after]
		localSTDev = np.std(std_subset)

		meanQueue.add_value(data[i])
		localAVG = meanQueue.sum / meanQueue.size

		localAvgs.append(localAVG)
		localSTDs.append(localSTDev)

		if (i < size -1):
			derivative = (data[i+1]-data[i-1])/2

		if (data[i] - localAVG > localSTDev * stDev_threshold) or (derivative > derivative_threshold):
			print("Spike detected at x = " + str(i))
			spikeIndices.add(i)

	localAvgs.extend([0] * post_interval)
	localSTDs.extend([0] * post_interval)

	return (spikeIndices, localAvgs, localSTDs)

data = hdf5manager("P2_timecourses.hdf5").load()
print("brain data below...")
ys = data['brain'][0][:2000]
print("Global Average: " + str(np.mean(ys)))
print("Global Stdev: " + str(np.std(ys)))

xs = list(np.linspace(0,2000,2000))
spikes, avgs, stdevs = detectSpike(ys,100,3)

legend = ("Data", "Avgs", "Stdevs")
plt.plot(xs,vals)
plt.plot(xs,avgs)
plt.plot(xs, stdevs)
plt.legend(legend)

for i in spikes:
	#print("(" + str(xs[i]) + "," + str(ys[i]) + ")" + "\n")
	plt.axvline(x = xs[i], color = 'red')


plt.show()


'''
xs = [1,2,3,4,5,6,7,8,9,10,11,12]
ys = [1,4,9,16,25,36,49,64,81,100,121,144]
