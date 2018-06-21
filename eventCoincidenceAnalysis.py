import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *

class FixedQueue:
	def __init__(self, size, values=None):
		if values != None and len(values) == size:
			self.values = values
		else:
			self.values = [0] * size

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
	
	mid_interval = interval#int(interval/2)

	front_avg = np.mean(data[:mid_interval])
	front_dev = np.std(data[:mid_interval])
	back_avg = np.mean(data[-mid_interval:])
	back_dev = np.std(data[-mid_interval:])

	for i in range(mid_interval):
		if (data[i] - front_avg > front_dev * stDev_threshold):
			print ("Spike detected at x = " + str(i))
			spikeIndices.add(i)

	for i in range(size - mid_interval, size):
		if (data[i] - back_avg > back_dev * stDev_threshold):
			print ("Spike detected at x = " + str(i))
			spikeIndices.add(i)

	localAvgs = [0] * mid_interval
	localSTDs = [0] * mid_interval
	for i in range(mid_interval, size):

		before = i-mid_interval
		after = i+mid_interval

		std_subset = data[before:i]
		mean_subset = data[before:i]
		localAVG = np.mean(mean_subset)
		localSTDev = np.std(std_subset)

		localAvgs.append(localAVG)
		localSTDs.append(localSTDev)

		if (i < size -1):
			derivative = (data[i+1]-data[i-1])/2

		if (data[i] - localAVG > localSTDev * stDev_threshold) | (derivative > derivative_threshold):
			print("Spike detected at x = " + str(i))
			spikeIndices.add(i)
	return (spikeIndices, localAvgs, localSTDs)

data = hdf5manager("P2_timecourses.hdf5").load()
vals = data['brain'][0][:2000]
print("Global Average: " + str(np.mean(vals)))
print("Global Stdev: " + str(np.std(vals)))

xs = list(np.linspace(0,2000,2000))

spikes, avgs, stdevs = detectSpike(vals,100,3)

legend = ("Data", "Avgs", "Stdevs")
plt.plot(xs,vals)
plt.plot(xs,avgs)
plt.plot(xs, stdevs)
plt.legend(legend)

for i in spikes:
	#print("(" + str(xs[i]) + "," + str(ys[i]) + ")" + "\n")
	plt.axvline(x = xs[i], color = 'red')


plt.show()

queueObj = FixedQueue(10)
print(queueObj)

answer = 0
while answer != -1:
	answer = int(input(">>> "))
	queueObj.add_value(answer)
	print(queueObj.sum())

