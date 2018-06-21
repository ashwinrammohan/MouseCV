import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
import cv2 as cv
import sys
sys.path.append("/Users/andrew/Code Github/pyWholeBrain")
from timecourseAnalysis import butterworth

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
def detectSpikes(data, second_deriv_thresh, second_deriv_batch = 3, deriv_start = 0):
	past_deriv = 0
	past_derivs = FixedQueue(second_deriv_batch)
	future_dervis = FixedQueue(second_deriv_batch)
	spikes = []

	# initialize batches
	for i in range(1, second_deriv_batch+1):
		deriv = data[i] - data[i-1]
		past_deriv = deriv
		past_derivs.add_value(deriv)

	for i in range(second_deriv_batch+2, 2*second_deriv_batch+2):
		deriv = data[i] - data[i-1]
		future_dervis.add_value(deriv)

	# calculate spikes
	for i in range(second_deriv_batch+1, len(data) - second_deriv_batch-1):
		deriv = data[i] - data[i-1]

		if (deriv < 0 and past_deriv >= 0) and ((future_dervis.sum - past_derivs.sum) < second_deriv_thresh):
			spikes.append(i)

		past_deriv = deriv
		past_derivs.add_value(deriv)

		deriv = data[i+second_deriv_batch+1] - data[i+second_deriv_batch]
		future_dervis.add_value(deriv)

	# find activity duration
	i = 0
	new_spikes = []
	for spike in spikes:
		i = spike - 1
		done = False
		while not(done):
			deriv = data[i] - data[i-1]
			if (deriv <= deriv_start):
				done = True
			else:
				new_spikes.append(i)
				i -= 1
				if (i < 1):
					done = True

	spikes.extend(new_spikes)
	return spikes

high_limit = 0.5
low_limit = 0.03
data = hdf5manager("P2_timecourses.hdf5").load()
vals = butterworth(data['brain'][0][:2000], high = high_limit, low = None)
xs = list(np.linspace(0,2000,2000))

spikes = detectSpikes(vals, -0.2)

legend = ("Data","Butterworth Data", "Avgs", "Stdevs")
plt.plot(xs,vals)
#plt.plot(xs,butter_vals)
#plt.plot(xs,avgs)
#plt.plot(xs, stdevs)
plt.legend(legend)

for i in spikes:
	#print("(" + str(xs[i]) + "," + str(ys[i]) + ")" + "\n")
	plt.axvline(x = xs[i], color = 'red')

plt.show()