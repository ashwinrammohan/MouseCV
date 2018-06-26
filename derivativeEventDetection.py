import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
import cv2 as cv
import sys

path_file = open("path.txt", "r")
sys.path.append(path_file.read())
path_file.close()

from timecourseAnalysis import butterworth

# This class is an extremely efficient queue with a fixed size. It keeps track of the
# sum of its values.
class FixedQueue:
	# Creates a FixedQueue with the given size. User can give starting values if they want, 
	# but the values must be the correct length
	def __init__(self, size, values=[]): 
		assert(size > 0), "Queue size must be more than 0"
		if len(values) == size:
			self.values = values
		else:
			self.values = [0.0] * size # fills with zeros

		self.size = size
		self.head = 0
		self.sum = 0

	# Adds a value to the queue, and removes the oldest value if the queue is full
	def add_value(self, value):
		self.sum -= self.values[self.head]
		self.sum += value # update the sum

		self.values[self.head] = value

		self.head += 1
		if self.head >= self.size: # if we've reached the end of the list, go to the beginning
			self.head = 0

# Loads an HDF5 file with an assumed structure. The dictionary produced must have up to 6 kv pairs,
# with the names footFL, footFR, footBL, footBR, head, tail. Not all kv pairs need to exist.
# The HDF5 file must be stored in the Assets folder.
def loadHDF5(file_name):
	data = hdf5manager("Assets/" + file_name + ".hdf5").load()

	def loadLimb(limbName, pos, color):
		if not(limbName in data.keys()): # check if this kv pair actually exists
			return

		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(data[limbName]["magnitude"], -0.3) # find events in magnitude data
		data[limbName]["pos"] = pos # position of the vector for graphing purposes
		data[limbName]["color"] = color # color of the vector when graphing
		data[limbName]["start_spikes"] = start_spikes
		data[limbName]["mid_spikes"] = mid_spikes
		data[limbName]["end_spikes"] = end_spikes

	loadLimb("footFL", (0,1), (1, 0, 0))
	loadLimb("footFR", (0,-1), (0, 1, 0))
	loadLimb("footBL", (-1,1), (1, 0.7, 0.3))
	loadLimb("footBR", (-1,-1), (0, 0, 1))
	loadLimb("head", (1,0), (1, 0.3, 1))
	loadLimb("tail", (-2,0), (0.3, 1, 1))
	return data

# Uses first and second derivatives to find peaks in data. `second_deriv_thresh` is the threshold for the second derivative of the data.
# Since the peaks are downward facing, this value should be negative, and any second derivative less negative than this value will not be considered a peak.
# `second_deriv_batch` is the size of the buffer of derivatives that are stored, both before and after the current data point being evaluated. 
# It is essentially an average, so a larger batch size will make wider peaks easier to detect, and reduce noise, but possibly miss narrow peaks.
# `deriv_start` is the minimum value of the first derivative that counts as the beginning of the event. This should rarely be changed
# `peak_height` is the minimum height of a peak. It will be calculated automatically, but the user may override the automatic calculation if they want.
# `high_pass` is the value passed to the butterworth filter which cuts off high frequencies. The number itself must range from 0 to 1, but 
# what the numbers actually means is unclear. Regardless, a lower value will cut off more frequencies.
def detectSpikes(data, second_deriv_thresh, second_deriv_batch = 3, deriv_start = 0, peak_height = 0, high_pass = 0.5):
	data = butterworth(data, high = high_pass, low = None) # smooth/de-noise the data

	past_deriv = 0
	past_derivs = FixedQueue(second_deriv_batch)
	future_dervis = FixedQueue(second_deriv_batch)
	spikes = [] # the top of each spike will be stored here
	if (peak_height == 0):
		peak_height = np.std(data)/3 # if no peak_height is given, the standard deviation is used to guess a good min height
		#("Min peak height: " + str(peak_height))

	# initialize past batch
	for i in range(1, second_deriv_batch+1):
		deriv = data[i] - data[i-1]
		past_deriv = deriv
		past_derivs.add_value(deriv)

	# initialize future batch
	for i in range(second_deriv_batch+2, 2*second_deriv_batch+2):
		deriv = data[i] - data[i-1]
		future_dervis.add_value(deriv)

	# calculate spikes
	for i in range(second_deriv_batch+1, len(data) - second_deriv_batch-1):
		deriv = data[i] - data[i-1] # first derivative

		deriv_test = (deriv < 0 and past_deriv >= 0) # whether there is a peak of any kind (positive slope to negative slope)
		second_deriv_test = ((future_dervis.sum - past_derivs.sum) < second_deriv_thresh) # whether the peak is extreme enough 
		if deriv_test and second_deriv_test:
			spikes.append(i)

		past_deriv = deriv
		past_derivs.add_value(deriv) # update past derivs

		deriv = data[i+second_deriv_batch+1] - data[i+second_deriv_batch]
		future_dervis.add_value(deriv) # update future derivs

	# find activity duration
	i = 0
	start_spikes = [] # will hold the beginning of each spike, using the `deriv_start` parameter
	mid_spikes = [] # will hold all indicies within activities. This is mostly used for graphing
	for j in range(len(spikes)-1, -1, -1): # iterate backwards so that spikes can be removed without issue
		spike = spikes[j]
		i = spike - 1
		done = False
		while not(done): # walk backwards on the data until the beginning of the activity is detected
			deriv = data[i] - data[i-1]
			if (deriv <= deriv_start):
				done = True
			else:
				i -= 1
				if (i < 1): # if you reach the start of the data, it also counts as the beginning of the activity
					done = True

		if data[spike] - data[i] < peak_height: # check whether the total height of the peak is enough
			del spikes[j]
		else:
			start_spikes.append(i)
			for h in range(spike-1, i, -1):
				mid_spikes.append(h)

	return (start_spikes, mid_spikes, spikes, data)

def test_ROI_timecourse():
	data = hdf5manager("P5_timecourses.hdf5").load()
	brain_data_size = data['brain'].shape[0]
	start_spike_set = []
	for i in range(brain_data_size):
		size = len(data['brain'][i])
		#plt.figure("ROI " + str(i))
		xs = list(np.linspace(0,size,size))
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(data['brain'][i], -0.3)
		start_spike_set.append(start_spikes)
		#plt.plot(xs,vals)
		'''
		for i in start_spikes:
			plt.axvline(x = xs[i], color = 'red')
		for i in mid_spikes:
			plt.axvline(x = xs[i], color = (1,1,0,0.3))
		for i in end_spikes:
			plt.axvline(x = xs[i], color = 'red')
		'''
	plt.show()
	plot_event_Coincidence_Rates(start_spike_set,data['brain'].shape[1])

def plot_event_Coincidence_Rates(start_spike_set, size):
	print(len(start_spike_set))
	bins = np.arange(100,size+100,100)
	eventCoincidenceRates = []
	for i in range(len(start_spike_set)):
		start_spikes = start_spike_set[i]
		for j in range(0,bins.shape[0]):
			start = bins[j] - 100
			end = bins[j]
			num_events = 0
			for k in range(len(start_spikes)):
				if (start_spikes[k] >= start and start_spikes[k] <= end):
					num_events +=1
			#print("(" + str(start) + " to " + str(end) + "): " + str(num_events/10))
			eventCoincidenceRates.append(num_events/10)
	max_val = max(eventCoincidenceRates)
	print(str(max_val) + " occurred " + str(eventCoincidenceRates.count(max_val)) + " times")
	plt.hist(np.asarray(eventCoincidenceRates),6), plt.xlabel("Number of events/10 seconds"), plt.ylabel("Number of Occurrences")
	plt.title("Event Coincidence Rate Analysis")
	plt.show()


def test_amplitude():
	data = loadHDF5("mouse_vectors")

	for limbKey in data.keys():
		plt.figure("Limb: " + limbKey)
		xs = list(np.linspace(0,len(data[limbKey]["magnitude"]),len(data[limbKey]["magnitude"])))
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(data[limbKey]["magnitude"], -0.1, second_deriv_batch=8, high_pass = 0.75, peak_height=0.25)

		plt.plot(xs,vals)
		for i in start_spikes:
			plt.axvline(x = i, color = 'red')
		for i in mid_spikes:
			plt.axvline(x = i, color = (1,1,0,0.3))
		for i in end_spikes:
			plt.axvline(x = i, color = 'red')

	plt.show()


test_amplitude()