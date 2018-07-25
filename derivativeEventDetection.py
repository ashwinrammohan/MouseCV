import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
from scipy.stats import poisson
#import cv2 as cv
import sys

try:
	path_file = open("path.txt", "r")
	sys.path.append(path_file.read())
	path_file.close()
except:
	print("Can't import, path.txt doesn't exist")
	pass

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

		start_spikes, end_spikes, vals = detectSpikes(data[limbName]["magnitude"], -0.3) # find events in magnitude data
		data[limbName]["pos"] = pos # position of the vector for graphing purposes
		data[limbName]["color"] = color # color of the vector when graphing
		data[limbName]["start_spikes"] = start_spikes
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
# `peak_tolerance` is the minimum height of a peak. It will be calculated automatically, but the user may override the automatic calculation if they want.
# `high_pass` is the value passed to the butterworth filter which cuts off high frequencies. The number itself must range from 0 to 1, but 
# what the numbers actually means is unclear. Regardless, a lower value will cut off more frequencies.
def detectSpikes(data, second_deriv_thresh, second_deriv_batch = 3, deriv_start = 0, peak_tolerance = 0, high_pass = 0.5):
	mina = np.amin(data)
	maxa = np.amax(data)
	mean = np.mean(data)

	if mean > (mina + maxa)/2: # flip data if needed
		data = data*-1

	data = butterworth(data, high = high_pass, low = None) # smooth/de-noise the data

	df = data[1:] - data[:-1] # get derivates

	binary_df = np.copy(df) # copy derivates
	binary_df[binary_df < 0] = -1 # make derivates binarized (-1 if negative, 1 if positive, 0 if 0)
	binary_df[binary_df > 0] = 1

	binary_dff = binary_df[1:] - binary_df[:-1] # get binarized second derivates
	df_peaks = np.where(binary_dff < 0)[0] + 1 # get indecies of possible peaks from first deriv test

	# df_batches = np.copy(df[second_deriv_batch-1:]) # get sum of derivates in batches of `second_deriv_batch`
	# for i in range(second_deriv_batch-1):
	# 	lower = i
	# 	upper = second_deriv_batch - i - 1
	# 	df_batches += df[lower : -upper]

	# dff_batches = df_batches[second_deriv_batch:] - df_batches[:-second_deriv_batch] # get differences of batches
	# dff_peaks = np.where(dff_batches < second_deriv_thresh)[0] + second_deriv_batch # get indecies of possible peaks from second deriv test

	end_spikes = list(df_peaks + 1)#list(np.intersect1d(df_peaks, dff_peaks) + 1)

	if (peak_tolerance == 0):
		peak_tolerance = max(np.std(data)/3, 8) # if no peak_tolerance is given, the standard deviation is used to guess a good min height
	else:
		peak_tolerance = max(np.std(data) * peak_tolerance, 8)
	# find activity duration
	i = 0
	to_delete = []
	last_base = end_spikes[0]
	avg_size = 20
	max_delta = np.std(data) * 0.75
	j = 0

	start_spikes = [] # will hold the beginning of each spike, using the `deriv_start` parameter

	for spike_location in range(len(end_spikes)):
		spike = end_spikes[spike_location]
		i = spike

		done = False
		add_to_prev = True
		while not(done) or (add_to_prev and j < i): # walk backwards on the data until the beginning of the activity is detected
			if not(done):
				i -= 1
				deriv = df[i-1]
				
				if (deriv <= deriv_start) or (i == 0): # if you reach the start of the data, it also counts as the beginning of the activity
					done = True

			if add_to_prev:
				j += 1
				base = data[j]

				avg = np.sum(data[last_base-avg_size:last_base+1]) / avg_size
				if abs(avg - base) <= max_delta or base < 0:
					add_to_prev = False

		if add_to_prev and spike_location > 0:
			to_delete.append(spike_location-1)

		else:
			last_base = i
			if data[spike] - data[i] < peak_tolerance: # check whether the total height of the peak is enough
				to_delete.append(spike_location)
			else:
				start_spikes.append(i)
		
		j = spike

	for i in to_delete[::-1]:
		del end_spikes[i]
		
	return (np.asarray(start_spikes, dtype="uint32"), np.asarray(end_spikes, dtype="uint32"), data)
	
def test_amplitude():
	data = loadHDF5("mouse_vectors")

	for limbKey in data.keys():
		plt.figure("Limb: " + limbKey)
		xs = np.linspace(0,len(data[limbKey]["magnitude"]),len(data[limbKey]["magnitude"]))
		start_spikes, end_spikes, vals = detectSpikes(data[limbKey]["magnitude"], -0.1, second_deriv_batch=8, high_pass = 0.75, peak_tolerance=0.25)

		print(len(start_spikes), len(end_spikes))
		print(start_spikes[0], end_spikes[0])
		print(start_spikes[-1], end_spikes[-1])

		plt.plot(xs,vals)
		print("new code", limbKey, start_spikes)
		for i in start_spikes:
			plt.axvline(x = i, color = 'red')
		for i in end_spikes:
			plt.axvline(x = i, color = 'red')

	plt.show()

def test_timecourse():
	data = hdf5manager("P7_timecourses_domainROI.hdf5").load()["ROI_timecourses"]
	start_spikes, end_spikes, vals = detectSpikes(data[5], -0.3, peak_tolerance = 0)
	plt.plot(vals)
	plt.show()
	start_spikes, end_spikes, vals = detectSpikes(data[7], -0.3, peak_tolerance = 0.5)
	print(len(start_spikes)+len(end_spikes))

	specific_is = [0,5,10,200,266]

	for i in specific_is:

		plt.figure("timecourse #" + str(i))
		start_spikes, end_spikes, vals = detectSpikes(data[i], -0.3, peak_tolerance = 0.5)
		plt.plot(vals)

		mina = np.amin(data[i])
		maxa = np.amax(data[i])
		mean = np.mean(data[i])
		plt.axhline(y = mina, color='red')
		plt.axhline(y = maxa, color='blue')
		plt.axhline(y = mean, color='green')

		for i in start_spikes:
			plt.axvline(x = i, color = 'red')
		for i in end_spikes:
			plt.axvline(x = i, color = 'green')

		plt.show()

	return
	for i, row in enumerate(data):
		mina = np.amin(row)
		maxa = np.amax(row)
		mean = np.mean(row)

		if mean > (mina + maxa)/2:
			plt.figure("inverted? #:" + str(i))
			start_spikes, end_spikes, vals = detectSpikes(row, -0.3, peak_tolerance = 0.5)
			plt.plot(vals)

			plt.axhline(y = mina, color='red')
			plt.axhline(y = maxa, color='blue')
			plt.axhline(y = mean, color='green')

			plt.show()

		# for i in start_spikes:
		# 	plt.axvline(x = i, color = 'red')
		# for i in end_spikes:
		# 	plt.axvline(x = i, color = 'red')


#test_timecourse()

