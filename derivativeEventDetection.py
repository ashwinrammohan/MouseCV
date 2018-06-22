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

def loadHDF5(file_name):
	data = hdf5manager("Assets/" + file_name + ".hdf5").load()

	def loadLimb(limbName, pos, color):
		if not(limbName in data.keys()):
			return

		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(data[limbName]["magnitude"], -0.3)
		data[limbName]["pos"] = pos
		data[limbName]["color"] = color
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

def detectSpikes(data, second_deriv_thresh, second_deriv_batch = 3, deriv_start = 0, peak_height = 0, high_pass = 0.5):
	data = butterworth(data, high = high_pass, low = None)

	past_deriv = 0
	past_derivs = FixedQueue(second_deriv_batch)
	future_dervis = FixedQueue(second_deriv_batch)
	spikes = []
	if (peak_height == 0):
		peak_height = np.std(data)/3
		print("Min peak height: " + str(peak_height))

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

		deriv_test = (deriv < 0 and past_deriv >= 0)
		second_deriv_test = ((future_dervis.sum - past_derivs.sum) < second_deriv_thresh)
		if deriv_test and second_deriv_test:
			spikes.append(i)

		past_deriv = deriv
		past_derivs.add_value(deriv)

		deriv = data[i+second_deriv_batch+1] - data[i+second_deriv_batch]
		future_dervis.add_value(deriv)

	# find activity duration
	i = 0
	start_spikes = []
	mid_spikes = []
	for j in range(len(spikes)-1, -1, -1):
		spike = spikes[j]
		i = spike - 1
		done = False
		while not(done):
			deriv = data[i] - data[i-1]
			if (deriv <= deriv_start):
				done = True
			else:
				i -= 1
				if (i < 1):
					done = True

		if data[spike] - data[i] < peak_height:
			del spikes[j]
		else:
			start_spikes.append(i)
			for h in range(i+1, spike):
				mid_spikes.append(h)

	return (start_spikes, mid_spikes, spikes, data)

def test_ROI_timecourse():
	data = hdf5manager("P2_timecourses.hdf5").load()

	for i in range(10):
		plt.figure("ROI " + str(i))
		xs = list(np.linspace(0,2000,2000))
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(data['brain'][i][:2000], -0.3)

		plt.plot(xs,vals)
		for i in start_spikes:
			plt.axvline(x = xs[i], color = 'red')
		for i in mid_spikes:
			plt.axvline(x = xs[i], color = (1,1,0,0.3))
		for i in end_spikes:
			plt.axvline(x = xs[i], color = 'red')

	plt.show()

def test_amplitude():
	data = loadHDF5("mouse_vectors")

	for limbKey in data.keys():
		plt.figure("Limb: " + limbKey)
		xs = list(np.linspace(0,len(data[limbKey]["magnitude"]),len(data[limbKey]["magnitude"])))
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(data[limbKey]["magnitude"], -0.05, second_deriv_batch=10, high_pass = 0.3)

		plt.plot(xs,vals)
		for i in start_spikes:
			plt.axvline(x = i, color = 'red')
		for i in mid_spikes:
			plt.axvline(x = i, color = (1,1,0,0.3))
		for i in end_spikes:
			plt.axvline(x = i, color = 'red')

	plt.show()
