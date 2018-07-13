import numpy as np
from matplotlib import pyplot as plt
from hdf5manager import *
from derivativeEventDetection import detectSpikes

def getSpikesData(brain_data):
	start_spikes, mid_spikes, end_spikes, vals = detectSpikes(brain_data[0], -0.3, peak_tolerance = 0.5)
	est_num_spikes = (len(start_spikes) + len(end_spikes)) * brain_data.shape[0] * 1.5
	all_spikes = np.empty(est_num_spikes)

	lower = 0
	upper = len(start_spikes) + lower
	upper2 = len(end_spikes) + upper

	for i, row in enumerate(brain_data[1:]):
		all_spikes[lower:upper] = start_spikes
		all_spikes[upper:upper2] = end_spikes
		lower = upper2

		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(row, -0.3, peak_tolerance = 0.5)
		upper = len(start_spikes) + lower
		upper2 = len(end_spikes) + upper

		if upper2 >= all_spikes.shape[0]:
			est_num_spikes *= 2
			new_all_spikes = np.empty(est_num_spikes)
			new_all_spikes[:lower] = all_spikes
			all_spikes = new_all_spikes

def ratesAtPercentiles(na, nb, data, percentiles, batches=10000)