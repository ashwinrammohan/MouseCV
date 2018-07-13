import numpy as np
from matplotlib import pyplot as plt
from hdf5manager import *
from derivativeEventDetection import detectSpikes
from eventCharacterization import eventCharacterization

batches = 10000
indexer = np.arange(batches)[:,None]

def getSpikesData(brain_data):
	master_dict = eventCharacterization(brain_data)
	eventIntervals = master_dict["Average Inter-event Interval"]
	print(eventIntervals.shape)
	# start_spikes, mid_spikes, end_spikes, vals = detectSpikes(brain_data[0], -0.3, peak_tolerance = 0.5)
	# est_num_spikes = (len(start_spikes) + len(end_spikes)) * brain_data.shape[0] * 1.5
	# all_spikes = np.empty(est_num_spikes)

	# lower = 0
	# upper = len(start_spikes) + lower
	# upper2 = len(end_spikes) + upper

	# for i, row in enumerate(brain_data[1:]):
	# 	all_spikes[lower:upper] = start_spikes
	# 	all_spikes[upper:upper2] = end_spikes
	# 	lower = upper2

	# 	start_spikes, mid_spikes, end_spikes, vals = detectSpikes(row, -0.3, peak_tolerance = 0.5)
	# 	upper = len(start_spikes) + lower
	# 	upper2 = len(end_spikes) + upper

	# 	if upper2 >= all_spikes.shape[0]:
	# 		est_num_spikes *= 2
	# 		new_all_spikes = np.empty(est_num_spikes)
	# 		new_all_spikes[:lower] = all_spikes
	# 		all_spikes = new_all_spikes

def ratesAtPercentiles(na, nb, spike_list, percentiles, timecourse_length):
	a_spikes = spike_list[np.random.randint(spike_list.shape[0], size = na * batches)].reshape(batches, na)
	b_spikes = spike_list[np.random.randint(spike_list.shape[0], size = nb * batches)].reshape(batches, nb)

	timecourses1 = np.zeros((batches, timecourse_length))
	timecourses2 = np.zeros((batches, timecourse_length))

f = hdf5manager("P2_timecourses.hdf5")
data = f.load()
brain_data = data['brain']
getSpikesData(brain_data)