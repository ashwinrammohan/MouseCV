import time
import numpy as np
from matplotlib import pyplot as plt
from hdf5manager import hdf5manager as h5
from derivativeEventDetection import detectSpikes
from eventCharacterization import eventCharacterization

numRows = brain_data.shape[0]

all_start_spikes = []
all_end_spikes = []
total_start_events = 0
total_end_events = 0

for i in range(numRows):
	print("Doing timecourse number " + str(i))
	dataRow = brain_data[i]
	start_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3)
	
	total_start_events += start_spikes.shape[0]
	total_end_events += end_spikes.shape[0]
	all_start_spikes.append(start_spikes)
	all_end_spikes.append(end_spikes)

np_start_spikes = np.empty(total_start_events)
np_end_spikes = np.empty(total_end_events)

lower = 0
upper = 0

for spikes in all_start_spikes:
	upper += spikes.shape[0] 
	np_start_spikes[lower:upper] = spikes
	lower = upper

for spikes in all_end_spikes:
	upper += spikes.shape[0] 
	np_end_spikes[lower:upper] = spikes
	lower = upper

np_durations = np_end_spikes - np_start_spikes
np_intervals = np_start_spikes[1:] - np_end_spikes[:-1]
np_intervals = np_intervals[np_intervals >= 0]

duration_rand_inds = np.random.random(np_durations.shape[0], size = na * batches)
interval_rand_inds = np.random.random(np_intervals.shape[0], size = na * batches)

joined_array = np.empty((batches, na * 2))
joined_array[:,::2] = np_durations[duration_rand_inds].reshape((batches, na))
joined_array[:,1::2] = np_intervals[interval_rand_inds].reshape((batches, na))

a_inds = np.cumsum(joined_array, axis = 1)[:,::2]

duration_rand_inds = np.random.random(np_durations.shape[0], size = nb * batches)
interval_rand_inds = np.random.random(np_intervals.shape[0], size = nb * batches)

joined_array = np.empty((batches, nb * 2))
joined_array[:,::2] = np_durations[duration_rand_inds].reshape((batches, nb))
joined_array[:,1::2] = np_intervals[interval_rand_inds].reshape((batches, nb))

b_inds = np.cumsum(joined_array, axis = 1)[:,::2]