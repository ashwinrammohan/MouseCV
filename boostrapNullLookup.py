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

def ratesAtPercentiles(na, nb, spike_list, percentiles, timecourse_length):
	a_spikes = spike_list[np.random.randint(spike_list.shape[0], size = na * batches)].reshape(batches, na)
	b_spikes = spike_list[np.random.randint(spike_list.shape[0], size = nb * batches)].reshape(batches, nb)

	timecourses1 = np.zeros((batches, timecourse_length))
	timecourses2 = np.zeros((batches, timecourse_length))

f = hdf5manager("P2_timecourses.hdf5")
data = f.load()
brain_data = data['brain']
getSpikesData(brain_data)



#Characterizes each timecourse in a matrix of brain data by number of events, event frequency, maximum event magnitude
#and average inter-event interval (time between consecutive events). Each event in each timecourse is further characterized
#by its magnitude and duration
def eventCharacterization(brain_data):
	max_events = 0
	min_events = 1000
	numRows = brain_data.shape[0]

	master_dict = {"Duration": np.zeros((numRows, max_events)), "Number of Events": np.zeros(numRows), "Event Frequency": np.zeros(numRows), "Event Magnitude": np.zeros((numRows, max_events)), "Max Magnitude": np.zeros(numRows), "Average Inter-event Interval": np.zeros(numRows*max_events)}
	master_dict["Duration"][:][:] = np.NaN
	master_dict["Event Magnitude"][:][:] = np.NaN

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

		size = start_spikes.shape[0]
		if (size > 0):
			master_dict["Number of Events"][i] = size
			master_dict["Event Frequency"][i] = eventFrequencyMode(start_spikes,dataRow.shape[0])
			maxMag = 0
			sumIntervals = 0
			for j in range(size):
				master_dict["Duration"][i][j] = (end_spikes[j] - start_spikes[j])/10
				mag = dataRow[end_spikes[j]] - np.mean(dataRow[start_spikes[j]:end_spikes[j]])
				master_dict["Event Magnitude"][i][j] = mag
				if (j < size-1):
					sumIntervals += (start_spikes[j+1] - end_spikes[j])/10
				if (mag > maxMag):
					maxMag = mag
			master_dict["Max Magnitude"][i] = maxMag
			master_dict["Average Inter-event Interval"][i] = sumIntervals/size

	np_start_spikes = np.empty(total_start_events)
	lower = 0
	upper = 0
	for spikes in all_start_spikes:
		upper += spikes.shape[0] 
		np_start_spikes[lower:upper] = spikes
		lower = upper

	return master_dict
start[1:] - endspikes[:-1]