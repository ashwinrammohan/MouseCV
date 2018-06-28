import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
from scipy.stats import poisson
from scipy.stats import mode
from derivativeEventDetection import detectSpikes

def eventCharacterization(brain_data):
	max_events = 0
	numRows = brain_data.shape[0]
	for i in range(numRows):
		dataRow = brain_data[i]
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3)
		size = len(start_spikes)
		if (size > max_events):
			max_events = size
	master_dict = {"Duration": np.empty_like((numRows, max_events)), "Number of Events": np.empty_like(numRows), "Event Frequency": np.empty_like(numRows), "Event Magnitude": np.empty_like((numRows, max_events))}
	master_dict["Duration"][:] = np.NaN
	master_dict["Event Frequency"][:] = np.NaN

	start_spike_set = []
	for i in range(numRows):
		dataRow = brain_data[i]
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3)
		master_dict["Number of Events"][i] = len(start_spikes)
		master_dict["Event Frequency"][i] = event_Frequency(start_spikes,dataRow.shape[0])
		size = len(start_spikes)
		for j in range(size):
			master_dict["Duration"][i][j] = end_spikes[j] - start_spikes[j]
			master_dict["Event Magnitude"][i][j] = dataRow[end_spikes[j]] - np.mean(dataRow[start_spikes[j]:end_spikes[j]])

	return master_dict

#finds the most commonly occurring event frequency for a given time course to characterize it
#start_spikes - the starting indices of each of the events in the timecourse
#size is the length of the timecourse in frames
def event_Frequency_Mode(start_spikes, size):
	bins = np.arange(100,size+100,100) #10 second (100 frames at 10 fps) intervals along the entire timecourse
	eventCoincidenceRates = []

	for j in range(0,bins.shape[0]):
		start = bins[j] - 100
		end = bins[j]
		num_local_events = 0
		for k in range(len(start_spikes)):
			if (start_spikes[k] >= start and start_spikes[k] <= end):
				num_local_events +=1
		#print("(" + str(start) + " to " + str(end) + "): " + str(num_local_events/10))
		eventCoincidenceRates.append(num_local_events/10) #number of events per 10 seconds
	mode_val = mode(nd.asarray(eventCoincidenceRates)) #find the most commonly occurring event frequency in this timecourse
	return mode_val


data = hdf5manager("P2_timecourses.hdf5").load()
brain_data = data['brain'][0]
start_spikes, mid_spikes, end_spikes, vals = detectSpikes(brain_data, -0.3)
print(event_Frequency(start_spikes, brain_data.shape[0]))

