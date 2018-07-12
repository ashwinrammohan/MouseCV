import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
from scipy.stats import poisson
from scipy.stats import mode
from derivativeEventDetection import detectSpikes
from eventCoincidence import test_ROI_timecourse
import pandas as pd

#Characterizes each timecourse in a matrix of brain data by number of events, event frequency, maximum event magnitude
#and average inter-event interval (time between consecutive events). Each event in each timecourse is further characterized
#by its magnitude and duration
def eventCharacterization(brain_data):
	max_events = 0
	min_events = 1000
	numRows = brain_data.shape[0]
	for i in range(numRows):
		dataRow = brain_data[i]
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3)
		size = len(start_spikes)
		if (size > max_events):
			max_events = size
		if (size < min_events and size > 0):
			min_events = size
	print("Max events:" + str(max_events))
	print("Min events:" + str(min_events)) 
	master_dict = {"Duration": np.zeros((numRows, max_events)), "Number of Events": np.zeros(numRows), "Event Frequency": np.zeros(numRows), "Event Magnitude": np.zeros((numRows, max_events)), "Max Magnitude": np.zeros(numRows), "Inter-event Interval": np.zeros(numRows)}
	master_dict["Duration"][:][:] = np.NaN
	master_dict["Event Magnitude"][:][:] = np.NaN

	for i in range(numRows):
		print("Doing timecourse number " + str(i))
		dataRow = brain_data[i]
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3)
		size = len(start_spikes)
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
			master_dict["Inter-event Interval"][i] = sumIntervals/size

	return master_dict

#finds the most commonly occurring event frequency for a given time course to characterize it
#start_spikes - the starting indices of each of the events in the timecourse
#size is the length of the timecourse in frames
def eventFrequencyMode(start_spikes, size):
	bins = np.arange(100,size+100,100) #10 second (100 frames at 10 fps) intervals along the entire timecourse
	eventRates = []

	for j in range(0,bins.shape[0]):
		start = bins[j] - 100
		end = bins[j]
		num_local_events = 0
		for k in range(len(start_spikes)):
			if (start_spikes[k] >= start and start_spikes[k] <= end):
				num_local_events +=1
		#print("(" + str(start) + " to " + str(end) + "): " + str(num_local_events/10))
		eventRates.append(num_local_events/10) #number of events per 10 seconds
	eventRates = np.asarray(eventRates)
	relevantRates = eventRates[np.where(eventRates > 0.0)][0] #find nonzero event rates (the relevant ones)
	mode_val = mode(np.asarray(relevantRates))[0][0] #find the most commonly occurring event frequency in this timecourse
	return mode_val

data = hdf5manager("P2_timecourses.hdf5").load()
brain_data = data['brain']
domain_map = data['domainmap']
frequency_map = np.zeros((domain_map.shape[1], domain_map.shape[2])).astype('float')
number_map = np.zeros((domain_map.shape[1], domain_map.shape[2])).astype('float')
maxMag_map = np.zeros((domain_map.shape[1], domain_map.shape[2])).astype('float')
eventInterval_map = np.zeros((domain_map.shape[1], domain_map.shape[2])).astype('float')
master_dict = eventCharacterization(brain_data)
f = master_dict["Event Frequency"]
n = master_dict["Number of Events"]
mag = master_dict["Event Magnitude"]
dur = master_dict["Duration"]
maxMag = master_dict["Max Magnitude"]
eventInterval = master_dict["Inter-event Interval"]
length = brain_data.shape[0]

ars = np.empty((domain_map.shape[0]), dtype="uint32")
for i in range(domain_map.shape[0]):
	xy_data = domain_map[i]

	ar = np.sum(xy_data)
	#print("Domain " + str(i) + " area: " + str(ar))	
	ars[i] = ar

	local_freq = f[i]
	local_num = n[i]
	localMaxMag = maxMag[i]
	localInterval = eventInterval[i]
	for j in range(xy_data.shape[0]):
		oneS = np.where(xy_data[j]==1)[0]
		for k in range(oneS.shape[0]):
			frequency_map[j,oneS[k]] = local_freq
			number_map[j,oneS[k]] = local_num
			maxMag_map[j, oneS[k]] = localMaxMag
			eventInterval_map[j, oneS[k]] = localInterval

#s = np.arange(0,ars.shape[0],1)
# plt.plot(ars, 'r-')
# plt.show()

# plt.imshow(frequency_map, cmap = 'Reds'), plt.colorbar(), plt.title("Event Frequency Spatial Map"), plt.axis('off')
# plt.show()
# plt.imshow(number_map, cmap = 'Reds'), plt.colorbar(), plt.title("Number of Events Spatial Map"), plt.axis('off')
# plt.show()
# plt.imshow(maxMag_map, cmap = 'Reds'), plt.colorbar(), plt.title("Maximum Event Magnitude Spatial Map"), plt.axis('off')
# plt.show()
# plt.imshow(eventInterval_map, cmap = 'Reds'), plt.colorbar(), plt.title("Average Inter-Event Interval Magnitude Spatial Map"), plt.axis('off')
# plt.show()


# plt.subplot(221),plt.imshow(mag, cmap='hot', interpolation = 'nearest'), plt.colorbar(), plt.title("Event Magnitude")
# plt.subplot(222), plt.imshow(dur, cmap = 'hot', interpolation = 'nearest'), plt.colorbar(), plt.title("Event Duration")
# plt.subplot(223), plt.bar(np.arange(1,length+1,1),f), plt.title("Event Frequency (Mode)"), plt.xlabel("Time Course Number"), plt.ylabel("Event Frequency"), plt.xticks(np.arange(0,length+2, 2)), plt.tight_layout()
# plt.subplot(224), plt.bar(np.arange(1,length+1,1),n), plt.title("Number of Events"), plt.xlabel("Time Course Number"), plt.ylabel("Number of Events"), plt.xticks(np.arange(0,length+2, 2)), plt.tight_layout()
# plt.suptitle("P5 Time Course Event Characterization")
# plt.show()


# plt.figure(figsize = (50,6))
# freq = pd.DataFrame(mag.T)
# freq.boxplot()
# plt.savefig("Event Stuff.png")


f = hdf5manager("Outputs/P2_MatrixData_full.hdf5")
data = f.load()
eventMatrix, pMatrix, preMatrix = data['eventMatrix'], data['pMatrix'], data['precursors']
print(preMatrix.shape)

plt.subplot(121),plt.imshow(preMatrix[:,:,5])
numPrecursors = {}
plot_count = 0
preMatrixAdd = np.empty_like(preMatrix)

lt_mask = np.tri(preMatrix.shape[0], preMatrix.shape[0])[:,::-1]
lt_mask = np.broadcast_to(lt_mask[...,None], (preMatrix.shape[0], preMatrix.shape[0], preMatrix.shape[2]))

lt = lt_mask * preMatrix
preMatrixAdd = preMatrix + lt[::-1,::-1]

plt.subplot(122),plt.imshow(preMatrixAdd[:,:,5]), plt.colorbar(), plt.show()
fig, axs  = plt.subplots(3,2, figsize = (10,12))
axs = axs.ravel()

fig2, axs2  = plt.subplots(3,2, figsize = (10,12))
axs2 = axs2.ravel()

for i in range(preMatrix.shape[0]):
	done = False
	count = 0
	timeWindows = []
	for j in range(preMatrix[i].shape[0]):
		pre = preMatrix[i][j]
		trueIndices = [ind for ind, x in enumerate(pre) if x]
		if (len(trueIndices) > 0):
			count +=1
			timeWindows = [(x+1)*0.1 for i,x in enumerate(trueIndices)]
		random_j = np.random.randint(0,preMatrix[i].shape[0])
		labels = [i, random_j]
		if ((i != random_j) and (i % 50 == 0) and not done):
			done = True
			#print(plot_count)
			axs[plot_count].plot(pMatrix[i][random_j],'r-'),axs[plot_count].plot(eventMatrix[i][random_j]), axs[plot_count].set_title("Time Course " + str(i) + " coinciding with time course " + str(random_j))
			axs2[plot_count].plot(brain_data[i]), axs2[plot_count].plot(brain_data[random_j]),axs2[plot_count].set_title("Time Course " + str(i) + " coinciding with time course " + str(random_j)), axs2[plot_count].legend(labels)
			plot_count += 1
	if (len(timeWindows) > 0):
		print(str(i) + "\t" + str(count)+ "\t" + str(mode(np.asarray(timeWindows))[0][0]))
plt.show()

	#numPrecursors[count] = mode(np.asarray(timeWindows))[0][0]
print(numPrecursors)
#print([i for i,x in enumerate(np.asarray(numPrecursors)) if x>500])

# numRows = brain_data[:5].shape[0]
# for i in range(numRows):
# 	dataRow = brain_data[i]
# 	xs = np.linspace(0,dataRow.shape[0], dataRow.shape[0])
# 	start_spikes, mid_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3)
# 	plt.plot(xs,vals)
# 	for i in start_spikes:
# 		plt.axvline(x = i, color = 'red')
# 	for i in mid_spikes:
# 		plt.axvline(x = i, color = (1,1,0,0.3))
# 	for i in end_spikes:
# 		plt.axvline(x = i, color = 'red')

# 	plt.show()


# numRows = brain_data.shape[0]
# plt.imshow(brain_data, aspect = 'auto'), plt.colorbar()
# for i in range(numRows):
# 		dataRow = brain_data[i]
# 		binarizedRow = np.zeros_like(dataRow)
# 		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(dataRow, second_deriv_thresh = -0.3, peak_tolerance = 1)
# 		# binarizedRow[start_spikes] = 1
# 		# binarizedRow[mid_spikes] = 1
# 		if (np.std(dataRow) > 0.8):
# 			binarizedRow[end_spikes] = 1
# 		print(np.sum(binarizedRow))
# 		plt.plot(np.where(binarizedRow==1)[0], np.repeat(i, np.sum(binarizedRow)),'.k')
# #plt.savefig("Binarized_Overlay.svg")
# plt.plot(np.std(brain_data, axis = 1))
# plt.show()
# for i in range(numRows):
# 	dataRow = brain_data[i]
# 	xs = np.linspace(0,dataRow.shape[0], dataRow.shape[0])
# 	start_spikes, mid_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3)
# 	plt.plot(xs,vals)
# 	for i in start_spikes:
# 		plt.axvline(x = i, color = 'red')
# 	for i in mid_spikes:
# 		plt.axvline(x = i, color = (1,1,0,0.3))
# 	for i in end_spikes:
# 		plt.axvline(x = i, color = 'red')

# 	plt.show()

