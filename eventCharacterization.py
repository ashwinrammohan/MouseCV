import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
from scipy.stats import mode
from derivativeEventDetection import detectSpikes, FixedQueue
from eventCoincidence import eventCoin, _displayInfo, visualizeProgress
import ctypes as c
from multiprocessing import Process, Array, cpu_count, Manager
import pandas as pd
import time

nRange = 5

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


def bootstrapData(brain_data):
	numRows = brain_data.shape[0]

	np_durations = []
	np_intervals = []

	for i in range(numRows):
		print("Doing timecourse number " + str(i))
		dataRow = brain_data[i]
		start_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3)

		if (start_spikes.shape[0] > 250):
			plt.plot(vals)
			plt.show()

		dur = end_spikes - start_spikes
		inter = start_spikes[1:] - end_spikes[:-1]
		inter = inter[inter >= 0]

		bin = start_spikes.shape[0] // nRange
		if (bin >= len(np_durations)):
			needed = bin - len(np_durations) + 1
			np_durations.extend([None]*needed)
			np_intervals.extend([None]*needed)

		if np_durations[bin] is None:
			np_durations[bin] = dur
			np_intervals[bin] = inter
		else:
			newDur = np.empty(dur.shape[0] + np_durations[bin].shape[0])
			newInter = np.empty(inter.shape[0] + np_intervals[bin].shape[0])

			if dur.shape[0] > 0:
				newDur[:-dur.shape[0]] = np_durations[bin]
				newDur[-dur.shape[0]:] = dur
				np_durations[bin] = newDur

			if inter.shape[0] > 0:
				newInter[:-inter.shape[0]] = np_intervals[bin]
				newInter[-inter.shape[0]:] = inter
				np_intervals[bin] = newInter

	for i, dur in enumerate(np_durations):
		if dur is None:
			np_durations[i] = np.empty(0)
			np_intervals[i] = np.empty(0)

	return np_durations, np_intervals

def bootstrapInfo(np_durations, np_intervals, batches, n, max_length):
	if n // nRange >= len(np_intervals) or n // nRange >= len(np_durations):
		return np.zeros((batches, n))

	np_durations = np_durations[n // nRange]
	np_intervals = np_intervals[n // nRange]

	if np_durations.shape[0] == 0:
		return np.zeros((batches, n))

	duration_rand_inds = np.random.randint(np_durations.shape[0], size = n * batches)
	interval_rand_inds = np.random.randint(np_intervals.shape[0], size = n * batches)

	joined_array = np.empty((batches, n * 2))
	durations = np_durations[duration_rand_inds].reshape((batches, n))
	intervals = np_intervals[interval_rand_inds].reshape((batches, n))

	durations_totals = np.sum(durations, axis=1)
	intervals_totals = np.sum(intervals, axis=1)
	maxs = np.repeat(max_length, batches)
	spaces = maxs - durations_totals
	ratios = (spaces/intervals_totals)[...,None]
	print("Ratios:", ratios)
	intervals = intervals * ratios

	joined_array[:,::2] = durations
	joined_array[:,1::2] = intervals

	a_inds = np.cumsum(joined_array, axis = 1)[:,::2]
	return np.floor(a_inds)

def generate_lookup(brain_data, n_min, n_max, n_interval = 25, fps = 10,  max_window = 2, threads = 0, stDev_threshold = 0.8, batches = 10000, percent = 0.05, mid_interval = 100):
	timecourse_length = brain_data.shape[1]
	if threads == 0:
		wanted_threads = cpu_count()
	else:
		wanted_threads = threads

	win_t = np.arange((1/fps),max_window,(1/fps))
	wanted_threads = cpu_count()
	threads = []

	print("Creating spikes data...")
	rates = int(batches * percent)
	mid_indices = np.arange(rates, batches-rates, mid_interval)
	total_rates = rates*2 + mid_indices.shape[0]

	t = time.clock()
	np_sizes = np.arange(n_min, n_max, n_interval)
	numRows = np_sizes.shape[0]
	eventMatrix = Array(c.c_double, numRows*numRows*win_t.shape[0]*total_rates)
	spikes_a = np.empty((numRows, batches, n_max), dtype="uint8")
	spikes_b = np.empty((numRows, batches, n_max), dtype="uint8")
	np_durations, np_intervals = bootstrapData(brain_data)
	displayDict = Manager().dict()
	displayDict["done"] = 0

	for i in range(numRows):
		a_ind = bootstrapInfo(np_durations, np_intervals, batches, np_sizes[i], timecourse_length)
		b_ind = bootstrapInfo(np_durations, np_intervals, batches, np_sizes[i], timecourse_length)

		spikes_a[i][:,:np_sizes[i]] = a_ind
		spikes_b[i][:,:np_sizes[i]] = b_ind

	if wanted_threads > numRows:
		wanted_threads = numRows

	index_map = []
	for i in range(wanted_threads):
		index_map.extend(list(np.arange(i,numRows,wanted_threads)))

	index_map = np.asarray(index_map)
	print("index map:", index_map)
	inv_index_map = np.argsort(index_map)
	print("Created data arrays in", time.clock() - t, "seconds")

	print("Creating " + str(wanted_threads) + " threads...")
	dataPer = int(numRows / wanted_threads)
	upper = 0
	names = []
	for i in range(wanted_threads):
		name = "Thread " + str(i+1)
		names.append(name)

		displayDict["i_"+name] = 0
		displayDict["j_"+name] = 0
		displayDict["avg_na_"+name] = 0
		displayDict["avg_nb_"+name] = 0
		displayDict["total_time_"+name] = 0
		displayDict["avg_dt_"+name] = 0
		displayDict["processed_"+name] = 0
		displayDict["needed_"+name] = 1

		if i == wanted_threads-1:
			p = Process(target=_eventCoin, args=(upper, numRows, numRows, spikes_a[index_map], spikes_b[index_map], np_sizes[index_map], win_t, eventMatrix, timecourse_length, fps, rates, mid_interval, displayDict, name))
		else:
			lower = i*dataPer
			upper = (i+1)*dataPer
			p = Process(target=_eventCoin, args=(lower, upper, numRows, spikes_a[index_map], spikes_b[index_map], np_sizes[index_map], win_t, eventMatrix, timecourse_length, fps, rates, mid_interval, displayDict, name))
		p.start()
		threads.append(p)

	_displayInfo(displayDict, wanted_threads, names)
	print("All threads done")

	eventMatrix = np.frombuffer(eventMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0], total_rates))[inv_index_map][:,inv_index_map]

	p_vals = np.empty(total_rates)
	p_vals[:rates] = np.arange(0,rates)
	p_vals[-rates:] = np.arange(batches-rates,batches)
	p_vals[rates:rates+mid_indices.shape[0]] = mid_indices
	p_vals = p_vals/batches

	return eventMatrix, p_vals

def p_values(coins, rates):
	sorted_coins = np.sort(coins, axis=0)
	p_vals = np.empty((coins.shape[1], rates.shape[0]))

	for i in range(coins.shape[1]):
		indices = np.searchsorted(sorted_coins[:,i], rates)
		p_vals[i] = indices / coins.shape[0]

	return p_vals


def _eventCoin(rowsLower, rowsUpper, numRows, spikes_a, spikes_b, num_spikes, win_t, eventMatrix, timecourse_length, fps, rates, mid_interval, dispDict, name):
	print("New thread created, running from " + str(rowsLower) + " to " + str(rowsUpper))
	avg_na = FixedQueue(20)
	avg_nb = FixedQueue(20)
	total_time = time.clock()
	avg_dt = FixedQueue(20)
	dt = time.clock()
	processed = 0
	disp_time = 0
	batches = spikes_a.shape[1]
	#np_rates = np.arange(0,1,1.0/rates)
	mid_indices = np.arange(rates, batches-rates, mid_interval)
	total_rates = rates*2 + mid_indices.shape[0]

	eventResults = np.empty((rowsUpper - rowsLower, numRows, win_t.shape[0], total_rates))
	coins = np.empty((batches, win_t.shape[0]))

	needed = (rowsUpper - rowsLower) * numRows * batches
	dispDict["needed_"+name] = needed

	bin_tcs1 = np.zeros((timecourse_length), dtype="uint8")
	bin_tcs2 = np.zeros((timecourse_length), dtype="uint8")

	for i in range(rowsLower, rowsUpper):
		for j in range(numRows):
			all_spike_tcs1 = spikes_a[i,:,:num_spikes[i]]
			all_spike_tcs2 = spikes_b[j,:,:num_spikes[j]]

			for k in range(batches):
				if time.clock() - disp_time >= 1:
					dispDict["i_"+name] = i
					dispDict["j_"+name] = j
					dispDict["avg_na_"+name] = avg_na.sum/20
					dispDict["avg_nb_"+name] = avg_nb.sum/20
					dispDict["total_time_"+name] = time.clock() - total_time
					dispDict["avg_dt_"+name] = avg_dt.sum/20
					dispDict["processed_"+name] = processed
					disp_time = time.clock()

				processed += 1

				bin_tcs1[all_spike_tcs1[k]] = 1
				bin_tcs2[all_spike_tcs2[k]] = 1

				event_data, na, nb = eventCoin(bin_tcs1, bin_tcs2, win_t=win_t, ratetype='precursor', verbose = False, veryVerbose = False)
				coins[k] = event_data

				bin_tcs1[:] = 0
				bin_tcs2[:] = 0

				avg_dt.add_value(time.clock() - dt)
				dt = time.clock()
				avg_na.add_value(na)
				avg_nb.add_value(nb)

			#eventResults[i-rowsLower, j] = p_values(coins, np_rates)
			sorted_coins = np.sort(coins, axis=0)
			for k in range(win_t.shape[0]):
				eventResults[i-rowsLower, j, k, :rates] = sorted_coins[:rates, k]
				eventResults[i-rowsLower, j, k, rates:rates+mid_indices.shape[0]] = sorted_coins[mid_indices, k]
				eventResults[i-rowsLower, j, k, rates+mid_indices.shape[0]:] = sorted_coins[-rates:, k]

	eventNp = np.frombuffer(eventMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0], total_rates))
	eventNp[rowsLower:rowsUpper] = eventResults
	dispDict["done"] += 1
	dispDict["i_"+name] = i
	dispDict["j_"+name] = j
	dispDict["total_time_"+name] = time.clock() - total_time
	dispDict["processed_"+name] = processed

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
	

def eventGraphing(coincidenceFileName, dataFileName = None, dataFile = None):
	assert not (dataFileName is None and dataFile is None), "Source data was not given in any form"

	if dataFile is None:	
		data = hdf5manager(dataFileName).load()
	else:
		data = dataFile

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

	fig, axs = plt.subplots(2,2, figsize = (10,12))
	axs = axs.ravel()

	graph = axs[0].imshow(frequency_map, cmap = 'Reds')
	axs[0].set_title("Event Frequency")
	axs[0].axis('off')
	fig.colorbar(graph, ax = axs[0])

	graph = axs[1].imshow(number_map, cmap = 'Reds')
	axs[1].set_title("Number of Events")
	axs[1].axis('off')
	fig.colorbar(graph, ax = axs[1])


	graph = axs[2].imshow(maxMag_map, cmap = 'Reds')
	axs[2].set_title("Maximum Magnitude")
	axs[2].axis('off')
	fig.colorbar(graph, ax = axs[2])

	graph = axs[3].imshow(eventInterval_map, cmap = 'Reds')
	axs[3].set_title("Average Inter-Event Interval Magnitude")
	axs[3].axis('off')
	fig.colorbar(graph, ax = axs[3])
	
	plt.show()


	# plt.subplot(221),plt.imshow(mag, cmap='hot', interpolation = 'nearest'), plt.colorbar(), plt.title("Event Magnitude")
	# plt.subplot(222), plt.imshow(dur, cmap = 'hot', interpolation = 'nearest'), plt.colorbar(), plt.title("Event Duration")
	# plt.subplot(223), plt.bar(np.arange(1,length+1,1),f), plt.title("Event Frequency (Mode)"), plt.xlabel("Time Course Number"), plt.ylabel("Event Frequency"), plt.xticks(np.arange(0,length+2, 2)), plt.tight_layout()
	# plt.subplot(224), plt.bar(np.arange(1,length+1,1),n), plt.title("Number of Events"), plt.xlabel("Time Course Number"), plt.ylabel("Number of Events"), plt.xticks(np.arange(0,length+2, 2)), plt.tight_layout()
	# plt.suptitle("P5 Time Course Event Characterization")
	# plt.show()

	f = hdf5manager(coincidenceFileName)
	data = f.load()
	eventMatrix, pMatrix, preMatrix = data['eventMatrix'], data['pMatrix'], data['precursors']
	print(preMatrix.shape)

	plt.subplot(121),plt.imshow(preMatrix[:,:,5].astype("uint8")), plt.colorbar()
	numPrecursors = {}
	plot_count = 0
	preMatrixAdd = np.empty_like(preMatrix)

	lt_mask = np.tri(preMatrix.shape[0], preMatrix.shape[0])[:,::-1]
	lt_mask = np.broadcast_to(lt_mask[...,None], (preMatrix.shape[0], preMatrix.shape[0], preMatrix.shape[2]))

	lt = lt_mask * preMatrix
	preMatrixAdd = preMatrix + lt[::-1,::-1]

	plt.subplot(122),plt.imshow(preMatrixAdd[:,:,5]), plt.colorbar(), plt.show()
	fig2, axs2  = plt.subplots(3,2, figsize = (10,12))
	axs2 = axs2.ravel()

	fig3, axs3  = plt.subplots(3,2, figsize = (10,12))
	axs3 = axs3.ravel()

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
				axs2[plot_count].plot(eventMatrix[i][random_j])
				axs2[plot_count].set_xlabel("Time Window (seconds)")
				axs2[plot_count].set_ylabel("Event Coincidence Rate")
				axs2[plot_count].set_title("Time Course " + str(i) + " coinciding with time course " + str(random_j))

				fig2.tight_layout()

				axs3[plot_count].plot(brain_data[i])
				axs3[plot_count].plot(brain_data[random_j])
				axs3[plot_count].set_xlabel("Frame Number")
				axs3[plot_count].set_ylabel("Signal Intensity")
				axs3[plot_count].set_title("Time Course " + str(i) + " coinciding with time course " + str(random_j))
				axs3[plot_count].legend(labels)

				fig3.tight_layout()

				plot_count += 1

		if (len(timeWindows) > 0):
			print(str(i) + "\t" + str(count)+ "\t" + str(mode(np.asarray(timeWindows))[0][0]))
	
	print("graphs created")
	plt.show()

		#numPrecursors[count] = mode(np.asarray(timeWindows))[0][0]
	print(numPrecursors)
		#print([i for i,x in enumerate(np.asarray(numPrecursors)) if x>500])

def miscellaneousGraphs():
	numRows = brain_data.shape[0]
	plt.imshow(brain_data, aspect = 'auto'), plt.colorbar()
	for i in range(numRows):
			dataRow = brain_data[i]
			binarizedRow = np.zeros_like(dataRow)
			start_spikes, mid_spikes, end_spikes, vals = detectSpikes(dataRow, second_deriv_thresh = -0.3, peak_tolerance = 1)
			# binarizedRow[start_spikes] = 1
			# binarizedRow[mid_spikes] = 1
			if (np.std(dataRow) > 0.8):
				binarizedRow[end_spikes] = 1
			print(np.sum(binarizedRow))
			plt.plot(np.where(binarizedRow==1)[0], np.repeat(i, np.sum(binarizedRow)),'.k')
	#plt.savefig("Binarized_Overlay.svg")
	plt.plot(np.std(brain_data, axis = 1))
	plt.show()
	for i in range(numRows):
		dataRow = brain_data[i]
		xs = np.linspace(0,dataRow.shape[0], dataRow.shape[0])
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3)
		plt.plot(xs,vals)
		for i in start_spikes:
			plt.axvline(x = i, color = 'red')
		for i in mid_spikes:
			plt.axvline(x = i, color = (1,1,0,0.3))
		for i in end_spikes:
			plt.axvline(x = i, color = 'red')

		plt.show()
#eventGraphing("Outputs/171018_03_MatrixData_full.hdf5", dataFileName = "P2_timecourses.hdf5")

if __name__ == '__main__': 
	import argparse

	ap = argparse.ArgumentParser()
	ap.add_argument('-f', '--filename', type = str, nargs = 1, required = True, help = 'name of hdf5 input file with ICA-filtered timecourses')
	ap.add_argument('-g', '--graphs', action = "store_true", required = False, help = 'display graphs after completing')
	ap.add_argument('-l', '--lookup', action = "store_true", required = False, help = 'generate the lookup table')

	args = vars(ap.parse_args())

	data = hdf5manager(args['filename'][0]).load()

	if "ROI_timecourses" in data.keys():
		brain_data = data['ROI_timecourses']
	elif "brain" in data.keys():
		brain_data = data["brain"]
	else:
		print("No data found! Maybe the hdf5 is formatted incorrectly?")
		import sys
		sys.exit()

	if args['lookup']:
		max_n  = 100
		n_interval = 5
		min_n = n_interval
		eventMatrix, p_vals = generate_lookup(brain_data, min_n, max_n, n_interval = n_interval)
		shp = eventMatrix.shape
		fullMatrix = np.empty((shp[0]+1, shp[1]+1, shp[2], shp[3]), dtype=eventMatrix.dtype)
		fullMatrix[1:,1:] = eventMatrix
		fullMatrix[0] = 0.5 # for na of 0
		fullMatrix[:,0] = 0.5 # for nb of 0

		fileString = "Outputs/P5_lookup.hdf5"
		fileData = {"table": fullMatrix, "interval": n_interval, "max_n": max_n, "p_values": p_vals}
		saveData = hdf5manager(fileString)
		saveData.save(fileData)
		print("Saved event coincidence data to " + fileString)


	if args["graphs"]:
		eventGraphing(fileString, dataFile = data)

