import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
from scipy.stats import poisson
from derivativeEventDetection import detectSpikes
import time
from multiprocessing import Process, Array, cpu_count
import ctypes as c

def _eventCoin(rowsLower, rowsUpper, numRows, binarized_data, win_t, eventMatrix, pMatrix, brain_data, fps):
		print("New thread created, running from " + str(rowsLower) + " to " + str(rowsUpper))

		eventResults = np.empty((rowsUpper - rowsLower, numRows, win_t.shape[0]))
		pResults = np.empty((rowsUpper - rowsLower, numRows, win_t.shape[0]))

		for i in range(rowsLower, rowsUpper):
			for j in range(numRows):
				if (i != j):
					print("Comparing " + str(i) + " to " + str(j))
					bin_tcs1 = binarized_data[i]
					bin_tcs2 = binarized_data[j]
					event_data, na, nb = eventCoin(bin_tcs1,bin_tcs2, win_t=win_t, ratetype='precursor', verbose = False, veryVerbose = False)
					eventResults[i-rowsLower, j] = event_data
					pResults[i-rowsLower, j] = getResults(event_data, win_t=win_t, na=na, nb=nb, T = brain_data.shape[1]/fps, fps = fps, verbose = False, veryVerbose = False)
				else:
					eventResults[i-rowsLower, j] = np.NaN
					pResults[i-rowsLower, j] = np.NaN

		eventNp = np.frombuffer(eventMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0]))
		pNp = np.frombuffer(pMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0]))

		eventNp[rowsLower:rowsUpper] = eventResults
		pNp[rowsLower:rowsUpper] = pResults

def test_ROI_timecourse(brain_data, fps = 10,  max_window = 2, start_event = True, mid_event = True, end_event = True, threads = 0):
	binarized_data = np.zeros_like(brain_data).astype('uint8')
	numRows = brain_data.shape[0]
	start_spike_set = []
	mid_spike_set = []
	end_spike_set = []
	win_t = np.arange((1/fps),max_window,(1/fps))

	for i in range(brain_data.shape[0]):
		dataRow = brain_data[i]
		binarizedRow = np.zeros_like(dataRow)
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3)
		for j in range(dataRow.shape[0]):
			check = False
			if (start_event):
				check = check or j in start_spikes
			if (not check and mid_event):
				check = check or j in mid_spikes
			if (not check and end_event):
				check = check or j in end_spikes
			if (check):
				binarizedRow[j] = 1
		binarized_data[i,:] = binarizedRow

	if threads == 0:
		wanted_threads = cpu_count()
	else:
		wanted_threads = threads

	threads = []
	print("Creating " + str(wanted_threads) + " threads...")

	eventMatrix = Array(c.c_double, numRows*numRows*win_t.shape[0])
	pMatrix = Array(c.c_double, numRows*numRows*win_t.shape[0]) 
	print("Created empty data arrays")

	dataPer = int(numRows / wanted_threads)
	upper = 0
	for i in range(wanted_threads-1):
		lower = i*dataPer
		upper = (i+1)*dataPer

		p = Process(target=_eventCoin, args=(lower, upper, numRows, binarized_data, win_t, eventMatrix, pMatrix, brain_data, fps))
		p.start()
		threads.append(p)

	_eventCoin(upper, numRows, numRows, binarized_data, win_t, eventMatrix, pMatrix, brain_data, fps)
	for p in threads:
		p.join()

	eventMatrix = np.frombuffer(eventMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0]))
	pMatrix = np.frombuffer(pMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0]))

	print(eventMatrix[:,:,7])
	print(pMatrix[:,:,7])
	return eventMatrix, pMatrix

	#plt.plot(xs,vals)
	'''
	for i in start_spikes:
		plt.axvline(x = xs[i], color = 'red')
	for i in mid_spikes:
		plt.axvline(x = xs[i], color = (1,1,0,0.3))
	for i in end_spikes:
		plt.axvline(x = xs[i], color = 'red')
		'''		
	#plt.show()
	'''	
	diff = np.vstack((binary0,binary1))
	print("Reached vstack")
	plt.imshow(diff, aspect = 'auto')
	plt.show()

	win_t = np.arange(0, 2, 1/10)
	rand, na, nb = eventCoin(binary0,binary1, win_t=win_t, ratetype='precursor', verbose = False, veryVerbose = False)

	plt.plot(win_t, rand,  label='Random')
	plt.title('Rates per time window')
	plt.xlabel('Time window (sce)')
	plt.ylabel('Precursor coincidence rate')
	plt.legend()
	plt.show()

	rand_results = getResults(rand, win_t=win_t, na=na, nb=nb, verbose = True, veryVerbose = False)
	#plt.show()
	#plot_event_Coincidence_Rates(start_spike_set,data['brain'].shape[1])
	#return start_spike_set
	'''

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



def eventCoin(a, b, #two binary signals to compare
			  win_t, #vector of time (s) for window
			  na = None, nb = None, #number of total events in each comparitive vector
			  ratetype = 'precursor', #precursor or trigger
			  tau = 0, #lag coefficeint 
			  fps = 10, #sampling rate, frames per sec
			  verbose = True,
			  veryVerbose = False):
	
	if na == None:
		na = sum(a)
	if nb == None:
		nb = sum(b)
	
	#find all indices for each event
	a_ind = [i for i, e in enumerate(a) if e != 0]
	b_ind = [i for i, e in enumerate(b) if e != 0]
	
	#convert window times to window frames
	win_fr = win_t * fps 
	
	#create index difference matrix
	ind_diff = np.zeros((len(a_ind), len(b_ind)))

	for i,inda in enumerate(a_ind): # rows
		for j,indb in enumerate(b_ind): # columns
			ind_diff [i,j]= inda - (tau*fps) - indb	

	#create event matrix
	if ratetype == 'precursor':
		#print('Calculating PRECURSOR coincidence \n ----------------------------------')

		events = np.zeros((ind_diff.shape[0], len(win_fr)))
		ind_diff[ind_diff < 0] = 0 #max(win_fr)*1.5
		if veryVerbose:
			print('Size of difference array: ',ind_diff.shape)
			plt.imshow(ind_diff)
			plt.title('Difference in indices')
			plt.xlabel('indices from list a')
			plt.ylabel('indices from list b')
			plt.colorbar()
			plt.show()
		start_time = time.clock()

		for i, win in enumerate(win_fr):
			if verbose:
				print('Calculating PRECURSOR coincidence rate for window ' + str(win/fps) +'sec(s)')

			new_diff = ind_diff / win
			new_diff = 1 - new_diff
			new_diff = (np.absolute(new_diff) + new_diff) / 2
			binarized = np.ceil(new_diff)
			mask_zeros = 1 - np.floor(new_diff)
			binarized = np.multiply(binarized, mask_zeros)

			row_sum = np.heaviside(np.sum(binarized, axis=1), 0)
			events[:,i] = row_sum
			# events[:,i] = np.ceil(row_sum / new_diff.shape[0])

		rate_win = np.sum(events, axis=0)/na
		print("Took " + str(time.clock() - start_time) + " seconds.")

	if ratetype == 'trigger':
		#print('Calculating TRIGGER coincidence \n ----------------------------------')
		
		events = np.zeros((ind_diff.shape[1], len(win_fr)))
		ind_diff[ind_diff > 0] = 0
		
		start_time = time.clock()
		for i, win in enumerate(win_fr):
			if verbose:
				print('Calculating coincidence rate for window ' + str(win/fps) +'sec(s)')

			new_diff = -ind_diff / win
			new_diff = 1 - new_diff
			new_diff = (np.absolute(new_diff) + new_diff) / 2
			binarized = np.ceil(new_diff)
			mask_zeros = 1 - np.floor(new_diff)
			binarized = np.multiply(binarized, mask_zeros)

			row_sum = np.sum(binarized, axis=0)
			events[:,i] = np.heaviside(row_sum, 0)
		
		rate_win = np.sum(events, axis=0)/nb
		print("Took " + str(time.clock() - start_time) + " seconds.")
					
	if verbose:
		plt.imshow(events, aspect = 'auto')
		plt.title('Event matrix')
		plt.xlabel('Time window')
		plt.ylabel('Coincidence per event')
		plt.show()

		plt.plot(win_t, rate_win)
		plt.title('Rates per time window')
		plt.xlabel('Time window (sec)')
		if ratetype == 'precursor':
			plt.ylabel('Precursor coincidence rate')
		if ratetype == 'trigger':
			plt.ylabel('Trigger coincidence rate')
		plt.show()

	return rate_win, na, nb

def getResults(rate_win,              
			  win_t, #vector of time (s) for window
			  na, #number of events in a
			  nb, #number of event in b
			  ratetype = 'precursor', #precursor or trigger
			  T = 600, #length (s) of vector
			  tau = 0, #lag coefficeint 
			  fps = 10, #sampling rate, frames per sec
			  verbose = True,
			  veryVerbose = False):
	
	#expected rate and stdev of the rate
	if ratetype == 'precursor':
		rho = 1 - win_t/(T - tau)
		exp_rate = na*(1 - (rho)**nb)
		exp_std = np.sqrt(1/na*(1-rho**nb) * rho**nb)
	if ratetype == 'trigger':
		rho = 1 - win_t/(T - tau)
		exp_rate = nb*(1 - (rho)**na)
		exp_std = np.sqrt(1/nb*(1-rho**na) * rho**na)
	
	#quantiles used for graphing
	if verbose:
		perc = np.array([1, 2.5, 25, 50, 75, 97.5, 99])
		mark = ['k:', 'k-.', 'k--', 'k-', 'k--','k-.', 'k:']
		quantile = np.zeros((exp_rate.shape[0], perc.shape[0]))

	#number samples for null hypothesis
	k=10000

	sample = np.zeros((exp_rate.shape[0], k))
	results = np.zeros(exp_rate.shape[0])

	for i, r in enumerate(exp_rate):
		sample[i,:] = poisson.rvs(r, size=k)
		if ratetype == 'precursor':
			if verbose:
				quantile[i,:] = np.percentile(sample[i,:], perc)/na
			results[i] = sum(rate_win[i] < sample[i, :]/na)/k
		if ratetype == 'trigger':
			if verbose:
				quantile[i,:] = np.percentile(sample[i,:], perc)/nb
			results[i] = sum(rate_win[i] < sample[i, :]/nb)/k
		if veryVerbose:
			print(str(win_t[i]) + 'sec(s) time window produces a p value: ' + str(results[i]))

	if verbose:
		for j, r in enumerate(results):
			if r < 0.05:
				print(str(win_t[j]) + 'sec(s) time window produces a significant value: p=' + str(r))
	
	# plot sample values
	if veryVerbose:
		plt.imshow(sample, aspect = 'auto')
		plt.colorbar()
		plt.show()

	if verbose:
		for i in range(len(perc)):
			plt.plot(win_t, quantile[:, i], mark[i], label=perc[i])

		plt.plot(win_t, rate_win)
		plt.title('Rates per time window')
		plt.xlabel('Time window (sec)')
		plt.ylabel('Precursor coincidence rate')
		plt.legend()
		plt.show()
	
	return (results)



if __name__ == '__main__': 
	import argparse

	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--input', type = str, 
		nargs = 1, required = False, 
		help = 'name of hdf5 input file with ICA-filtered timecourses')
	# ap.add_argument('-e', '--experiment', 
	#     nargs = 1, required = False, 
	#     help = 'name of experiment (YYYYMMDD_EE) for loading associated files.\
	#         Requires folder argument -f')
	ap.add_argument('-f', '--folder', 
		nargs = 1, required = False, 
		help = 'name of folder to load hdf5 file and save output hdf5 file')
	# ap.add_argument('-o', '--output', type = argparse.FileType('a+'),
	# 	nargs = 1, required = False, 
	# 	help = 'path to the output experiment file to be written.  '
	# 	'Must be .hdf5 file.')
	args = vars(ap.parse_args())

	data = hdf5manager(args['input'][0]).load()
	brain_data = data['brain']
	metadata = data['expmeta']
	name = metadata['name']

	print(brain_data.shape)
	eventMatrix, pMatrix = test_ROI_timecourse(brain_data)
	fileData = {"eventMatrix": eventMatrix, "pMatrix": pMatrix, "expmeta": metadata}
	saveData = hdf5manager(name + "_MatrixData_full.hdf5")
	saveData.save(fileData)