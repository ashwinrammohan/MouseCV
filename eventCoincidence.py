import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
from scipy.stats import poisson
from derivativeEventDetection import detectSpikes, FixedQueue
import time
from multiprocessing import Process, Array, cpu_count, Manager
import ctypes as c
import cv2 as cv
from eventCharacterization import *

def _eventCoin(rowsLower, rowsUpper, numRows, binarized_data, win_t, eventMatrix, pMatrix, brain_data, fps, dispDict, name, graph = False):
	print("New thread created, running from " + str(rowsLower) + " to " + str(rowsUpper))
	avg_na = FixedQueue(20)
	avg_nb = FixedQueue(20)
	total_time = time.clock()
	avg_dt = FixedQueue(20)
	dt = time.clock()
	processed = 0
	needed = (rowsUpper - rowsLower) * numRows
	dispDict["needed_"+name] = needed
	disp_time = 0

	eventResults = np.empty((rowsUpper - rowsLower, numRows, win_t.shape[0]))
	pResults = np.empty((rowsUpper - rowsLower, numRows, win_t.shape[0]))

	for i in range(rowsLower, rowsUpper):
		for j in range(numRows):
			
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

			if (i != j):
				bin_tcs1 = binarized_data[i]
				bin_tcs2 = binarized_data[j]
				event_data, na, nb = eventCoin(bin_tcs1,bin_tcs2, win_t=win_t, ratetype='precursor', verbose = False, veryVerbose = False)

				if graph:
					plt.plot(bin_tcs1, 'bo'), plt.title("Index: " + str(i)), plt.show()

				eventResults[i-rowsLower, j] = event_data
				pResults[i-rowsLower, j] = getResults(event_data, win_t=win_t, na=na, nb=nb, T = brain_data.shape[1]/fps, fps = fps, verbose = False, veryVerbose = False)

				avg_dt.add_value(time.clock() - dt)
				dt = time.clock()
				avg_na.add_value(na)
				avg_nb.add_value(nb)
			else:
				eventResults[i-rowsLower, j] = np.NaN
				pResults[i-rowsLower, j] = np.NaN

	eventNp = np.frombuffer(eventMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0]))
	pNp = np.frombuffer(pMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0]))

	eventNp[rowsLower:rowsUpper] = eventResults
	pNp[rowsLower:rowsUpper] = pResults
	dispDict["done"] += 1
	dispDict["i_"+name] = i
	dispDict["j_"+name] = j
	dispDict["total_time_"+name] = time.clock() - total_time
	dispDict["processed_"+name] = processed


def test_ROI_timecourse(brain_data, fps = 10,  max_window = 2, start_event = True, end_event = True, threads = 0, stDev_threshold = 0.8):
	numRows = brain_data.shape[0]
	spikes = []
	win_t = np.arange((1/fps),max_window,(1/fps))
	max_events = 0

	print("Finding events...")

	for i, dataRow in enumerate(brain_data):
		start_time = time.clock()
		start_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3, peak_tolerance = 0.5)
		print("Spikes at", i, "found in", (time.clock() - start_time), "seconds")

		spikes.append([])
		if start_event:
			spikes[-1].append(start_spikes)
		if end_event:
			spikes[-1].append(end_spikes)

		num_events = start_spikes.shape[0] + end_spikes.shape[0]
		if num_events > max_events:
			max_events = num_events

	np_spikes = np.empty((numRows, max_events))
	np_spikes.fill(np.NaN)
	for i, events in enumerate(spikes):
		np_spikes[i][:events.shape[0]]

	if threads == 0:
		wanted_threads = cpu_count()
	else:
		wanted_threads = threads


	threads = []
	print("Creating " + str(wanted_threads) + " threads...")

	eventMatrix = Array(c.c_double, numRows*numRows*win_t.shape[0])
	pMatrix = Array(c.c_double, numRows*numRows*win_t.shape[0]) 
	displayDict = Manager().dict()
	displayDict["done"] = 0
	index_map = []
	for i in range(wanted_threads):
		index_map.extend(list(np.arange(i,brain_data.shape[0],wanted_threads)))

	index_map = np.asarray(index_map)
	inv_index_map = np.argsort(index_map)
	print("Created empty data arrays")

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
			p = Process(target=_eventCoin, args=(upper, numRows, numRows, binarized_data[index_map], win_t, eventMatrix, pMatrix, brain_data, fps, displayDict, name))
		else:
			lower = i*dataPer
			upper = (i+1)*dataPer
			p = Process(target=_eventCoin, args=(lower, upper, numRows, binarized_data[index_map], win_t, eventMatrix, pMatrix, brain_data, fps, displayDict, name))
		p.start()
		threads.append(p)

	_displayInfo(displayDict, wanted_threads, names)
	print("All threads done")

	# for p in threads:
	# 	p.join()

	eventMatrix = np.frombuffer(eventMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0]))[inv_index_map][:,inv_index_map]
	pMatrix = np.frombuffer(pMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0]))[inv_index_map][:,inv_index_map]

	#print(eventMatrix[:,:,9].shape)

	#plt.imshow("10th time window", eventMatrix[:,:,9])
	#plt.colorbar()
	#plt.show()

	#print(eventMatrix[:,:,7])
	#print(pMatrix[:,:,7])

	cutoff = 0.0000001 #1% cutoff for p values
	preMatrix = np.zeros_like(pMatrix, dtype=bool)
	preMatrix[pMatrix < cutoff] = True
	preMatrix[pMatrix > 1 - cutoff] = True

	return eventMatrix, pMatrix, preMatrix

def eventCoin(a_ind, b_ind, #indeces of events of two signals to compare
			  win_t, #vector of time (s) for window
			  na = None, nb = None, #number of total events in each comparitive vector
			  ratetype = 'precursor', #precursor or trigger
			  tau = 0, #lag coefficeint 
			  fps = 10, #sampling rate, frames per sec
			  verbose = True,
			  veryVerbose = False):
	
	overall_time = time.clock()
	start_time = time.clock()

	if na == None:
		na = a_ind.shape[0]
	if nb == None:
		nb = b_ind.shape[0]

	if na == 0 or nb == 0:
		return np.repeat(np.NaN, len(win_t)), na, nb
	
	#convert window times to window frames
	win_fr = win_t * fps 
	
	#create index difference matrix
	ind_diff = np.zeros((len(a_ind), len(b_ind)))
	for i,inda in enumerate(a_ind): # rows
		ind_diff[i] = inda - (tau*fps) - b_ind

	#replace all neg values with 0	
	if veryVerbose:
		print('Size of difference array: ', ind_diff.shape)
		plt.imshow(ind_diff)
		plt.title('Difference in indices')
		plt.xlabel('indices from list a')
		plt.ylabel('indices from list b')
		plt.colorbar()
		plt.show()

	if verbose:
		print("Setup time: " + str(time.clock() - start_time))

	#create event matrix
	if ratetype == 'precursor':
		#print('Calculating PRECURSOR coincidence \n ----------------------------------')

		start_time = time.clock()
		events = np.zeros((ind_diff.shape[0], len(win_fr)))
		ind_diff[ind_diff <= 0] = max(win_fr)+1
		results = np.zeros_like(ind_diff)

		for i, win in enumerate(win_fr):
			if verbose:
				print('Calculating PRECURSOR coincidence rate for window ' + str(win/fps) +'sec(s)')

			results[ind_diff < win] = 1
			row_sum = np.heaviside(np.sum(results, axis=1), 0)
			events[:,i] = row_sum

		rate_win = np.sum(events, axis=0)/na

		if verbose:
			print("Took " + str(time.clock() - start_time) + " seconds.")

	if ratetype == 'trigger':
		#print('Calculating TRIGGER coincidence \n ----------------------------------')
		
		start_time = time.clock()
		events = np.zeros((ind_diff.shape[1], len(win_fr)))
		ind_diff[ind_diff >= 0] = -(max(win_fr) + 1)
		ind_diff = -ind_diff
		results = np.zeros_like(ind_diff)

		for i, win in enumerate(win_fr):
			if verbose:
				print('Calculating TRIGGER coincidence rate for window ' + str(win/fps) +'sec(s)')

			results[ind_diff < win] = 1
			row_sum = np.heaviside(np.sum(results, axis=0), 0)
			events[:,i] = row_sum

		rate_win = np.sum(events, axis=0)/nb
		if verbose:
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

	if verbose:
		print("Took " + str(time.clock() - overall_time) + " seconds total.")
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
	
	if na == 0 or nb == 0:
		return np.ones(win_t.shape[0])

	start_time = time.clock()
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
		perc = np.array([1, 2.5, 25, 50, 75, 97.5, 99])/100
		mark = ['k:', 'k-.', 'k--', 'k-', 'k--','k-.', 'k:']
		quantile = np.zeros((exp_rate.shape[0], perc.shape[0]))

	results = np.zeros(exp_rate.shape[0])

	for i, r in enumerate(exp_rate):
		if ratetype == 'precursor':
			if verbose:
				quantile[i,:] = poisson.ppf(perc, r)/na
			results[i] = poisson.cdf(rate_win[i]*na,r)
		if ratetype == 'trigger':
			if verbose:
				quantile[i,:] = poisson.ppf(perc, r)/nb
			results[i] = poisson.cdf(rate_win[i]*nb,r)
		if veryVerbose:
			print(str(win_t[i]) + 'sec(s) time window produces a p value: ' + str(results[i]))

	if verbose:
		for j, r in enumerate(results):
			if r < 0.05:
				print(str(win_t[j]) + 'sec(s) time window produces a significant value: p=' + str(r))
	
		for i in range(len(perc)):
			plt.plot(win_t, quantile[:, i], mark[i], label=perc[i])

		plt.plot(win_t, rate_win)
		plt.title('Rates per time window')
		plt.xlabel('Time window (sec)')
		plt.ylabel('Precursor coincidence rate')
		plt.legend()
		plt.show()
	
	if veryVerbose:
		print("Elapsed time: " + str(time.clock() - start_time))

	return results

def _displayInfo(displayDict, wanted_threads, names):
	print("------ DISPLAYING INFO ------")
	positions = [(0, 0), (500, 0), (1000, 0), (0, 500), (500, 500), (1000, 500), (0, 1000), (500, 1000)]

	movewindows = True
	while displayDict["done"] < wanted_threads:
		for i, name in enumerate(names):
			visualizeProgress(name, displayDict["i_"+name], displayDict["j_"+name], displayDict["avg_na_"+name], displayDict["avg_nb_"+name] , displayDict["total_time_"+name] , displayDict["avg_dt_"+name] , displayDict["processed_"+name] , displayDict["needed_"+name], positions[i])
			if movewindows:
				cv.moveWindow(name, positions[i][0], positions[i][1])

		cv.waitKey(1000)
		movewindows = False
	cv.destroyAllWindows()

def visualizeProgress(window_name, i, j, avg_na, avg_nb, time_elapsed, avg_dt, processed, needed, pos):
	img = np.zeros((415, 460))
	cv.putText(img, "i:"+str(i), (25, 50), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
	cv.putText(img, "j:"+str(j), (135, 50), cv.FONT_HERSHEY_SIMPLEX, 1, 1)

	cv.putText(img, "avg na   avg nb", (25, 105), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
	cv.putText(img, str(int(avg_na)), (40, 140), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
	cv.putText(img, str(int(avg_nb)), (195, 140), cv.FONT_HERSHEY_SIMPLEX, 1, 1)

	cv.putText(img, "Total Time: %.3f min"%(time_elapsed/60), (25, 200), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
	cv.putText(img, "Average dt: %.6f sec"%(avg_dt), (25, 235), cv.FONT_HERSHEY_SIMPLEX, 1, 1)

	cv.putText(img, "Progress:", (25, 300), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
	cv.rectangle(img, (25, 315), (400, 350), (255, 255, 255), 1)
	cv.rectangle(img, (25, 315), (int(375 * processed / needed) + 25, 350), (255, 255, 255), -1)
	cv.putText(img, "(" + str(processed) + "/" + str(needed) + ")", (25, 385), cv.FONT_HERSHEY_SIMPLEX, 1, 1)

	cv.imshow(window_name, img)


if __name__ == '__main__': 
	import argparse

	ap = argparse.ArgumentParser()
	ap.add_argument('-f', '--filename', type = str, nargs = 1, required = True, help = 'name of hdf5 input file with ICA-filtered timecourses')
	ap.add_argument('-i', '--i', type = int, nargs = 1, required = False, help = 'index of specific timecourse')
	ap.add_argument('-j', '--j', type = int, nargs = 1, required = False, help = 'index of other specific timecourse')
	ap.add_argument('-g', '--graphs', action = "store_true", required = False, help = 'display graphs after completing')

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

	print(brain_data.shape)

	if args['i'] is not None:
		data_i = args['i'][0]
		data_j = args['j'][0]

		binarized_data = np.zeros_like(brain_data).astype('uint8')
		numRows = brain_data.shape[0]
		win_t = np.arange((1/10),2,(1/10))

		print("Finding events...")

		for i, dataRow in enumerate(brain_data):
			binarizedRow = np.zeros_like(dataRow)

			start_time = time.clock()
			start_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3, peak_tolerance = 0.5)
			print("Spikes at", i, "found in", (time.clock() - start_time), "seconds")
			if True:
				binarizedRow[start_spikes] = 1
			if True:
				binarizedRow[end_spikes] = 1

			binarized_data[i,:] = binarizedRow
		plt.plot(binarized_data[data_i]), plt.title(data_i)
		plt.show()
		plt.plot(binarized_data[data_j]), plt.title(data_j)
		plt.show()
		event_data, na, nb = eventCoin(binarized_data[data_i], binarized_data[data_j], win_t=win_t, ratetype='precursor', verbose = False, veryVerbose = False)
		plt.plot(event_data)
		plt.show()
	else:
		eventMatrix, pMatrix, preMatrix = test_ROI_timecourse(brain_data)
		fileData = {"eventMatrix": eventMatrix, "pMatrix": pMatrix, "precursors":preMatrix}
		fileString = ""
		if ("expmeta" in data.keys()):
			fileString = data['expmeta']['name']
		else:
			fileString = args['filename'][0].split("_")[0]

		fileString = "Outputs/" + fileString + "_MatrixData_full.hdf5"
		saveData = hdf5manager(fileString)
		saveData.save(fileData)
		print("Saved event coincidence data to Outputs/" + fileString + "_MatrixData_full.hdf5")

		if args["graphs"] is not None:
			eventGraphing(fileString, dataFile = data)
