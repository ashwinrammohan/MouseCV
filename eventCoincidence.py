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

'''
This is the function that runs on each thread. It runs eventCoin on its section of the data, 
keeps track of the results, and keeps track of various pieces of diagnostic information.
rowsLower and rowsUpper define the lower and upper bounds of where this function should analyze the data.
binarized_data is the data as a matrix, where the first dimension is the timecourse index, and the 
second dimension is the frame number. win_t is a matrix with the time windows that should be analyzed.
eventMatrix and pMatrix are special objects which allow this process to modify data that also exists in the main process. 
eventMatrix stores the coincidence rates, and pMatrix stores the p values.brain_data_shape is the shape of the brain data. 
fps is the frames per second of the data. dispDict is another special object which can be shared between
processes, and is where diagnostic information goes. name is the name of this process, for diagnostic purposes. 

'''
def _eventCoin(rowsLower, rowsUpper, binarized_data, win_t, eventMatrix, pMatrix, brain_data_shape, fps, dispDict, name):
	print("New thread created, running from " + str(rowsLower) + " to " + str(rowsUpper))
	numRows = brain_data_shape[0]
	avg_na = FixedQueue(20)
	avg_nb = FixedQueue(20)
	total_time = time.clock()
	avg_dt = FixedQueue(20)
	dt = time.clock()
	processed = 0
	needed = (rowsUpper - rowsLower) * numRows
	dispDict["needed_"+name] = needed
	disp_time = 0

	eventResults = np.empty((rowsUpper - rowsLower, numRows, win_t.shape[0])) # these matrices are made to hold values while the code is running
	pResults = np.empty((rowsUpper - rowsLower, numRows, win_t.shape[0])) # it creates a buffer which can pasted into eventMatrix and pMatrix once finished

	for i in range(rowsLower, rowsUpper):
		for j in range(numRows):
			
			if time.clock() - disp_time >= 1: # updates diagnostic information every second
				dispDict["i_"+name] = i
				dispDict["j_"+name] = j
				dispDict["avg_na_"+name] = avg_na.sum/20
				dispDict["avg_nb_"+name] = avg_nb.sum/20
				dispDict["total_time_"+name] = time.clock() - total_time
				dispDict["avg_dt_"+name] = avg_dt.sum/20
				dispDict["processed_"+name] = processed
				disp_time = time.clock()

			processed += 1

			if (i != j): # check that this data isn't being compared to itself
				bin_tcs1 = binarized_data[i]
				bin_tcs2 = binarized_data[j]
				# get coincidence rates for each time window
				event_data, na, nb = eventCoin(bin_tcs1,bin_tcs2, win_t=win_t, ratetype='precursor', verbose = False, veryVerbose = False)

				eventResults[i-rowsLower, j] = event_data # store rate
				# get p values for each time window
				pResults[i-rowsLower, j] = getResults(event_data, win_t=win_t, na=na, nb=nb, T = brain_data_shape[1]/fps, fps = fps, verbose = False, veryVerbose = False)

				# diagnostic info
				avg_dt.add_value(time.clock() - dt)
				dt = time.clock()
				avg_na.add_value(na)
				avg_nb.add_value(nb)
			else: # if the data is being compared to itself, just return NaNs
				eventResults[i-rowsLower, j] = np.NaN
				pResults[i-rowsLower, j] = np.NaN

	eventNp = np.frombuffer(eventMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0])) # extract numpy matrix from special objects
	pNp = np.frombuffer(pMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0]))

	eventNp[rowsLower:rowsUpper] = eventResults # put data in
	pNp[rowsLower:rowsUpper] = pResults

	# finish diagnostic info
	dispDict["done"] += 1
	dispDict["i_"+name] = i
	dispDict["j_"+name] = j
	dispDict["total_time_"+name] = time.clock() - total_time
	dispDict["processed_"+name] = processed


'''
This method takes brain data, and returns an eventMatrix, pMatrix, and preMatrix
The eventMatrix has coincidence rates for each component to each other.
The pMatrix has p values for each component to each other.
The preMatrix has True or False to indicate if each component is another's precursor
'''
def test_ROI_timecourse(brain_data, fps = 10,  max_window = 2, start_event = True, end_event = False, threads = 0, stDev_threshold = 0.8):
	binarized_data = np.zeros_like(brain_data).astype('uint8')
	numRows = brain_data.shape[0]
	start_spike_set = []
	end_spike_set = []
	win_t = np.arange((1/fps),max_window,(1/fps))

	print("Finding events...")

	# First, brain data is binarized using event detection
	for i, dataRow in enumerate(brain_data):
		binarizedRow = np.zeros_like(dataRow)

		start_time = time.clock()
		start_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3, peak_tolerance = 0.5) # check derivateEventDetection.py for more info on this
		print("Spikes at", i, "found in", (time.clock() - start_time), "seconds")
		if start_event:
			binarizedRow[start_spikes] = 1
		if end_event:
			binarizedRow[end_spikes] = 1

		binarized_data[i,:] = binarizedRow # binarized data put into main matrix

	if threads == 0: # if the user didn't specify the number of threads, use what is likely the best number
		wanted_threads = cpu_count()
	else:
		wanted_threads = threads


	threads = []
	print("Creating " + str(wanted_threads) + " threads...")

	eventMatrix = Array(c.c_double, numRows*numRows*win_t.shape[0]) # create inter-process variables
	pMatrix = Array(c.c_double, numRows*numRows*win_t.shape[0]) 
	displayDict = Manager().dict()
	displayDict["done"] = 0

	# the index map is used to distribute the brain data among the processes such that no process does much more work than any other
	# in this case, the data is already roughly sorted by processing time (was sorted like this from ICA analysis), so
	# a good index map is one which gives process 0 index 0, process 1 index 1, process 2 index 2, etc. 
	index_map = []
	for i in range(wanted_threads):
		index_map.extend(list(np.arange(i,brain_data.shape[0],wanted_threads)))

	index_map = np.asarray(index_map)
	inv_index_map = np.argsort(index_map) # the inverse index map will be used to unscramble the data
	print("Created empty data arrays")

	dataPer = int(numRows / wanted_threads) # number of rows of data per thread
	upper = 0
	names = []
	for i in range(wanted_threads): # create all threads
		name = "Thread " + str(i+1)
		names.append(name)

		displayDict["i_"+name] = 0 # initialize diagnostic display dictionary
		displayDict["j_"+name] = 0
		displayDict["avg_na_"+name] = 0
		displayDict["avg_nb_"+name] = 0
		displayDict["total_time_"+name] = 0
		displayDict["avg_dt_"+name] = 0
		displayDict["processed_"+name] = 0
		displayDict["needed_"+name] = 1

		if i == wanted_threads-1: # for the last thread, give it any leftover rows of data, for example 23 rows, 5 threads, this process will do #17-23
			p = Process(target=_eventCoin, args=(upper, numRows, binarized_data[index_map], win_t, eventMatrix, pMatrix, brain_data.shape, fps, displayDict, name))
		else: # otherwise just divide up the rows into each process normally
			lower = i*dataPer
			upper = (i+1)*dataPer
			p = Process(target=_eventCoin, args=(lower, upper, binarized_data[index_map], win_t, eventMatrix, pMatrix, brain_data.shape, fps, displayDict, name))
		p.start()
		threads.append(p)

	_displayInfo(displayDict, wanted_threads, names) # run the diagnostic windows on the main thread

	for p in threads: # insure that all threads have finished
	 	p.join()

	 print("All threads done")

	# get matrices from objects, reshape then unscramble it
	eventMatrix = np.frombuffer(eventMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0]))[inv_index_map][:,inv_index_map]
	pMatrix = np.frombuffer(pMatrix.get_obj()).reshape((numRows, numRows, win_t.shape[0]))[inv_index_map][:,inv_index_map]

	# create preMatrix using p-values
	cutoff = 0.00001
	preMatrix = np.zeros_like(pMatrix, dtype=bool)
	preMatrix[pMatrix < cutoff] = True
	preMatrix[pMatrix > 1 - cutoff] = True

	return eventMatrix, pMatrix, preMatrix

'''
Compares 2 binary sequences of data to see the percent of time an event in A happens 
before an event in B, within some time window.
'''
def eventCoin(a, b, #two binary signals to compare
			  win_t, #vector of time (s) for window
			  na = None, nb = None, #number of total events in each comparitive vector
			  ratetype = 'precursor', #precursor or trigger
			  tau = 0, #lag coefficeint 
			  fps = 10, #sampling rate, frames per sec
			  verbose = True,
			  veryVerbose = False):
	
	overall_time = time.clock()
	start_time = time.clock()
	
	#find all indices for each event
	a_ind = np.where(a != 0)[0]
	b_ind = np.where(b != 0)[0]

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

		start_time = time.clock()
		events = np.zeros((ind_diff.shape[0], len(win_fr)))
		ind_diff[ind_diff <= 0] = max(win_fr)+1 # put any values zero or less to a very large number so that they aren't picked up as coinciding
		results = np.zeros_like(ind_diff)

		for i, win in enumerate(win_fr):
			if verbose:
				print('Calculating PRECURSOR coincidence rate for window ' + str(win/fps) +'sec(s)')

			results[ind_diff < win] = 1 # set results to 1 whenever there's a coincidence
			row_sum = np.heaviside(np.sum(results, axis=1), 0) # if index in A ever coincided with an index B, return 1, otherwise 0
			events[:,i] = row_sum # put the result into the result matrix

		rate_win = np.sum(events, axis=0)/na # calculate the rate using coincidences

		if verbose:
			print("Took " + str(time.clock() - start_time) + " seconds.")

	if ratetype == 'trigger':
		
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

'''
Compares a given rate to similar random data to see if the rate is significant, or simpy due to random chance
'''
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

	# for the rate of each time window, use a poisson distribution to see if the data is relevant 
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

'''
This method is no longer used. It takes a lookup table and interpolates to find the p-value for a given rate.
'''
def pValForRate(lookup_table, rate, na, nb, win_t_index):
	if (na == 0 or nb == 0):
		print("Zeros,", rate)

	tbl_interval = lookup_table["interval"]
	data = lookup_table["table"]
	p_vals = lookup_table["p_values"]
	sz = data.shape[3] // 2

	na_lower = int(na // tbl_interval)
	na_upper = min(na_lower + 1, data.shape[0] - 1)

	nb_lower = int(nb // tbl_interval)
	nb_upper = min(nb_lower + 1, data.shape[1] - 1)

	lower_array = data[na_lower, nb_lower, win_t_index]
	upper_na_array = data[na_upper, nb_lower, win_t_index]
	upper_nb_array = data[na_lower, nb_upper, win_t_index]

	p_lower_iL = np.searchsorted(lower_array, rate, side='left') - sz
	p_lower_iR = np.searchsorted(lower_array, rate, side='right') - sz
	p_lower_i = sz + ( p_lower_iL if abs(p_lower_iL) < abs(p_lower_iR) else p_lower_iR )
	p_lower = listInterp(lower_array, p_vals, p_lower_i, data.shape[3] - 1, rate)

	p_upper_na_iL = np.searchsorted(upper_na_array, rate, side='left') - sz
	p_upper_na_iR = np.searchsorted(upper_na_array, rate, side='right') - sz
	p_upper_na_i = sz + (p_upper_na_iL if abs(p_upper_na_iL) < abs(p_upper_na_iR) else p_upper_na_iR )
	p_upper_na = listInterp(upper_na_array, p_vals, p_upper_na_i, data.shape[3] - 1, rate)

	p_upper_nb_iL = np.searchsorted(upper_nb_array, rate, side='left') - sz
	p_upper_nb_iR = np.searchsorted(upper_nb_array, rate, side='right') - sz
	p_upper_nb_i = sz + ( p_upper_nb_iL if abs(p_upper_nb_iL) < abs(p_upper_nb_iR) else p_upper_nb_iR )
	p_upper_nb = listInterp(upper_nb_array, p_vals, p_upper_nb_i, data.shape[3] - 1, rate)

	p1 = np.empty(3)
	p1[0] = na_lower
	p1[1] = nb_lower
	p1[2] = p_lower

	p2 = np.empty(3)
	p2[0] = na_upper
	p2[1] = nb_lower
	p2[2] = p_upper_na

	p3 = np.empty(3)
	p3[0] = na_lower
	p3[1] = nb_upper
	p3[2] = p_upper_nb

	v1 = p3 - p1
	v2 = p2 - p1
	crs = np.cross(v1, v2) # ax + by + cz = d
	a, b, c = crs
	d = np.sum(np.multiply(p1, crs))

	# (d - ax - by) / c = z
	p_val = (d - a*na/tbl_interval - b*nb/tbl_interval) / c

	return p_val

def linearInterp(x1, x2, y1, y2, x):
	dx = x2 - x1
	dy = y2 - y1

	if dx == 0:
		return (y1 + y2) / 2
	else:
		t = (x - x1) / dx
		return dy * t + y1

def listInterp(xList, yList, lowerIndex, max_index, value):
	lowerIndex = min(lowerIndex, max_index-1)
	upperIndex = lowerIndex + 1

	xStart = xList[lowerIndex]
	xEnd = xList[upperIndex]
	yStart = yList[lowerIndex]
	yEnd = yList[upperIndex]

	# print("Rate:", value, "Rate - Lower:", xStart, "Upper:", xEnd, "P-Val - Lower:", yStart, "Upper:", yEnd)
	# print("Indices - Lower:", lowerIndex, "Upper:", upperIndex)

	return linearInterp(xStart, xEnd, yStart, yEnd, value)

# Show all windows with diagnostic info
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

# Code for 1 diagnostic window
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
	from eventCharacterization import *

	ap = argparse.ArgumentParser()
	ap.add_argument('-f', '--filename', type = str, nargs = 1, required = True, help = 'name of hdf5 input file with ICA-filtered timecourses')
	ap.add_argument('-i', '--i', type = int, nargs = 1, required = False, help = 'index of specific timecourse')
	ap.add_argument('-j', '--j', type = int, nargs = 1, required = False, help = 'index of other specific timecourse')
	ap.add_argument('-g', '--graphs', action = "store_true", required = False, help = 'display graphs after completing')
	#ap.add_argument('-t', '--table', type = str, nargs = 1, required=True, help = 'file location of lookup table for p values')

	args = vars(ap.parse_args())

	data = hdf5manager(args['filename'][0]).load()
	#table = hdf5manager(args['table'][0]).load()

	if "ROI_timecourses" in data.keys():
		brain_data = data['ROI_timecourses']
	elif "brain" in data.keys():
		brain_data = data["brain"]
	else:
		print("No data found! Maybe the hdf5 is formatted incorrectly?")
		import sys
		sys.exit()

	print(brain_data.shape)

	if args['i'] is not None: # this option allows the user to find coincidence for a specific component against another specific component
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
		table = None
		eventMatrix, pMatrix, preMatrix = test_ROI_timecourse(brain_data, table)
		fileData = {"eventMatrix": eventMatrix, "pMatrix": pMatrix, "precursors":preMatrix}
		fileString = ""
		if ("expmeta" in data.keys()):
			fileString = data['expmeta']['name']
		else:
			fileString = args['filename'][0].split("_")[0]

		fileString = "Outputs/" + fileString + "_MatrixData_full.hdf5"
		saveData = hdf5manager(fileString)
		saveData.save(fileData)
		print("Saved event coincidence data to " + fileString)

		if args["graphs"]:
			eventGraphing(fileString, dataFile = data)
