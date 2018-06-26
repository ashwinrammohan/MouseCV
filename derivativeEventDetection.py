import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
from scipy.stats import poisson
import cv2 as cv
import sys

path_file = open("path.txt", "r")
sys.path.append(path_file.read())
path_file.close()

from timecourseAnalysis import butterworth

# This class is an extremely efficient queue with a fixed size. It keeps track of the
# sum of its values.
class FixedQueue:
	# Creates a FixedQueue with the given size. User can give starting values if they want, 
	# but the values must be the correct length
	def __init__(self, size, values=[]): 
		assert(size > 0), "Queue size must be more than 0"
		if len(values) == size:
			self.values = values
		else:
			self.values = [0.0] * size # fills with zeros

		self.size = size
		self.head = 0
		self.sum = 0

	# Adds a value to the queue, and removes the oldest value if the queue is full
	def add_value(self, value):
		self.sum -= self.values[self.head]
		self.sum += value # update the sum

		self.values[self.head] = value

		self.head += 1
		if self.head >= self.size: # if we've reached the end of the list, go to the beginning
			self.head = 0

# Loads an HDF5 file with an assumed structure. The dictionary produced must have up to 6 kv pairs,
# with the names footFL, footFR, footBL, footBR, head, tail. Not all kv pairs need to exist.
# The HDF5 file must be stored in the Assets folder.
def loadHDF5(file_name):
	data = hdf5manager("Assets/" + file_name + ".hdf5").load()

	def loadLimb(limbName, pos, color):
		if not(limbName in data.keys()): # check if this kv pair actually exists
			return

		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(data[limbName]["magnitude"], -0.3) # find events in magnitude data
		data[limbName]["pos"] = pos # position of the vector for graphing purposes
		data[limbName]["color"] = color # color of the vector when graphing
		data[limbName]["start_spikes"] = start_spikes
		data[limbName]["mid_spikes"] = mid_spikes
		data[limbName]["end_spikes"] = end_spikes

	loadLimb("footFL", (0,1), (1, 0, 0))
	loadLimb("footFR", (0,-1), (0, 1, 0))
	loadLimb("footBL", (-1,1), (1, 0.7, 0.3))
	loadLimb("footBR", (-1,-1), (0, 0, 1))
	loadLimb("head", (1,0), (1, 0.3, 1))
	loadLimb("tail", (-2,0), (0.3, 1, 1))
	return data

# Uses first and second derivatives to find peaks in data. `second_deriv_thresh` is the threshold for the second derivative of the data.
# Since the peaks are downward facing, this value should be negative, and any second derivative less negative than this value will not be considered a peak.
# `second_deriv_batch` is the size of the buffer of derivatives that are stored, both before and after the current data point being evaluated. 
# It is essentially an average, so a larger batch size will make wider peaks easier to detect, and reduce noise, but possibly miss narrow peaks.
# `deriv_start` is the minimum value of the first derivative that counts as the beginning of the event. This should rarely be changed
# `peak_height` is the minimum height of a peak. It will be calculated automatically, but the user may override the automatic calculation if they want.
# `high_pass` is the value passed to the butterworth filter which cuts off high frequencies. The number itself must range from 0 to 1, but 
# what the numbers actually means is unclear. Regardless, a lower value will cut off more frequencies.
def detectSpikes(data, second_deriv_thresh, second_deriv_batch = 3, deriv_start = 0, peak_height = 0, high_pass = 0.5):
	data = butterworth(data, high = high_pass, low = None) # smooth/de-noise the data

	past_deriv = 0
	past_derivs = FixedQueue(second_deriv_batch)
	future_dervis = FixedQueue(second_deriv_batch)
	spikes = [] # the top of each spike will be stored here
	if (peak_height == 0):
		peak_height = np.std(data)/3 # if no peak_height is given, the standard deviation is used to guess a good min height
		#("Min peak height: " + str(peak_height))

	# initialize past batch
	for i in range(1, second_deriv_batch+1):
		deriv = data[i] - data[i-1]
		past_deriv = deriv
		past_derivs.add_value(deriv)

	# initialize future batch
	for i in range(second_deriv_batch+2, 2*second_deriv_batch+2):
		deriv = data[i] - data[i-1]
		future_dervis.add_value(deriv)

	# calculate spikes
	for i in range(second_deriv_batch+1, len(data) - second_deriv_batch-1):
		deriv = data[i] - data[i-1] # first derivative

		deriv_test = (deriv < 0 and past_deriv >= 0) # whether there is a peak of any kind (positive slope to negative slope)
		second_deriv_test = ((future_dervis.sum - past_derivs.sum) < second_deriv_thresh) # whether the peak is extreme enough 
		if deriv_test and second_deriv_test:
			spikes.append(i)

		past_deriv = deriv
		past_derivs.add_value(deriv) # update past derivs

		deriv = data[i+second_deriv_batch+1] - data[i+second_deriv_batch]
		future_dervis.add_value(deriv) # update future derivs

	# find activity duration
	i = 0
	start_spikes = [] # will hold the beginning of each spike, using the `deriv_start` parameter
	mid_spikes = [] # will hold all indicies within activities. This is mostly used for graphing
	for j in range(len(spikes)-1, -1, -1): # iterate backwards so that spikes can be removed without issue
		spike = spikes[j]
		i = spike - 1
		done = False
		while not(done): # walk backwards on the data until the beginning of the activity is detected
			deriv = data[i] - data[i-1]
			if (deriv <= deriv_start):
				done = True
			else:
				i -= 1
				if (i < 1): # if you reach the start of the data, it also counts as the beginning of the activity
					done = True

		if data[spike] - data[i] < peak_height: # check whether the total height of the peak is enough
			del spikes[j]
		else:
			start_spikes.append(i)
			for h in range(spike-1, i, -1):
				mid_spikes.append(h)

	return (start_spikes, mid_spikes, spikes, data)

def test_ROI_timecourse(brain_data, start_event = False, mid_event = False, end_event = False):
	binarized_data = np.zeros_like(brain_data)
	numRows = brain_data.shape[0]
	start_spike_set = []
	mid_spike_set = []
	end_spike_set = []
	eventMatrix = np.zeros((20,numRows,numRows))

	for i in range(brain_data.shape[0]):
		dataRow = brain_data[i]
		binarizedRow = np.zeros_like(dataRow)
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(dataRow, -0.3)
		for j in range(dataRow.shape[0]):
			if j in start_spikes or j in mid_spikes or j in end_spikes:
				binarizedRow[j] = 1
		binarized_data[i,:] = binarizedRow

	win_t = np.arange(0,2,1/10)
	for i in range(numRows):
		for j in range(numRows):
			if (i != j):
				bin_tcs1 = binarized_data[i]
				bin_tcs2 = binarized_data[j]
				rand, na, nb = eventCoin(bin_tcs1,bin_tcs2, win_t=win_t, ratetype='precursor', verbose = False, veryVerbose = False)
				eventMatrix[:,i,j] = rand
				rand_results = getResults(rand, win_t=win_t, na=na, nb=nb, verbose = False, veryVerbose = False)

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


def test_amplitude():
	data = loadHDF5("mouse_vectors")

	for limbKey in data.keys():
		plt.figure("Limb: " + limbKey)
		xs = list(np.linspace(0,len(data[limbKey]["magnitude"]),len(data[limbKey]["magnitude"])))
		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(data[limbKey]["magnitude"], -0.1, second_deriv_batch=8, high_pass = 0.75, peak_height=0.25)

		plt.plot(xs,vals)
		for i in start_spikes:
			plt.axvline(x = i, color = 'red')
		for i in mid_spikes:
			plt.axvline(x = i, color = (1,1,0,0.3))
		for i in end_spikes:
			plt.axvline(x = i, color = 'red')

	plt.show()


def eventCoin(a, b, #two binary signals to compare
              win_t, #vector of time (s) for window
              ratetype = 'precursor', #precursor or trigger
              tau = 0, #lag coefficeint 
              fps = 10, #sampling rate, frames per sec
              verbose = True,
              veryVerbose = False):
    
    na = sum(a)
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

    #replace all neg values with 0
    ind_diff[ind_diff< 0] = 0
    
    if veryVerbose:
        print('Size of difference array: ',ind_diff.shape)
        plt.imshow(ind_diff)
        plt.title('Difference in indices')
        plt.xlabel('indices from list a')
        plt.ylabel('indices from list b')
        plt.colorbar()
        plt.show()

    #create event matrix
    events = np.zeros((ind_diff.shape[0], len(win_fr)))
    
    for i, win in enumerate(win_fr):
        if verbose:
            print('Calculating coincidence rate for window ' + str(win/fps) +'sec(s)')
        for j in range(ind_diff.shape[0]):
            for k in range(ind_diff.shape[1]):
                if ind_diff[j,k] > 0 and ind_diff[j,k] < win:
                    events[j, i] = 1
                    
    if ratetype == 'precursor':
        rate_win = np.sum(events, axis=0)/na
        
    if ratetype == 'trigger':
        rate_win = np.sum(events, axis=0)/nb
    
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
    rho = 1 - win_t/(T - tau)
    exp_rate = na*(1 - (rho)**nb)
    exp_std = np.sqrt(1/na*(1-rho**nb) * rho**nb)
    
    #quantiles used for graphing
    if verbose:
        perc = np.array([1, 2.5, 25, 50, 75, 97.5, 99])
        mark = ['k:', 'k-.', 'k--', 'k-', 'k--','k-.', 'k:']
    #number samples for null hypothesis
    k=10000

    sample = np.zeros((exp_rate.shape[0], k))
    quantile = np.zeros((exp_rate.shape[0], perc.shape[0]))
    results = np.zeros(exp_rate.shape[0])

    for i, r in enumerate(exp_rate):
        sample[i,:] = poisson.rvs(r, size=k)
        quantile[i,:] = np.percentile(sample[i,:], perc)/na
        results[i] = sum(rate_win[i] < sample[i, :]/na)/k
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

data = hdf5manager("P2_timecourses.hdf5").load()
brain_data = data['brain'][:2,:]
print(brain_data.shape)
test_ROI_timecourse(brain_data)