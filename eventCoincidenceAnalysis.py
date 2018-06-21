import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *

#Method for detecting spikes in data given certain thresholds
#data is of the form: ((xvals), (yvals))
#delta_threshold - threshold for delta between current value and moving average
#stDev_threshold - if a y-value is greater than the mean by this threshold * stDev, that point is declared a spike
def detectSpike(data, interval = 20, stDev_threshold = 1.5):
	xvals = data[0]
	yvals = data[1]
	spikeIndices = set()
	localAvgs = []
	localSTDs = []
	
	mid_interval = int(interval/2)

	new_xvals = xvals[mid_interval:len(xvals) - mid_interval + 1]
	new_yvals = yvals[mid_interval:len(yvals) - mid_interval + 1]

	extra_data = []

	first_xvals = xvals[:mid_interval]
	first_yvals = yvals[:mid_interval]
	extra_data.append(first_xvals)
	extra_data.append(first_yvals)
	last_xvals = xvals[len(xvals) - mid_interval:]
	last_yvals = yvals[len(xvals) - mid_interval:]
	extra_data.append(last_xvals)
	extra_data.append(last_yvals)

	for i in range(0,len(extra_data),2): #only iterates over the x value lists
		curr_xlist = extra_data[i]
		curr_ylist = extra_data[i+1] #references the corresponding y values

		avg = np.mean(curr_ylist)
		localAvgs.append(avg)
		stDev = np.std(curr_ylist)
		localSTDs.append(stDev)

		for j in range(0, len(curr_xlist)):
			orig_index = xvals.index(curr_xlist[j])
			if (yvals[orig_index] - avg > stDev * stDev_threshold):
				print ("Spike detected at x = " + str(xvals[orig_index]))
				print("\n" + "Index: " + str(orig_index) + " y value: " + str(yvals[orig_index]) + ", local mean: " + str(avg) + ", local std: " + str(stDev) + "\n")
				spikeIndices.add(orig_index)

	localSTDev = 0
	localAVG = 0
	for i in range(0,len(new_xvals)):
		orig_index = xvals.index(new_xvals[i])
		before = orig_index-mid_interval
		after = orig_index+mid_interval

		std_subset = yvals[before:i]
		mean_subset = yvals[before:i]
		localAVG = np.mean(mean_subset)
		localAvgs.append(localAVG)
		localSTDev = np.std(std_subset)
		localSTDs.append(localSTDev)

		if (yvals[orig_index] - localAVG > localSTDev * stDev_threshold):
			#print("Spike detected at x = " + str(xvals[i]))
			#print("\n" + "Index: " + str(i) + " y value: " + str(yvals[i]) + ", local mean: " + str(localAVG) + ", local std: " + str(localSTDev) + "\n")

			spikeIndices.add(orig_index)

	return (spikeIndices, localAvgs, localSTDs)

data = hdf5manager("P2_timecourses.hdf5").load()
print("brain data below...")
ys = data['brain'][0][:2000]
print("Global Average: " + str(np.mean(ys)))
print("Global Stdev: " + str(np.std(ys)))

xs = list(np.linspace(0,2000,2000))

plot_data = [xs,ys]
spikes, avgs, stdevs = detectSpike(plot_data,100,3)

legend = ("Data", "Avgs", "Stdevs")
plt.plot(xs,ys)
plt.plot(xs,avgs)
plt.plot(xs, stdevs)
plt.legend(legend)


for i in spikes:
	#print("(" + str(xs[i]) + "," + str(ys[i]) + ")" + "\n")
	plt.axvline(x = xs[i], color = 'red')


plt.show()

'''
xs = [1,2,3,4,5,6,7,8,9,10,11,12]
ys = [1,4,9,16,25,36,49,64,81,100,121,144]

data = [xs,ys]
spikes = detectSpike(data,10,1)

for i in spikes:
	print("(" + str(xs[i]) + "," + str(ys[i]) + ")" + "\n")
	plt.axvline(x = xs[i], color = 'red')

plt.scatter(xs, ys)
plt.show()
'''





