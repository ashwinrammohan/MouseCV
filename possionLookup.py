import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
from scipy.stats import poisson

perc = np.array([1, 2.5, 25, 50, 75, 97.5, 99])
samples = 10000
tau = 0
fps = 10
max_window = 2
T = 600

def load_lookup():
	data = hdf5manager("poisson_lookup_T" + str(T) + ".hdf5").load()
	return data["data"]

def save_lookup(dictionary):
	loc = hdf5manager("poisson_lookup_T" + str(T) + ".hdf5")
	loc.save(dictionary)

def generate_lookup(windows, lower_n, upper_n):
	arr = np.empty((len(windows), upper_n - lower_n, upper_n - lower_n, perc.shape[0]))
	for w, window in enumerate(windows):
		print("Computing window #"+str(w+1)+"...")
		for na in range(lower_n, upper_n):
			print(str(round((na-lower_n)/(upper_n-lower_n)*100)) + "% done...")
			for nb in range(lower_n, upper_n):
				rho = 1 - window/(T - tau)
				exp_rate = na * (1 - (rho) ** nb)
				arr[w, (na-lower_n), (nb-lower_n)] = np.percentile(poisson.rvs(exp_rate, size=samples), perc) / na

	return arr

arr = generate_lookup(np.arange((1/fps), max_window, (1/fps)), 10, 130)
save_lookup({"data":arr})