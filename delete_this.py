import numpy as np

index_map = []
for i in range(wanted_threads):
	index_map.extend(list(np.arange(i,brain_data.shape[0],wanted_threads)))

index_map = np.asarray(index_map)
inv_index_map = np.argsort(index_map)