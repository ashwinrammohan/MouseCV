import time
import numpy as np

a = []
for i in range(267):
	a.append([])
	for j in range(267):
		a[i].append([])
		for k in range(20):
			a[i][j].append(k)

start_time = time.clock()
a2 = np.asarray(a)
second_start = time.clock()
a2 = list(a2)
print(time.clock() - second_start)
#print(a2.shape)