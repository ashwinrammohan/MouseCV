import numpy as np
import matplotlib
from matplotlib import pyplot

#Method for detecting spikes in data given certain thresholds
#data is of the form: ((xvals), (yvals))
#d_threshold - threshold for change in derivative at a given point
#delta_threshold - threshold for delta between current value and moving average
def detectSpike(data, d_threshold = 5, delta_threshold = 5):
	xvals = data[0]
	yvals = data[1]

	ROC = 0
	movingAVG = 0
	done = False

	for i in range(1,len(xvals)-1):
		if (not done):
			leftX = xvals[i-1]
			rightX = xvals[i+1]

			leftY = yvals[i-1]
			rightY = yvals[i+1]

			newROC = (rightY - leftY)/(rightX - leftX) #symmetric difference quotient
			movingAVG = sum(yvals[0:i+1])/(i+1)
			#print(str(i) + "(" + str(xvals[i]) + "," + str(yvals[i]) + ")" + " newROC = " + str(newROC) + "Old ROC: " + str(ROC))

			if (newROC - ROC > d_threshold):
				print("Spike detected at x = " + str(xvals[i]) + " due to derivative change")
				print("Index: " + str(i) + " Old ROC: " + str(ROC) + ", New ROC: " + str(newROC))
				done = True
			elif (yvals[i]- movingAVG > delta_threshold):
				print("Spike detected at x = " + str(xvals[i]) + " 2")
				print("Index: " + str(i) + ": Y value: " + str(yvals[i]) + ", moving average: " + str(movingAVG))
				done = True
			else:
				ROC = newROC

xs = [1,2,3,4,5,6,7,8,9,10]
ys = [1,4,9,16,25,36,49,64,81,125]
data = [xs,ys]
detectSpike(data, 5, 300)





