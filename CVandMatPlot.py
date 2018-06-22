import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
import wholeBrain as wb
import matplotlib.figure
import cv2 as cv
import math
import datetime
from derivativeEventDetection import *


def load_avi(vid_name):
	cap = cv.VideoCapture("Assets/" + vid_name + ".avi")
	frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
	frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
	print(str(frameCount) + " frames in source video")

	vid = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))

	fc = 0
	ret = True

	while (True):
		result = cap.read()
		if not(result[0]):
			break

		vid[fc] = result[1][:,:,0]
		fc += 1

	cap.release()
	return vid

def drawMatplotFrame(figure, frameIndex, data):
	parseAndDrawHDF5(figure, data, frameIndex)
	drawFigureForRange(figure, data, frameIndex, 15)

#Draws a graph showing position vector magnitude over time for each of the limbs (each limb's
#data is shown in the color assigned to it in the loadLimb method). Because each limb's data
#spans multiple frames, the graph shows "rolling data" where for each frame, a certain interval
#before and after it is shown. A vertical red(blue) line at x = current frame is plotted so that it is
#clear to the viewer where the current frame's motion is being graphed. Each limb also contains data
#for where the start, duration, and end of events in the vector magnitude occur. Vertical lines
#are drawn in red (blue) for the start and end of each event, and the spikes in between are shown
#by yellow lines.
def drawFigureForRange(figure, data, frame, range):
	sub = figure.add_subplot(212)

	for limbKey in data.keys():
		limb = data[limbKey]
		bottom = max(0, frame - range)
		top = min(len(limb["magnitude"]), frame + range)
		sub.plot(np.arange(bottom, top), limb["magnitude"][bottom:top], '-', color=limb["color"])
		for i in limb["start_spikes"]:
			if i < frame - range:
				continue
			if i > frame + range:
				break
			sub.axvline(x = i - frame, color = 'red')

		for i in limb["mid_spikes"]:
			if i < frame - range:
				continue
			if i > frame + range:
				break
			sub.axvline(x = i - frame, color = (1,1,0,0.3))

		for i in limb["end_spikes"]:
			if i < frame - range:
				continue
			if i > frame + range:
				break
			sub.axvline(x = i - frame, color = 'red')

	sub.axvline(x=frame, color="red") #really a blue line b/c of OpenCV - Matplotlib changes in color formatting

	#sets axes labels and limits for the subplot
	sub.set_xlabel("Frame #")
	sub.set_ylabel("Magnitude")
	sub.set_xlim(frame - range, frame + range)
	sub.set_ylim(0, 10)


def arrayFromFigure(figure):
	fig.canvas.draw()
	width, height = fig.get_size_inches() * fig.get_dpi()
	image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
	image = image.reshape(int(height), int(width), 3)
	#image = image[...,::-1] #switch red and blue channels because matplot is RGB but CV2 is BGR
	return image

def loadHDF5(file_name):
	data = hdf5manager("Assets/" + file_name + ".hdf5").load()

	def loadLimb(limbName, pos, color):
		if not(limbName in data.keys()):
			return

		start_spikes, mid_spikes, end_spikes, vals = detectSpikes(data[limbName]["magnitude"], -0.05, second_deriv_batch=10, high_pass = 0.3)
		data[limbName]["pos"] = pos
		data[limbName]["color"] = color
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

#Parses the .hdf5 file for the position and velocity vectors for each limb
#Then, this method draws arrows (vectors) on the subplots with the head of the vectors starting at
#the limb's designated position and the tails of the vectors corresponding to the direction of the
#limb's motion.
def parseAndDrawHDF5(figure, data, frameIndex):
	sub = figure.add_subplot(211)
	for limbKey in data.keys():
		limb = data[limbKey]
		sub.arrow(limb["pos"][0], limb["pos"][1], limb["dx"][frameIndex], limb["dy"][frameIndex], color=limb["color"])
	#sub.quiver(x0s, y0s, dxs, dys, angles='xy', scale_units='xy', scale=1)

	#Sets axes labels and limits for the subplot
	sub.set_xlabel('x')
	sub.set_ylabel('y')
	sub.set_xlim(-5,5)
	sub.set_ylim(-5,5)
		
movie_name = "paint3Contours"
data_name = "mouse_vectors"
source_movie = load_avi(movie_name)
new_movie = np.zeros((source_movie.shape[0], 700, 1200, 3), dtype="uint8")
data = loadHDF5(data_name)

print(str(len(data["footFR"]["x"])) + " entries in vectorized data")

fig = plt.figure(figsize=(4,6))

for i, frame in enumerate(new_movie[1:]):
	sourceShape = source_movie.shape
	frame[:sourceShape[1],:sourceShape[2],0] = source_movie[i]
	frame[:sourceShape[1],:sourceShape[2],1] = source_movie[i]
	frame[:sourceShape[1],:sourceShape[2],2] = source_movie[i]

	drawMatplotFrame(fig, i, data)
	matplotArray = arrayFromFigure(fig)
	matplotShape = matplotArray.shape

	xstart = frame.shape[1] - matplotShape[1]
	xend = frame.shape[1]
	frame[:matplotShape[0], xstart:xend] = matplotArray

	fig.clear()
	print("Frame " + str(i) + " done.")

print("Done! Saving to avi...")
wb.saveFile(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+"_DATA.avi", new_movie, fps = 30)
print("Playing movie...")
wb.playMovie(new_movie, cmap=cv.COLORMAP_BONE)
cv.waitKey(0)
cv.destroyAllWindows()
