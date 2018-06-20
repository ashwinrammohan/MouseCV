import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
import wholeBrain as wb
import matplotlib.figure
import cv2 as cv
import math
import datetime

def load_mp4(vid_name):
	cap = cv.VideoCapture("Assets/" + vid_name + ".mp4")
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

def drawFigureForRange(figure, data, frame, range):
	sub = figure.add_subplot(212)

	for limbKey in data.keys():
		limb = data[limbKey]
		bottom = max(0, frame - range)
		top = min(len(limb["magnitude"]), frame + range)
		sub.plot(np.arange(bottom, top), limb["magnitude"][bottom:top], '-', color=limb["color"])

	sub.axvline(x=frame, color="red")
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

	data["footFL"]["pos"] = (0,1)
	data["footFL"]["color"] = (1, 0, 0)
	data["footFR"]["pos"] = (0,-1)
	data["footFR"]["color"] = (0, 1, 0)
	#data["footBL"]["pos"] = (-1,1)
	#data["footBL"]["color"] = (1, 0.7, 0.3)
	data["footBR"]["pos"] = (-1,-1)
	data["footBR"]["color"] = (0, 0, 1)
	#data["head"]["pos"] = (1,0)
	#data["head"]["color"] = (1, 0.3, 1)
	#data["tail"]["pos"] = (-2,0)
	#data["tail"]["color"] = (0.3, 1, 1)
	return data

def parseAndDrawHDF5(figure, data, frameIndex):
	sub = figure.add_subplot(211)
	for limbKey in data.keys():
		limb = data[limbKey]
		sub.arrow(limb["pos"][0], limb["pos"][1], limb["dx"][frameIndex], limb["dy"][frameIndex], color=limb["color"])
	#sub.quiver(x0s, y0s, dxs, dys, angles='xy', scale_units='xy', scale=1)

	sub.set_xlabel('x')
	sub.set_ylabel('y')
	sub.set_xlim(-5,5)
	sub.set_ylim(-5,5)
		
movie_name = "paint3Contours"
data_name = "mouse_vectors"
source_movie = load_mp4(movie_name)
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

print("Done! Saving to mp4...")
wb.saveFile(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+"_DATA.avi", new_movie, fps = 30)
print("Playing movie...")
wb.playMovie(new_movie, cmap=cv.COLORMAP_BONE)
cv.waitKey(0)
cv.destroyAllWindows()