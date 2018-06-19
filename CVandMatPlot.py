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
	cap = cv.VideoCapture(vid_name + ".mp4")
	frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
	frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

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
		sub.plot(np.arange(bottom, top), limb["magnitude"][bottom:top], 'g-')

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
	data = hdf5manager(file_name+".hdf5").load()

	#data["footFL"]["pos"] = (0,1)
	data["footFR"]["pos"] = (0,-1)
	#data["footBL"]["pos"] = (-1,1)
	#data["footBR"]["pos"] = (-1,-1)
	#data["head"]["pos"] = (1,0)
	#data["tail"]["pos"] = (-2,0)
	return data

def parseAndDrawHDF5(figure, data, frameIndex):
	dxs = []
	dys = []
	xs = []
	ys = []
	x0s = []
	y0s = []

	for limbKey in data.keys():
		limb = data[limbKey]
		xs.append(limb["x"][frameIndex])
		ys.append(limb["y"][frameIndex])
		dxs.append(limb["dx"][frameIndex])
		dys.append(limb["dy"][frameIndex])
		x0s.append(limb["pos"][0])
		y0s.append(limb["pos"][1])

	sub = figure.add_subplot(211)
	sub.quiver(x0s, y0s, dxs, dys, angles='xy', scale_units='xy', scale=1)

	sub.set_xlabel('x')
	sub.set_ylabel('y')
	sub.set_xlim(-5,5)
	sub.set_ylim(-5,5)
		
movie_name = "bottom1Contours"
data_name = "testHDF5"
source_movie = load_mp4(movie_name)
new_movie = np.zeros((source_movie.shape[0], 700, 1200, 3), dtype="uint8")
data = loadHDF5(data_name)

fig = plt.figure(figsize=(4,6))

for i, frame in enumerate(new_movie):
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
