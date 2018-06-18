import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from hdf5manager import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.figure
import cv2 as cv
import math
from OpenCVTest import load_mp4

def drawMatplotFrame(figure, frameIndex, data):
	parseAndDrawHDF5(data, frameIndex)
	drawFigureForRange(figure, data, frameIndex, 15)

def drawFigureForRange(figure, array, frame, range):
	sub = figure.add_subplot(133)

	bottom = max(0, frame - range)
	top = min(array.shape[0], frame + range)

	sub.plot(np.arrange(bottom, top), array[bottom:top], 'g-')
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

	data["foot1"]["pos"] = (-1,1)
    #data["foot2"]["pos"] = (1,1)
    #data["foot3"]["pos"] = (-1,-1)
    #data["foot4"]["pos"] = (1,-1)
    #data["head"]["pos"] = (0,2)
    #data["tail"]["pos"] = (0,-2)
	return data

def parseAndDrawHDF5(data, frameIndex):
    dxs = []
    dys = []
    xs = []
    ys = []
    x0s = []
    y0s = []
    for limbKey in data.keys:
    	limb = data[limbKey]

    	xs.append(limb["x"][frameIndex])
    	ys.append(limb["y"][frameIndex])
    	dxs.append(limb["dx"][frameIndex])
    	dys.append(limb["dy"][frameIndex])
    	x0s.append(limb["pos"][0])
    	y0s.append(limb["pos"][1])

	sub = figure.add_subplot(131)
    sub.quiver(x0s, y0s, dxs, dys, angles='xy', scale_units='xy', scale=1)

    sub.set_xlabel('x')
    sub.set_ylabel('y')
    sub.set_xlim(-5,5)
    sub.set_ylim(-5,5)
    

    sub2 = figure.add_subplot(132)      
    sub2.scatter(xs, ys)
    
    sub2.set_xlim(-5,5)
    sub2.set_ylim(-5,5)
        


movie_name = "bottom1"
data_name = "testHDF5"
source_movie = load_mp4(movie_name)
new_movie = np.zeros((source_movie.shape[0], 1200, 700, 3), dtype="uint8")
data = loadHDF5()

fig = plt.figure()

for i, frame in enumerate(new_movie):
	sourceShape = source_movie.shape
	frame[0:sourceShape[1],0:sourceShape[2]] = source_movie[i]
	
	drawMatplotFrame(fig, i, data)
	matplotArray = arrayFromFigure(fig)
	matplotShape = matplotArray.shape
	frame[sourceShape[1]+20:matplotShape[0]+sourceShape[1]+20, 0:matplotShape[1]] = matplotArray

	fig.clear()

wb.playMovie(new_movie, cmap=cv.COLORMAP_BONE)
cv.waitKey(0)
cv.destroyAllWindows()