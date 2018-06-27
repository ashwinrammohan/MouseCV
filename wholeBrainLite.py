#!/usr/bin/env python3
'''
Functions used for GCaMP whole brain video analysis for low memory 
analysis, and unsupervised triggered operations on data server
wholeBrainLite.py

Importing from python terminal/script:
	- import wholeBrainLite as wbl
	- from wholeBrainLite import functionName

If wholeBrainLite.py file not within folder of script to import in, 
first add its containing folder to the python path with:
	sys.path.append('/home/sydney/Lab/pyWholeBrain/')
	import wholeBrainLite as wbl

How to use testing section for saving/loading files:
	loads 1+ videos into class for testing functionality of program.
	- python wholeBrainLite.py -m ../testfile*
	- python wholeBrainLite.py -m testfile*.tif

Authors: Sydney C. Weiser
Date: 2017-02-08
'''

import fileManager as fm
import metaManager as mm
import wholeBrain as wb

from tifffile import TiffFile
import warnings
import numpy as np
from timeit import default_timer as timer
import cv2
import os
import time
import _thread
import multiprocessing
from multiprocessing import Process, Manager


class ThreadManager:
	def __init__(self, thread_method, thread_callback, finished_callback, threads=0, settings = {}):
		if threads == 0:
			self.wanted_threads = multiprocessing.cpu_count()
		else:
			self.wanted_threads = threads

		self.settings = Manager().dict()
		self.settings["finished_threads"] = 0
		for key in settings.keys():
			self.settings[key] = settings[key]

		self.thread_method = thread_method
		self.thread_callback = thread_callback
		self.finished_callback = finished_callback

	def callback(self, data, settings):
		self.thread_callback(data, settings)
		settings["finished_threads"] += 1
		print("Thread " + str(settings["finished_threads"]) + " completed \n")

	def run(self, paths, data={}):
		paths = paths[:16]

		if (len(paths) > self.wanted_threads):
			self.wanted_threads = len(paths)

		threads = []
		pathsPerThread = int(len(paths) / self.wanted_threads)
		print("Threads:", self.wanted_threads)
		print("Files:", len(paths))

		upper = 0
		for i in range(self.wanted_threads-1):
			subpaths = paths[i::self.wanted_threads]
			cpy = {**data}
			cpy["index"] = i
			p = Process(target = self.thread_method, args = (subpaths, cpy, self.settings, self.callback))
			p.start()
			threads.append(p)

		cpy = {**data}
		cpy["index"] = self.wanted_threads-1
		subpaths = paths[self.wanted_threads-1::self.wanted_threads]
		self.thread_method(subpaths, cpy, self.settings, self.callback)
		for p in threads:
			p.join()

		self.finished_callback(self.settings)



		

def _rollingAverageAverage(data, manager):
	avimg = data["avimg"]
	frame_count = data["frame_count"]

	if (data["final_avgimg"] == None):
		data["final_avgimg"] = avimg
		data["total_frame_count"] = frame_count
	else:
		denom = data["total_frame_count"] + frame_count
		data["final_avgimg"] = frame_count / denom * avimg + data["total_frame_count"] / denom * data["final_avgimg"]
		data["total_frame_count"] += frame_count


def _rollingAverage(pathlist, data, manager):
	'''
	Calculates rolling average of a series of tiff files summing by one to 
	save memory.
	Takes list of paths, returns average in a numpy array
	'''

	# Ignores UserWarning tags not ordered by code from tiff files
	warnings.simplefilter('ignore', UserWarning)
	file_loadtime = 0

	# Open each file and sum all frames
	for f, path in enumerate(pathlist):
		
		nframes = 0
		file_loadtime = time.clock()

		with TiffFile(path) as tif:
			print("TiffFile load time: " + str(time.clock() - file_loadtime) + "\n")
			print("Loaded file " + path + "\n")

			if f == 0:
				# Get sizing information on the first frame of each tiff file
				shape = tif.pages[0].shape[-2:]
				nframes = 0
				
			else:
				assert tif.pages[0].shape[-2:] == shape, "Frames shapes don't match"


			for i, page in enumerate(tif.pages):
				if i == 0:
					sumimg = page.asarray()
					nframes += 1

				if len(page.shape) > 2:
					frame = page.asarray()
					sumimg += frame.sum(axis=0)
					nframes += frame.shape[0]

				else:
					tonp_loadtime = time.clock()
					frame = page.asarray()
					sumimg += frame
					nframes += 1

			if f == 0:
				avgimg = sumimg / nframes
				n = nframes
			else:
				avgimg = (n*avgimg + sumimg) / (n + nframes) # rolling weighted average
				n += nframes

	print("Thread finished with paths: \n" + pathlist + "\n")
	data["avgimg"] = avgimg
	data["frame_count"] = nframes
	manager.callback(data)

def startDFOF(data):
	final_avgimg = data["final_avgimg"]

	print("Average done!")
	print('Writing average image to', avgpath)

	final_avgimg = wb.rescaleMovie(final_avgimg).astype('uint8')
	cv2.imwrite(avgpath, final_avgimg)

		# find codec to use if not specified
	if codec is None:
		if dfofpath.endswith('.mp4'):
			if os.name is 'posix':
				codec = 'X264'
			elif os.name is 'nt':
				# codec = 'H264'
				codec = 'XVID'
		else:
			if os.name is 'posix':
				codec = 'MP4V'
			elif os.name is 'nt':
				codec = 'XVID'
			else:
				assert os.name is 'nt', 'Unknown os type: {0}'.format(os.name)
	
	# Initialize Parameters
	resize_factor = 1/resize_factor
	sz = final_avgimg.shape

	# if roipath is provided, load it and create mask.
	# initiate file and storage container for mean timecourses.
	if (roipath is not None) and (tcpath is not None):

		print('loading ROI mask...')
		rois = wb.roiLoader(roipath, verbose=False)
		roimask = np.zeros(sz, dtype = 'uint8')

		# Add mask region from all ob/cortex/sc rois
		for i, roi in enumerate(rois):
			if re.search('(ob)|(cortex)|(sc)', roi.lower()) is not None:
				roimask += wb.makeMask(rois[roi], sz)
		roimask[np.where(roimask > 1)] = 1

		from hdf5manager import hdf5manager as h5
		logmean = True
		f_mean = h5(tcpath)
		file_mean = [None] * len(pathlist)
		file_std = [None] * len(pathlist)
		file_min = [None] * len(pathlist)
		file_max = [None] * len(pathlist)
	else:
		roimask = None
		logmean = False
	
	# Ignores UserWarning tags not ordered by code from tiff files
	warnings.simplefilter('ignore', UserWarning)

	# Set up resizing factors
	if args['rotate']:
		sz = (sz[1], sz[0])
	w = int(sz[0] // (1/downsample))
	h = int(sz[1] // (1/downsample))
	
	# initialize movie writer
	display_speed = fps * speed
	fourcc = cv2.VideoWriter_fourcc(*codec) 
	out = cv2.VideoWriter(dfofpath, fourcc, display_speed, (h,w), isColor=True)

	data = {"average":final_avgimg, "logmean":logmean, "out":out, "roipath":roipath, "rotate":rotate, "cmap":cmap, "neededIndex":{"value":0}}
	dfofThreads = ThreadManager(_rollingDFOFandSTD, _comebineRollingDFOF, dfofFinished)
	dfofThreads.run(data["pathlist"], data = data)

def rollingAverage(pathlist):
	threadManager = ThreadManager(_rollingAverage, _rollingAverageAverage, startDFOF)
	threadManager.run(pathlist, data = {"final_avgimg":None, "total_frame_count":0, "pathlist":pathlist})


def _comebineRollingDFOF(data, manager):
	print("Thread " + str(data["index"]) + " finished, its waitng for its turn")
	while data["neededIndex"]["value"] < data["index"]:
		pass

	print("Thread " + str(data["index"]) + " is now writing its data")
	for frame in data["movie"]:
		data["out"].write(frame)

	data["neededIndex"]["value"] += 1
	if data["neededIndex"]["value"] >= manager.wanted_threads:
		data["neededIndex"]["value"] = 0

	print("Thread " + str(data["neededIndex"]["value"]) + " should now write its data")

def dfofFinished(data):
	data["out"].release()
	print("TECHNICALLY WE SHOULD HAVE A VIDEO NOW")

	if data["logmean"]:
		file_mean = np.concatenate(file_mean)
		file_std = np.concatenate(file_std)
		file_min = np.concatenate(file_min)
		file_max = np.concatenate(file_max)
		data = {'dfof_mean':file_mean,
				'dfof_std':file_std,
				'dfof_min':file_min,
				'dfof_max':file_max,
				'exp_mean':mean,
				'exp_std':std,
				'cmin':fmin,
				'cmax':fmax}
		f_mean.save(data)

def _rollingDFOFandSTD(pathlist, data, manager):

	'''
	Calculates rolling dfof of a series of tiff files one by one to save memory.
	Takes list of paths and the averaged image, writes .avi file to output.
	'''

	average = data["average"]
	logmean  data["logmean"]
	out = data["out"]
	roipath = data["roipath"]
	cmap = data["cmap"]
	rotate = data["rotate"]
	result = None
	index = 0
	
	def writeFrame(frame):
			
		# rescale and convert to uint8
		frame = frame - fmin
		frame = frame * fslope

		# cap min and max to prevent incorrect cmap application
		frame[frame > 255] = 255
		frame[frame < 0] = 0

		frame = frame.astype('uint8')

		# resize and rotate
		if rotate:
			frame = cv2.resize(frame, (w,h), 
				interpolation = cv2.INTER_AREA)
			frame = np.rot90(frame, 3)
		else:
			frame = cv2.resize(frame, (h,w), 
				interpolation = cv2.INTER_AREA)

		# apply colormap, write frame to .avi
		if cmap is not None: 
			frame = cv2.applyColorMap(frame.astype('uint8'), cmap)
		else:
			frame = np.repeat(frame[:,:,None], 3, axis=2)
		result[index] = frame
		index += 1

	#print('Saving dfof video to: ' + output)

	# Open each file and sum all frames
	for f, path in enumerate(pathlist):
		print('Loading tiff file at', path, "\n")

		with TiffFile(path) as tif:
			if logmean:
				if (tif.pages[0].shape == 2):
					tc_length = len(tif.pages)

				elif (tif.pages[0].shape == 3):
					tc_length = tif.pages[0].shape[0]

				file_timecourse = np.zeros(tc_length)
				file_std_tc = np.zeros(tc_length)
				file_min_tc = np.zeros(tc_length)
				file_max_tc = np.zeros(tc_length)

			movie_shape = (len(tif.pages),) + tif.pages[0].shape
			result = np.empty(movie_shape, dtype="uint8")
			index = 0
			for i, page in enumerate(tif.pages):
					
				if i % 100 == 0:
					print('\t{0}/{1}'.format(i,len(tif.pages)))

				frame = page.asarray()
				frame = np.divide(frame, average)
				frame -= 1.0

				if logmean and (frame.ndim == 2): 
					file_timecourse[i] = frame[np.where(roimask == 1)].mean()
					file_std_tc[i] = frame[np.where(roimask == 1)].std()
					file_min_tc[i] = frame[np.where(roimask == 1)].min()
					file_max_tc[i] = frame[np.where(roimask == 1)].max()

				elif logmean and (frame.ndim == 3):
					masked = wb.getMaskedRegion(frame, roimask)
					file_timecourse = masked.mean(axis=1)
					file_std_tc[i] = masked.std(axis=1)
					file_min_tc[i] = masked.min(axis=1)
					file_max_tc[i] = masked.max(axis=1)
					del masked

				
				# if first frame, calculate scaling parameters
				if (i == 0) and (f == 0):

					if logmean:
						mean = file_timecourse[i]
						std = file_std_tc[i]

					else:
						mean = frame.mean()
						std = frame.std()
					
					fmin = mean - 3 * std
					fmax = mean + 7 * std
					fslope = 255.0/(fmax-fmin)

				writeFrame(frame)

			if logmean:
				file_mean[f] = file_timecourse
				file_std[f] = file_std_tc
				file_min[f] = file_min_tc
				file_max[f] = file_max_tc

			if f < len(pathlist)-1:
				manager.finished_threads -= 1
			data["movie"] = result

			#data["file_mean"] = file_mean
			#data["file_std"] = file_std
			#data["file_min"] = file_min
			#data["file_max"] = file_max

			manager.callback(data)


if __name__ == '__main__':

	import argparse
	import os
	import re

	ap = argparse.ArgumentParser()
	ap.add_argument('-m', '--movie', type=argparse.FileType('r'), 
		nargs='+', required=False, help='path to the movie to be loaded')
	ap.add_argument('-r', '--rois', type = argparse.FileType('r'),
		nargs = 1, required = False, help = 'path to .zip file with .rois.')
	ap.add_argument('-f', '--folder', type=str,
		nargs=1, required=False, help='path to the folder to be scanned')
	ap.add_argument('-s', '--speed', type=int,
		nargs=1, required=False, help='speed of output video')
	ap.add_argument('-a', '--avg', help='write average images', action='store_true')
	ap.add_argument('-rt', '--rotate', help='toggle rotation--default is on.',
		action='store_false')
	ap.add_argument('-d', '--downsample', help='factor to downsize video', 
		nargs = 1, required=False, type=int)
	ap.add_argument('-e', '--experiment', type=str, nargs=1, required=False,
		help='name of experiment to process (string equality)')
	ap.add_argument('-g', '--grayscale', help='create grayscale movie', action='store_true')
	args = vars(ap.parse_args())

	if args['speed'] is not None:
		speed = args['speed'][0]
	else:
		speed = 2

	if args['downsample'] is not None:
		downsample = args['downsample'][0]
	else:
		downsample = 4

	if args['folder'] is not None:
		# scan for metadata files and make dfof movies
		directory = args['folder'][0]
		
		print('\nFolder parser and metadata reader\n-----------------------')

		# take folder path from first argument passed by bash
		output_file = 'metadata.csv'
		output_path = os.path.join(directory, output_file)

		# find all files, and all *_meta.yaml files in folder
		yamlmetafiles = fm.findFiles(directory, '_meta.yaml', suffix=True)

		if len(yamlmetafiles) > 0:
			# find database in sister folder to pywholebrain, or in D drive
			try:
				db = 'D:/waves-db/yamlFiles/'
				assert os.path.isdir(db), 'Metadatabase not found in D drive'
			
			except:
				parent = os.path.dirname(os.getcwd())
				db = os.path.join(parent, 'waves-db', 'yamlFiles') + os.sep

				if not os.path.isdir(db):
					db = None

			if db is not None:
				pathdict = mm.databaseSort(yamlmetafiles)

				for key, value in pathdict.items():
					meta_fnm = db + key + '_metadata.yaml'
					meta = mm.combineYaml(value)
					meta = mm.mergeYaml(db, meta)
					mm.saveYaml(meta_fnm, meta)
			else:
				print('\n-----------------------\nMetadata database not found')
				print("Check that '../waves-db/yamlFiles' exists.")
				print('-----------------------\n')

		else:
			print('No .yaml metadata files were found')

		videofiles = fm.findFiles(directory, '(\d{6}_\d{2})(?:[@-](\d{4}))?\.tif', 
			regex=True)

		roifiles = fm.findFiles(directory, '(\d{6}_\d{2})_RoiSet.zip', 
			regex=True)

		experiments = fm.movieSorter(videofiles)

		print('\nWriting videos for each experiment\n-----------------------')

		for expname in sorted(experiments):

			# if experiment filter arg was found, only make movie for that exp
			if args['experiment'] is not None:
				if expname != args['experiment'][0]:
					print(expname, 'does not match experiment key:', 
						args['experiment'][0]+'.  skipping..')
					continue
				else:
					print('found match:', expname)

			for roipath in roifiles:
				if roipath.find(expname) > 0:
					print('found roi:', roipath)
					break
				print('No Rois Found')
				roipath = None

			# Make output filenames based on name
			dfofpath = directory+os.path.sep+expname+'_{0}x_dfof.mp4'.format(speed)
			avgpath = os.path.join(directory, expname + '_avgimg.png')
			tcpath = os.path.join(directory, expname + '_videodata.hdf5')
			pathlist = experiments[expname]

			# Calculate average image and write
			avgimg = rollingAverage(pathlist)

			'''if args['avg'] is True:
				print('Writing average image to', avgpath)
				avgimg = wb.rescaleMovie(avgimg).astype('uint8')
				cv2.imwrite(avgpath, avgimg)

			 # Calculate average image and write
			if args['sd'] is True:
				print('Writing standard deviation image to', sdpath)
				avgimg = wb.rescaleMovie(sdimg).astype('uint8')
				cv2.imwrite(sdpath, sdimg)

			# Use average image to calculate dfof and write NAME_dfof.avi
			if args['grayscale'] is True:
				cmap = None
			else:
				cmap = cv2.COLORMAP_JET

			rollingDFOF(pathlist, avgimg, dfofpath, speed=speed, roipath=roipath,
				resize_factor=1/downsample, rotate=args['rotate'], cmap=cmap, 
				tcpath=tcpath)'''


	elif args['movie'] is not None:
		# Create a dfof movie from movie file(s) given 

		# Get pathlist from arguments
		pathlist = [path.name for path in args['movie']]
		print('{0} files found:'.format(len(pathlist)))

		for path in pathlist:
			print('\t' + path)
		print('\n')

		# What's the base name, directory
		directory = os.path.dirname(os.path.abspath(pathlist[0]))
		name = fm.getBaseName(pathlist[0])

		# Make output filenames based on name
		dfofpath = directory + os.path.sep + name + '_{0}x_dfof.mp4'.format(speed)
		avgpath = directory + os.path.sep + name + '_avgimg.png'

		# Calculate average image and write
		avgimg = rollingAverage(pathlist)
		if args['avg'] is True:
			print('Writing average image to', avgpath)
			avgimg = wb.rescaleMovie(avgimg).astype('uint8')
			cv2.imwrite(avgpath, avgimg)

		# Use average image to calculate dfof and write NAME_dfof.avi
		rollingDFOF(pathlist, avgimg, dfofpath, speed=speed, 
			resize_factor = 1/downsample, rotate=args['rotate'])

	else:
		print('Provide either folder or movie path.\nExiting...')


				
