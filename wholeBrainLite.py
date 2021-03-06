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

def rollingAverage(pathlist, byframe=True):
	'''
	Calculates rolling average of a series of tiff files summing by one to 
	save memory.
	Takes list of paths, returns average in a numpy array
	'''
	print('\nTaking Rolling Average\n-----------------------')

	# Ignores UserWarning tags not ordered by code from tiff files
	warnings.simplefilter('ignore', UserWarning)

	# Open each file and sum all frames
	for f, path in enumerate(pathlist):
		
		print('Loading tiff file at', path)

		with TiffFile(path) as tif:
			t0 = timer()

			if f == 0:
				# Get sizing information on the first frame of each tiff file
				shape = tif.pages[0].shape[-2:]
				nframes = 0
				print('\tframe shape:', shape)
				
			else:
				assert tif.pages[0].shape[-2:] == shape, "Frames shapes don't match"

			if byframe: #loop through frame by frame to average

				for i, page in enumerate(tif.pages):
					if i == 0:
						sumimg = np.zeros(shape)

					if len(page.shape) > 2:
						frame = np.array(page.asarray())
						sumimg += frame.sum(axis=0)
						nframes += frame.shape[0]

					else:
						frame = np.array(page.asarray())
						sumimg += frame
						nframes += 1

				i += 1 # account for 0th element in n frames
				print('\trolling average took', timer() - t0, 'seconds')

			else: #open one file at a time
				array = np.array(tif.asarray())
				sumimg = array.sum(axis=0)
				nframes += array.shape[0]

				

			if f == 0:
				avgimg = sumimg / nframes
				n = nframes
			else:
				avgimg = (n*avgimg + sumimg) / (n + nframes) # rolling weighted average
				n += nframes

			print('\t%i frames averaged' %n)
		
	return avgimg


def rollingDFOF(pathlist, average, output, resize_factor=1, codec=None, speed=1, 
	fps=10, cmap=cv2.COLORMAP_JET, rotate=True, roipath=None, tcpath=None,
	byframe=True, dfof_std=True):

	'''
	Calculates rolling dfof of a series of tiff files one by one to save memory.
	Takes list of paths and the averaged image, writes .avi file to output.
	'''

	print('\nCalculating Rolling dFoF\n-----------------------')

	# find codec to use if not specified
	if codec is None:
		if output.endswith('.mp4'):
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
	sz = average.shape

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
	if rotate:
		sz = (sz[1], sz[0])
	w = int(sz[0] // resize_factor)
	h = int(sz[1] // resize_factor)
	
	# initialize movie writer
	display_speed = fps * speed
	fourcc = cv2.VideoWriter_fourcc(*codec) 
	out = cv2.VideoWriter(output, fourcc, display_speed, (h,w), isColor=True)
	
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
		out.write(frame)

	print('Saving dfof video to: ' + output)
	dfof_std_img = np.zeros_like(average)
	nframes = 0

	# Open each file and sum all frames
	for f, path in enumerate(pathlist):
		print('Loading tiff file at', path)

		with TiffFile(path) as tif:
			t0 = timer()

			if logmean:
				if (tif.pages[0].shape == 2):
					tc_length = len(tif.pages)

				elif (tif.pages[0].shape == 3):
					tc_length = tif.pages[0].shape[0]

				file_timecourse = np.zeros(tc_length)
				file_std_tc = np.zeros(tc_length)
				file_min_tc = np.zeros(tc_length)
				file_max_tc = np.zeros(tc_length)

			if byframe:
				for i, page in enumerate(tif.pages):
					
					if i % 100 == 0:
						print('\t{0}/{1}'.format(i,len(tif.pages)))

					frame = np.array(page.asarray())
					frame = np.divide(frame, average)
					frame -= 1.0

					if dfof_std:
						dfof_std_img += np.square(frame)
						nframes += 1

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

					if (frame.ndim == 3):
						movie = frame
						for frame in movie:
							writeFrame(frame)
					else:
						writeFrame(frame)
			else:
				print('loading..')
				movie = tif.asarray()
				print('loaded!')

				movie = np.divide(movie, average)
				movie -= 1.0
				print('divided')

				if f == 0:
					mean = movie.mean()
					std = movie.std()
					fmin = mean - 3 * std
					fmax = mean + 7 * std
					fslope = 255.0/(fmax-fmin)

				if logmean:
					masked = wb.getMaskedRegion(movie, roimask)
					file_timecourse = masked.mean(axis=1)
					file_std_tc[i] = masked.std(axis=1)
					file_min_tc[i] = masked.min(axis=1)
					file_max_tc[i] = masked.max(axis=1)
					del masked

				for i, frame in enumerate(movie):
					print('writing frame', i)
					writeFrame(frame)

			if logmean:
				file_mean[f] = file_timecourse
				file_std[f] = file_std_tc
				file_min[f] = file_min_tc
				file_max[f] = file_max_tc

			print('Writing file took {0} seconds\n'.format(timer()-t0))
		
	out.release()

	if logmean:
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

	if dfof_std:
		dfof_std_img = np.sqrt(dfof_std_img / nframes)
		return dfof_std_img


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
	ap.add_argument('-si', '--sumimage', help='path to average image if already computed (use default path if none supplied)', 
		nargs = '?', required=False, type=str, const="c")
	ap.add_argument('-g', '--grayscale', help='create grayscale movie', action='store_true')
	ap.add_argument('-v', '--deviation', help='compute standard deviation of dfof', action='store_true')
	args = vars(ap.parse_args())

	print("suminage:", args['sumimage'])

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
			if args['sumimage'] is None:
				avgimg = rollingAverage(pathlist)
				if args['avg'] is True:
					print('Writing average image to', avgpath)
					avgimg = wb.rescaleMovie(avgimg).astype('uint8')
					cv2.imwrite(avgpath, avgimg)
			elif args['sumimage'] == "$c$":
				avgimg = cv2.imread(avgpath)
			else:
				avgimg = cv2.imread(args['sumimage'])

			# Use average image to calculate dfof and write NAME_dfof.avi
			if args['grayscale'] is True:
				cmap = None
			else:
				cmap = cv2.COLORMAP_JET

			
			if args['deviation'] is True:
				dev = rollingDFOF(pathlist, avgimg, dfofpath, speed=speed, roipath=roipath,
				resize_factor=1/downsample, rotate=args['rotate'], cmap=cmap, 
				tcpath=tcpath, dfof_std = True)

				cv2.imshow("Standard Deviation", dev)
				cv2.imwrite(os.path.join(directory, expname + '_std_dev.png'), dev)
			else:
				rollingDFOF(pathlist, avgimg, dfofpath, speed=speed, roipath=roipath,
				resize_factor=1/downsample, rotate=args['rotate'], cmap=cmap, 
				tcpath=tcpath, dfof_std = False)


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


				
