#!/usr/bin/env python3
'''
Functions and class used for GCaMP whole brain video analysis.
wholeBrain.py

Importing from python terminal/script:
    - import wholeBrain as wb
    - from wholeBrain import functionName

If wholeBrain.py file not within folder of script to import in, 
first add its containing folder to the python path with:
    sys.path.append('/home/sydney/Lab/pyWholeBrain/')
    import wholeBrain as wb

How to use testing section for saving/loading class files:
    loads 1+ videos into class for testing functionality of program.
    - python wholeBrain.py -m matfile*.mat
    - python wholeBrain.py -m ../testfile*
    - python wholeBrain.py -m testfile*.tif

Authors: Sydney C. Weiser
Date: 2016-10-25
'''

import cv2
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import tifffile
import zipfile
from io import BytesIO
import warnings
import re
import os
import time

try:
    import wholeBrainPCA as wbpca
except:
    print('wholeBrainPCA.py not found or had import errors.')
    print('Download from pyWholeBrain to use Principal '
        'Component Analysis')

try:
    import metaManager as mm
except:
    print('metaManager.py not found or had import errors.')
    print('Download from pyWholeBrain to import metadata files')

try:
    import wholeBrainFilter as wbf
except:
    print('wholeBrainFilter.py not found or had import errors.')
    print('Download from pyWholeBrain to use image processing filters')

try:
    from hdf5manager import hdf5manager
except:
    print('hdf5manager.py was not found or had import errors.')
    print('Download from pyWholeBrain to use hdf5 saving/loading functionality')

try:
    import fileManager as fm
except:
    print('Error importing fileManager.py')
    print('Download from pyWholeBrain to use file management functionality')

try:
    import wholeBrainParcellation as wbp
except:
    print('Error importing wholeBrainParcellation.py')
    print('Download from pyWholeBrain to implement ica domain calculations')

def loadMovie(pathlist, downsample=False, dtype=None, verbose=False, tiffloading=True):
    '''
    Loads a list of tiff paths, returns concatenated arrays.
    Implemented size-aware loading for tiff arrays with pre-allocation
    Expects a list of pathnames: 
    ex: ['./testvideo1.mat', '/home/sydney/testfile2.tif']
    Files in list must be the same xy dimensions.
    if downsample is an integer greater than one, movie will be downsampled by that factor.
    '''
    print('\nLoading Files\n-----------------------')
    
    if type(pathlist) is str: # if just one path got in
        pathlist = [pathlist]
        
    # make sure pathlist is a list of strings
    assert type(pathlist) is list
    
    # use tiff loading if all paths are tiffs.
    for obj in pathlist:
        assert type(obj) is str
        if not (obj.endswith('.tif') | obj.endswith('.tiff')):
            print('Found non-tiff file to load:', obj)
            print('Not using size-aware tiff loading.')
            tiffloading = False
        
    # if downsample is not False, assert it's an integer
    if downsample:
        assert type(downsample) is int
    
    # ignore tiff warning (lots of text, unnecessary info)
    warnings.simplefilter('ignore', UserWarning) 

    # if no datatype was given, assume it's uint16
    if dtype == None:
        dtype = 'uint16'
    
    # use size-aware tiff loading to preallocate matrix and load one at a time
    if tiffloading:
        print('Using size-aware tiff loading.')
        nframes = 0

        # loop through tiff files to determine matrix size
        for f, path in enumerate(pathlist):
            with tifffile.TiffFile(path) as tif:
                if (len(tif.pages) == 1) and (len(tif.pages[0].shape) == 3):
                    # sometimes movies save as a single page
                    pageshape = tif.pages[0].shape[1:]
                    nframes += tif.pages[0].shape[0]

                else:
                    nframes += len(tif.pages)
                    pageshape = tif.pages[0].shape

                if f == 0:
                    shape = pageshape
                else:
                    assert pageshape == shape, \
                        'shape was not consistent for all tiff files loaded'    

        shape = (nframes, shape[0], shape[1])
        print('shape:', shape)

        # resize preallocated matrix if downsampling
        if downsample:
            shape = (shape[0], shape[1]//downsample, shape[2]//downsample)
            print('downsample:', shape, '\n')

        A = np.empty(shape, dtype=dtype)

        # load video one at a time and assign to preallocated matrix
        i = 0
        for f, path in enumerate(pathlist):
            t0 = timer()
            print('Loading file:', path)
            with tifffile.TiffFile(path) as tif:
                if (len(tif.pages) == 1) and (len(tif.pages[0].shape) == 3):
                    # when movies are saved as a single page
                    npages = tif.pages[0].shape[0]

                else:
                    npages = len(tif.pages)

                if downsample:
                    print('\t downsampling by {0}..'.format(downsample))
                    temp = tif.asarray()
                    A[i:i+npages] = scaleVideo(temp, downsample, verbose=verbose)
                else:
                    A[i:i+npages] = tif.asarray()

                i += npages

            print("\t Loading file took: {0} sec".format(timer() - t0))
            
    # don't use tiff size-aware loading.  load each file and append to growing matrix
    else:
        print('Not using size-aware tiff loading.')

        
        # general function for loading a path of any type.  
        # add if/elif statements for more file types
        def loadFile(path, downsample):
            t0 = timer()
            
            if path.endswith('.tif') | path.endswith('.tiff'):
                print("Loading tiff file at " + path)
                with tifffile.TiffFile(path) as tif:
                    A = tif.asarray()
                    if type(A) is np.memmap:
                        A = np.array(A, dtype=dtype)
            
            elif path.endswith('.mat'):
                print("Loading hdf5 file at " + path)
                with h5py.File(path) as f:
                    A = f.get('BrainCrop', 'r')
                    A = np.array(A, dtype=dtype)

            else:
                print('File path is of unknown file type!')
                raise Exception("'{0}' does not have a supported \
                    path extension".format(path))

            if downsample is not False:
                print('\t downsampling by {0}..'.format(downsample))
                A = scaleVideo(A, downsample, verbose=False)

            print("Loading file took: {0} sec".format(timer() - t0))
                
            return A

        # load either one file, or load and concatenate list of files 
        if len(pathlist) == 1:
            A = loadFile(pathlist[0], downsample)
        else:
            for i, path in enumerate(pathlist):
                Atemp = loadFile(path, downsample)

                t0 = timer()
                if i == 0:
                    A = Atemp
                else:
                    A = np.concatenate([A, Atemp], axis=0)

                print("Concatenating arrays took: {0} sec\n".format(
                    timer() - t0))
    
    return A


def saveFile(path, array, resize_factor = 1, apply_cmap = True,
    speed = 1, fps = 10, codec = None):
    '''
    Check what the extension of path is, and use the appropriate function
    for saving the array.  Functionality can be added for more 
    file/data types.

    For AVIs:
    Parameters (args 3+) are used for creating an avi output.  
    MJPG and XVID codecs seem to work well for linux systems, 
    so they are set as the default.
    A full list of codecs could be found at:
    http://www.fourcc.org/codecs.php.

    We may need to investigate which codec gives us the best output. 
    '''
    print('\nSaving File\n-----------------------')
    assert(type(array) == np.ndarray), ('Movie to save was not a '
        'numpy array')

    if path.endswith('.tif') | path.endswith('.tiff'):
        print('Saving to: ' + path)
        t0 = timer()
        if array.shape[2] == 3:
            # convert RGB to BGR
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            print('Converted RGB image to BGR.')
        with tifffile.TiffWriter(path) as tif:
            tif.save(array)
        print("Save file: {0} sec\n".format(timer()-t0))

    elif path.endswith('.png'):
        assert array.ndim <= 3, 'File was not an image'
        cv2.imwrite(path, array)

    elif path.endswith('.avi') | path.endswith('.mp4'):
        sz = array.shape

        if codec == None: # no codec specified
            if path.endswith('.avi'):
                    codec = 'MJPG'
            elif path.endswith('.mp4'):
                if os.name is 'posix':
                    codec = 'X264'
                elif os.name is 'nt':
                    # codec = 'H264'
                    codec = 'XVID'

        # check codec and dimensions
        if array.ndim == 3:
            if apply_cmap == False:
                sz = array.shape
                array = array.reshape(sz[0], sz[1]*sz[2])
                array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
                array = array.reshape((sz[0], sz[1], sz[2], 3))
                movietype = 'black and white'
            else:
                movietype = 'color'
        elif array.ndim == 4:
            movietype = 'color'
        else:
            raise Exception('Input matrix was {0} dimensions. .avi '
                'cannot be written in this format.'.format(array.ndim))

        print('Movie will be written in {0} using the {1} codec'.format(
            movietype, codec))
        print('Saving to: ' + path)
    
        # Set up resize
        w = int(ceil(sz[1] * resize_factor))
        h = int(ceil(sz[2] * resize_factor))

        # initialize movie writer
        display_speed = fps * speed
        fourcc = cv2.VideoWriter_fourcc(*codec) 
        out = cv2.VideoWriter(path, fourcc, display_speed, (h,w), True)

        for i in range(sz[0]):
            frame = cv2.resize(array[i], (h,w), interpolation = cv2.INTER_AREA)
            frame = frame.astype('uint8')
            if apply_cmap:
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE)
            out.write(frame)
        out.release()



    else:
        print('Save path is of unknown file type!')
        raise Exception("'{0}' does not have a supported \
            path extension".format(path))

    print('File saved to:' + path)
    
    return

def convertFloat(A, verbose=True):
    Amin = A.min()
    Amax = A.max()

    f16 = np.finfo('float16')
    f32 = np.finfo('float32')

    # if (Amin > f16.min) & (Amax < f16.max):
    #     print('Converting matrix to float16')
    #     A = A.astype('float16', copy=False)
    if (Amin > f32.min) & (Amax < f32.max):
        if verbose: print('Converting matrix to float32')
        t0 = timer()
        A = A.astype('float32', copy=False)
        if verbose: print('Conversion took {0} sec'.format(timer() - t0))

    return A


def dFoF(A):
    '''
    Calculates the change in fluorescence over mean 
    fluorescense for a video.
    Updates most movies in-place
    '''
    print('\nCalculating dF/F\n-----------------------')

    assert(type(A) == np.ndarray), 'Input was not a numpy array'

    if A.ndim == 3:
        reshape = True
        sz = A.shape
        A = np.reshape(A, (sz[0], int(A.size/sz[0])))
    elif A.ndim == 2:
        reshape = False
    else:
        assert A.ndim == 1, ('Input was not 1-3 dimensional: '
            '\{0} dim'.format(A.ndim))
        reshape = False
        A = A[:, None]
        print('Reshaped to two dimensional.')

    print("Array Shape (t,xy): {0}".format(A.shape))
    print('Array Type:', A.dtype)

    t0 = timer()
    Amean = np.mean(A, axis=0, dtype='float32')
    print("z mean: {0} sec".format(timer()-t0))
    print("Amean shape (xy): {0}".format(Amean.shape))
    print("Amean type: {0}".format(Amean.dtype))

    t0 = timer()
    A = A.astype('float32', copy=False)
    print("float32: {0} sec".format(timer()-t0))

    t0 = timer()
    for i in np.arange(A.shape[0]):
        A[i,:] /= Amean
        A[i,:] -= 1.0

    print("dfof normalization: {0} sec".format(timer()-t0))
    if reshape:
        A = np.reshape(A, sz)
    print("A type: {0}".format(A.dtype))
    print("A shape (t,x,y): {0}\n".format(A.shape))

    return A

def getMaskedRegion(A, mask, maskval = None):
    '''
    Extract a spatially masked array where the mask == 1 or 
    mask == maskval. 
    
    Accepts (t,x,y) arrays or (x,y,c) arrays.
    Returns the masked array in (t,xy) or (xy,c) format.
    Reinsert masked region using insertMaskedRegion function.
    '''

    if maskval == None:
        maskind = np.where(mask == 1)
    else:
        maskind = np.where(mask == maskval)

    if A.shape[0:2] == mask.shape: #check if dimensions align for masking
        M = A[maskind]
    elif (A.shape[1], A.shape[2]) == mask.shape:
        M = A.swapaxes(0,1).swapaxes(1,2)[maskind]
        M = M.swapaxes(0,1)
    else:
        raise Exception('Unknown mask indices with the following '
            'dimensions:\n', 
            'Array: {0} Mask: {1}'.format(A.shape, mask.shape))

    return M

def insertMaskedRegion(A, M, mask, maskval = 1):
    '''
    Insert a spatially masked array from getMaskedRegion.  
    Masked array is inserted where the mask == 1 or mask == maskval. 
    Accepts masked array in (t,xy) or (xy,c) format.
    Accepts (t,x,y) arrays or (x,y,c) arrays, returns them in the 
    same format.
    '''
    maskind = np.where(mask == maskval)

    if A.shape[0:2] == mask.shape: #check if dimensions align for masking
        A[maskind] = M
    elif (A.shape[1], A.shape[2]) == mask.shape:
        M = M.swapaxes(0,1)
        A = A.swapaxes(0,1).swapaxes(1,2)
        A[maskind] = M
        A = A.swapaxes(1,2).swapaxes(0,1)
        # A.swapaxes(0,1).swapaxes(1,2)[maskind] = M
    else:
        raise Exception('Unknown mask indices with the following '
            'dimensions:\n' 
            'Array: {0}, Mask: {1}'.format(A.shape, mask.shape))

    return A


def rescaleMovie(A, low=3, high=7, cap=True, mean_std=None, 
    mask=None, maskval=1, verbose=True, min_max=None):
    '''
    determine upper and lower limits of colormap for playing movie files. 
    limits based on standard deviation from mean.  low, high are defined 
    in terms of standard deviation.  Image is updated in-place, 
    and doesn't have to be returned.
    '''

    # Mask the region if mask provided
    if mask is not None:
        copy = A.copy()
        A = getMaskedRegion(A, mask, maskval)
        print(A.shape)

    # if unmasked color image, add extra temporary first dimension
    if A.ndim == 3:
        if A.shape[2] == 3: 
            A = A[None,:]

    if min_max is None:
        if mean_std is None:
            mean = np.nanmean(A, dtype=np.float64)
            std = np.nanstd(A, dtype=np.float64)
            if verbose: print('mean:', mean, 'std:', std)
        else:
            mean = mean_std[0]
            std = mean_std[1]

        newMin = mean - low*std
        newMax = mean + high*std
    else:
        assert len(min_max) == 2
        newMin = min_max[0]
        newMax = min_max[1]

        mean = np.nanmean(A, dtype=np.float64)
        std = np.nanstd(A, dtype=np.float64)

    if verbose: print('mean:', mean, 'low:', low, 'high:', high, 'std:', std)
    if verbose: print('newMin:', newMin)

    if cap: # don't reduce dynamic range
        if verbose: print('amin',np.nanmin(A))
        if newMin < A.min():
            newMin = A.min()

        if verbose: print('amax', np.nanmax(A))
        if newMax > A.max():
            newMax = A.max()

    newSlope = 255.0/(newMax-newMin)
    if verbose: print('newSlope:', newSlope)
    A = A - newMin
    A = A * newSlope

    if mask is not None:
        A = insertMaskedRegion(copy, A, mask, maskval)

    A[np.where(A > 255)] = 255
    A[np.where(A < 0)] = 0

    if A.shape[0] == 1: #if was converted to one higher dimension
        A = A[0,:]

    return A

def applyVideoColormap(video):

    print('\nApplying Color Map to Movie\n-----------------------')

    sz = video.shape
    A2 = np.zeros((sz[0], sz[1], sz[2], 3), 
        dtype = 'uint8') #create extra 4th dim for color
    for i in range(sz[0]):
        cv2.applyColorMap(video[i,:,:].astype('uint8'), 
            cv2.COLORMAP_JET, A2[i,:,:,:])

    print('\n')
    return A2.astype('uint8', copy=False)


def playMovie(A, min_max = None, preprocess = True, 
    toolbarsMinMax = False, rescale = True, cmap = cv2.COLORMAP_JET, 
    loop = True):
    '''
    play movie in opencv after normalizing display range
    A is a numpy 3-dimensional movie
    newMinMax is an optional tuple of length 2, the new display range

    Note: if preprocess is set to true, the array normalization is done 
    in place, thus the array will be rescaled outside scope of 
    this function
    '''
    print('\nPlaying Movie\n-----------------------')
    assert (type(A) == np.ndarray), 'A was not a numpy array' 
    assert(A.ndim == 3) | (A.ndim == 4), ('A was not three or '
        'four-dimensional array')

    windowname = "Press Esc to Close"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    #Create a resizable window

    if A.ndim == 3:
        if min_max == None:
            # if min/max aren't set, default to 3 and 7
            lowB = [3]
            highB = [7]
        else:
            # otherwise, use values given
            lowB = min_max[0]
            highB = min_max[1]

        if rescale == False: 
            # if the movie shouldn't be rescaled, don't display rescaling toolbars
            toolbarsMinMax = False
        else:
            # mean, std not required if not rescaling
            mean = A.mean(dtype=np.float64)
            std = A.std(dtype=np.float64)

        if preprocess:
            #Normalize movie range and change to uint8 before display
            if toolbarsMinMax:
                imgclone = A.copy()
            t0 = timer()
            sz = A.shape
            A = np.reshape(A, (sz[0], int(A.size/sz[0])))
            A = rescaleMovie(A, low=lowB[0], high=highB[0],
                mean_std=(mean, std))
            
            A = np.reshape(A, sz)
            A = A.astype('uint8', copy=False)
            print("Movie range normalization: {0}".format(timer()-t0))

        if toolbarsMinMax:
            def updateColormap(A):
                lowB[0] = 0.5 * ( 8 - cv2.getTrackbarPos("Low Limit", 
                    windowname))
                highB[0] = 0.5 * cv2.getTrackbarPos("High Limit", 
                    windowname)

                if preprocess:
                    A = imgclone.copy()
                    A = rescaleMovie(A, low=lowB[0], 
                        high=highB[0], mean_std=(mean, std))
                return

            cv2.createTrackbar("Low Limit", windowname, 
                (-2 * lowB[0] + 8), 8, lambda e: updateColormap(A))
            cv2.createTrackbar("High Limit", windowname, 
                (2 * highB[0]), 16, lambda e: updateColormap(A))

    i = 0
    toggleNext = True
    tf = True
    zoom = 1
    zfactor = 5/4

    while True:
        im = np.copy(A[i])
        if zoom != 1:
            im = cv2.resize(im,None,fx=1/zoom, fy=1/zoom, 
                interpolation = cv2.INTER_CUBIC)

        if A.ndim == 3:
            if (preprocess != True) & (rescale == True):
                im = rescaleMovie(im, low=lowB[0], high=highB[0],
                    mean_std=(mean, std), verbose=False)

            color = np.zeros((im.shape[0], im.shape[1], 3))
            color = cv2.applyColorMap(im.astype('uint8'), cmap, color)
            cv2.putText(color, str(i), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (255,255,255)) #draw frame text
            cv2.imshow(windowname,color)

        elif A.ndim == 4:
            cv2.putText(im, str(i), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (255,255,255)) #draw frame text
            cv2.imshow(windowname, im)

        k = cv2.waitKey(10) 
        if k == 27: #if esc is pressed
            break
        elif (k == ord(' ')) and (toggleNext == True):
            tf = False
        elif (k == ord(' ')) and (toggleNext == False):
            tf = True
        toggleNext = tf #toggle the switch

        if k == ord("="):
            zoom = zoom * 1/zfactor
        if k == ord("-"):
            zoom = zoom*zfactor

        if k == ord('b') and toggleNext:
            i -= 100
        elif k == ord('f') and toggleNext:
            i += 100
        elif k == ord('m') and (toggleNext == False):
            i += 1
        elif k == ord('n') and (toggleNext == False):
            i -= 1
        elif toggleNext:
            i += 1
        
        if (i > (A.shape[0]-1)) or (i < 0) :
            # reset to 0 if looping, otherwise break the while loop
            if loop: 
                i = 0
            else:
                break

    cv2.destroyAllWindows()
    for i in range(5):
        cv2.waitKey(1)

    print('\n')

def drawBoundingBox(image):
    '''
    Draw a bounding box on a two dimensional image.  
    Returns a ROI bounding box with shape [[x0,x1],[y0,y1]]
    '''
    print('\nDrawing Bounding Box\n-----------------------')

    assert (len(image.shape) == 2), '''The image is not two dimensional.  
        Shape: '''.format(image.shape)

    global refPt, cropping
    # initialize the list of reference points
    # and boolean indicating whether cropping is being performed
    window_name = "Draw bounding box on image.  \
    'r' = reset, 'a' = all, s' = save, +/- = zoom"
    refPt = []
    cropping = False
    zoom = 1
    zfactor = 5/4

    def click_and_crop(event, x, y, flags, param):
        global refPt, cropping
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x,y)]
            cropping = True

        #Check to see if left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x,y) coordinates and indicate that 
            # the cropping operation is finished
            refPt.append((x,y))
            cropping = False

            # draw a rectangle around the point of interest
            cv2.rectangle(draw, refPt[0], refPt[1], 255, 2)
            cv2.imshow(window_name, draw)

    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)

    # load the image, clone it to draw, and setup the mouse 
    # callback function
    
    image = rescaleMovie(image).astype('uint8')
    draw = image.copy()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow(window_name, draw)
        key = cv2.waitKey(1) & 0xFF

        # if the '=' key is pressed, zoom in
        if key == ord("="):
            draw = cv2.resize(draw,None,fx=zfactor, fy=zfactor, 
                interpolation = cv2.INTER_CUBIC)
            zoom = zoom*zfactor
            
        # if the '-' key is pressed, zoom out
        if key == ord("-"):
            draw = cv2.resize(draw,None,fx=1/zfactor, fy=1/zfactor, 
                interpolation = cv2.INTER_CUBIC)
            zoom = zoom * 1/zfactor

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            draw = image.copy()
            zoom = 1

        if key == ord("a"):
            print('Taking entire image')
            refPt = [(0,0), (image.shape[0], image.shape[1])]
            break

        # if the 's' key is pressed, break from the loop and save ROI
        elif key == ord("s"):
            break

    # if there are two reference points, then crop the region of interestdd 

    if len(refPt) == 2:
        ref_coord = np.array([sorted([refPt[0][1], refPt[1][1]]), 
            sorted([refPt[0][0], refPt[1][0]])])

        # unzoom reference coordinates
        for i in range(2):
            ref_coord[i] = [round(y/zoom) for y in ref_coord[i]]

    else:
        print('Error!!')
        ref_coord = None
    
    cv2.destroyAllWindows()
    for i in range(5):
        cv2.waitKey(1)
    
    assert (ref_coord is not None), 'Exited with no bounding box'
    print('Bounding box: x:{0}, y:{1}\n'.format(
            ref_coord[0], ref_coord[1]))
    print('\n')

    return ref_coord


def unitConversion(resolution=None, unit='um', area=False):
    '''
    Input is number of microns per pixel. Returns conversion factor for number of 
    '''
    if resolution is None:
        resolution = 6.75
        
    if unit is 'um':
        resultion = resolution
    elif unit is 'mm':
        resolution = resolution / 1000
    if area:
        resolution = resolution * resolution
    return resolution


def roiLoader(path, verbose=True):
    print('\nLoading Rois\n-----------------------')

    def loadRoiFile(fileobj):
        '''
        points = roiLoader(ROIfile)
        ROIfile is a .roi file view.  
        It must first be opened as a bitstring through BytesIO to allow 
        for seeking through the bitstring.
        Read ImageJ's ROI format. Points are returned in a nx2 array. Each row
        is in (x,y) order.
        This function may not work for float32 formats, or with images that 
        have subpixel resolution.

        This is based on a gist from luis pedro:
        https://gist.github.com/luispedro/3437255
        '''

        def get8():
            s = fileobj.read(1)
            if not s:
                raise IOError('readroi: Unexpected EOF')
            return ord(s)

        def get16():
            b0 = get8()
            b1 = get8()
            return (b0 << 8) | b1

        assert fileobj.read(4) == b'Iout'
        version = get16()
        roi_type = get8()
        get8()

        top = get16()
        left = get16()
        bottom = get16()
        right = get16()
        
        n_coordinates = get16()
        fileobj.seek(64) # seek to after header, where coordinates start

        points = np.empty((n_coordinates, 2), dtype=np.int16)
        points[:,0] = [get16() for i in range(n_coordinates)] # X coordinate
        points[:,1] = [get16() for i in range(n_coordinates)] # Y coordinate
        points[:, 1] += top
        points[:, 0] += left
        
        return points

    rois = dict()

    # Load a .zip file of .roi files
    if path.endswith('.zip'):
        f = zipfile.ZipFile(path)
        roifilelist = f.namelist()

        for roifile in roifilelist:
            if verbose: print('Loading ROIs at: ', roifile)
            roiname = re.sub('.roi', '', roifile)

            roidata = BytesIO(f.read(roifile))
            points = loadRoiFile(roidata)
            rois[roiname] = points
            roidata.close()

    # # Load a single .roi file
    # elif path.endswith('.roi'):
    #     roidata = file.open(path)
    #     points = roiLoader(roidata)
    #     if len(points > 0):
    #         rois.append(points)
    #     roidata.close()

    # Not a valid file extension
    else:
        raise Exception('{0} does not have a valid roi '
            'file extension.  Accepted file formats are '
            '.zip.'.format(path))

    return rois


def makeMask(polylist, shape):
    '''
    Makes mask from coordinates of polygon(s).  
    
    Polylist is a list of numpy arrays, each representing a 
    closed polygon to draw.
    '''
    assert (len(shape) == 2), 'Shape was not 2D'

    roimask = np.zeros(shape, dtype='uint8')
    cv2.fillPoly(roimask, [polylist.astype(np.int32).reshape((-1,1,2))], 1)

    return roimask

def scaleVideo(array, factor, verbose=True):
    if verbose: print('\nRescaling Video\n-----------------------')
    assert array.ndim == 3 or array.ndim == 2, 'Input was not a video or image'
    assert type(factor) is int

    shape = array.shape

    if array.ndim ==3:
        newshape = (shape[0], shape[1]//factor, shape[2]//factor)
        cropshape = (shape[0], newshape[1] * factor, newshape[2]*factor)
    else:
        newshape = (shape[0]//factor, shape[1]//factor)
        cropshape = (newshape[0] * factor, newshape[1]*factor)

    if cropshape != array.shape:
        if verbose: print('Cropping', array.shape, 'to', cropshape)
        if array.ndim == 3:
            array = array[:, :cropshape[1], :cropshape[2]]
        else:
            array = array[:cropshape[0], :cropshape[1]]

    if verbose: print('Downsampling by', factor)
    array = downSample(array, newshape)
    if verbose: print('New shape:', array.shape)

    return array

def downSample(array, new_shape):
    # reshape m by n matrix by factor f by reshaping matrix into
    # m f n f matricies, then applying sum across mxf, nxf matrices
    
    if array.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(array.shape,
                                                           new_shape))

    compression_pairs = [(d, c//d) for d,c in zip(new_shape, array.shape)]
    flattened_pairs = [l for p in compression_pairs for l in p]
    array = array.reshape(flattened_pairs)

    axis_to_iterate = [0, -2, -3]
    for i in range(len(new_shape)):
        array = array.mean(-1*(i+1))

    return(array)

def makeGridRois(movie, rois, npix = 7):
    '''
    movie is dFoF movie for best results
    npix = number of pixels per grid roi 
    '''

    ds_rois = {}

    # Find rois that correspond to ob/cortex/sc, find min/max 
    # values for each
    for i, roi in enumerate(rois):
        if re.search('(ob)|(cortex)|(sc)',roi.lower()) is not None:
            minval = rois[roi].min(0)
            maxval = rois[roi].max(0)

            minmax = np.vstack([minval, maxval])

            hemi = movie[:,
                minmax[:,1][0]:minmax[:,1][1],
                minmax[:,0][0]:minmax[:,0][1]]

            # flip it before rescaling, picking grid values so that 
            # l/r are aligned
            if re.search('l', roi.lower()):
                hemi = hemi[:,:,::-1] #flip movie left/right

            # make sure indices are a multiple of npix resizing factor
            x_ds = hemi.shape[1] // npix
            y_ds = hemi.shape[2] // npix
            hemi = hemi[:, 0:(x_ds*npix), 0:(y_ds*npix)]

            newshape = (hemi.shape[0], x_ds, y_ds)
            hemi_ds = downSample(hemi, newshape)

            ds_rois[roi] = hemi_ds

    return ds_rois

def getParallelTimecourse(region, ds_rois, xy):
    regionindex = []
    for roi in ds_rois:
        if re.search(region, roi) is not None:
            regionindex.append(roi)

    assert len(regionindex) == 2, ('More than one regions were found '
        'for the given region name \n{0}'.format(regionindex))

    result = []
    for roi in regionindex:
        print('Getting parallel timecourse from', roi)
        result.append(ds_rois[roi][:, xy[0], xy[1]])
    return result, regionindex

class wholeBrain:
    '''
    Define the wholeBrain class.  Must be a (t,x,y) 3 dimensinal video 
    (no color!)
    '''
    def __init__(self, pathlist, loadraw = False, downsample=False):

        print('\nInitializing wholeBrain Instance\n-----------------------')
        if isinstance(pathlist, str):
            pathlist = [pathlist]

        if (pathlist[0].endswith('.hdf5')) & (len(pathlist) == 1 ):
            f = hdf5manager(pathlist[0])
            data = f.load()

            for key in data:
                setattr(self, key, data[key])

            if 'movie' not in data.keys():
                print('Movie not in .hdf5 file')
                if loadraw:
                    try:
                        print('Trying to load movie from path...')
                        self.movie = loadMovie(self.path)
                    except:
                        print('Failed to load movie from path in .hdf5 file.')
                else:
                    print('Not attempting to load video.')

        else:
            movie = loadMovie(pathlist, downsample=downsample)
            assert(len(movie.shape) 
                == 3), 'File was not a 3 dimensional video.\n'

            self.downsample = downsample
            self.movie = movie
            self.path = pathlist
            # define default bounding box as full size video
            self.bounding_box = np.array([[0, self.movie.shape[1]], [0, 
                self.movie.shape[2]]])
            self.shape = self.boundMovie().shape

            name = os.path.basename(pathlist[0])
            # remove extension .XXX .XXXX from path 
            name = re.sub('(\.)(\w){3,4}', '', name) 
            # remove @0001 or -0001 from path
            name = re.sub('([@-])(\d){4}', '', name)
            self.name = name

            self.dir = os.path.dirname(pathlist[0])
            self.rotate()

    def rotate(self, n = None):
        # rotates a t,y,x movie counter-clockwise n times and 
        # updates relevant parameters
        
        if n is None:
            # Most common rotation is 3x ccw
            if int(self.name[:2]) < 15: # if movie is older than 2016
                print('Movie older than 2015--Not rotating')
                old = True
            else:
                old = False

            if (self.movie.shape[1] < self.movie.shape[2]) & (not old):
                n = 3
            else:
                n = 0

        if n > 0:
            movie = self.movie
            movie = (movie.swapaxes(0,2))
            movie = movie.swapaxes(0,1)

            movie = np.rot90(movie, n)
            movie = movie.swapaxes(0,1)
            movie = (movie.swapaxes(0,2))

            self.movie = movie
            self.bounding_box = np.array([[0, movie.shape[1]], 
            [0, movie.shape[2]]]) #resets to whole movie
            self.shape = self.boundMovie().shape

            # ADD ROTATION OF ROIS/ROIMASK


    def loadROIs(self, path):

        rois = roiLoader(path)

        # Store in class file
        print(len(rois), 'ROIs found')

        # resize (and flip) if necessary 
        if self.downsample is not False:
            print('video was downsampled.. downsampling rois.')
            for roi in rois:
                rois[roi] = rois[roi] // self.downsample

        self.rois = rois

        # Initialize Empty Mask
        roimask = np.zeros(self.shape[1:3], dtype = 'uint8')

        # Add mask region from all ob/cortex/sc rois
        for i, roi in enumerate(rois):
            if re.search('(ob)|(cortex)|(sc)', roi.lower()) is not None:
                roimask += makeMask(rois[roi], self.shape[1:3])

        roimask[np.where(roimask > 1)] = 1

        print('')
        self.roimask = roimask

    def loadMeta(self, metapath):

        print('\nLoading Metadata\n-----------------------\n')

        assert metapath.endswith('.yaml'), 'Metadata was not a valid yaml file.'
        meta = mm.readYaml(metapath)
        self.meta = meta

    def dFoF_brain(self, movie = None, roimask = None, 
        bounding_box = None):

        if roimask is None:
            if hasattr(self, 'roimask'):
                roimask = self.boundMask(bounding_box)

        if movie is None:
            movie = self.boundMovie(bounding_box)

        if roimask is not None:
            dfof_brain = dFoF(getMaskedRegion(movie, roimask))
            dfof_brain = rescaleMovie(dfof_brain).astype(movie.dtype)
            movie = rescaleMovie(movie, mask=roimask, maskval=0)
            movie = insertMaskedRegion(movie, dfof_brain, roimask)

        else:
            movie = rescaleMovie(dFoF(movie))

        return movie
            
    def makeGridRois(self, npix = 7, movie = None):

        print('\nMaking Grid Rois\n-----------------------')

        assert hasattr(self, 'rois'), 'Class instance does not have rois\
        Load rois before attempting to downsample them.'

        if movie is None:
            if hasattr(self, 'filtered_movie'):
                print('Creating downsampled ROIs '
                    'from dFoF PCA filtered movie.')
                movie = dFoF(self.filtered_movie)
            else:
                print('Creating downsampled ROIs from dFoF movie.')
                movie = dFoF(self.movie)


        self.ds_rois = makeGridRois(movie, self.rois, npix)

    def drawBoundingBox(self):
        frame = self.movie[0,:,:].copy()
        frame = rescaleMovie(frame, cap=False).astype('uint8')

        # if class has rois loaded, draw them on image
        if hasattr(self, 'rois'):
            rois = self.rois
            for roi in rois:
                polylist = rois[roi]
                cv2.polylines(frame, [polylist.astype(np.int32).reshape((-1,1,2))], 
                    1, 255, 3)

        ROI = drawBoundingBox(frame)

        self.bounding_box = ROI
        self.shape = (self.shape[0], ROI[0][1] - ROI[0][0], 
            ROI[1][1] - ROI[1][0])

    def defineMaskBoundaries(self):
        assert hasattr(self, 'roimask'), ('Define roimask before '
            'finding boundaries')

        row,cols = np.nonzero(self.roimask)
        ROI = np.array([[np.min(row), np.max(row)], [np.min(cols), np.max(cols)]])

        self.bounding_box = ROI

    def boundMovie(self, movie = None, bounding_box = None):
        if bounding_box == None:
            bounding_box = self.bounding_box

        if movie is None:
            movie = self.movie

        ROI = bounding_box
        return movie[:,ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]

    def boundMask(self, bounding_box = None):
        try:
            if bounding_box == None:
                bounding_box = self.bounding_box

            ROI = bounding_box
            return self.roimask[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]
        except:
            return None

    def maskedMovie(self, movie=None):
        assert hasattr(self, 'roimask'), ('Class instance does not have '
            'a roi mask.  Load a mask before attempting to call the '
            'masked movie.')

        if movie is None:
            movie = self.boundMovie()

        return movie * self.boundMask()
    
    def PCAfilter(self, A=None, savedata=True, savefigures=True, gui=True, 
        preload_timecourses=True, preload_thresholds=True, calc_dFoF=True, 
        del_movie=True, PCAtype='ica', n_components=None):
        
        print('\nPCA Filtering\n-----------------------')

        if savedata:
            savepath = os.path.dirname(self.path[0]) + os.path.sep + \
            self.name + '_' + PCAtype.lower() + '.hdf5'
        else:
            savepath = None

        if savedata:
            f = hdf5manager(savepath)
            components = f.load()
        else:
            components = {}

        # Load all attributes of experiment class into expmeta dictionary
        # to keep info in ica and filtered files.
        ignore = ['movie', 'filtered']
        expdict = self.__dict__
        expmeta = {}
        for key in expdict:
            if key not in ignore:
                expmeta[key] = expdict[key]
        components['expmeta'] = expmeta
        print('Saving keys under expmeta in PC components:')
        for key in expmeta:
            print(key)

        if savedata:
            f.save(components)

        # calculate decomposition:
        if 'vector' and 'eig_vec' and 'eig_val' in components:
            # if data was already in the save path, use it
            print('Found pca decomposition in components')
        else:

            if hasattr(self, 'roimask'):
                roimask = self.boundMask()
            else:
                roimask = None

            if A is None:
                A = self.boundMovie()
                
                if calc_dFoF:
                    A = dFoF(A)

            if del_movie:
                print('Deleting movie to save space..')
                del self.movie

            #drop dimension and flip to prepare timecourse for PCA
            shape = A.shape
            t,x,y = shape
            vector = A.reshape(t, x*y)
            vector = vector.T # now vector is (x*y, t) for PCA along x*y dimension
            print('M has been reshaped from {0} to {1}\n'.format(A.shape, 
                vector.shape))
            components = wbpca.PCA_project(vector, shape, roimask=roimask, 
                savepath=savepath, PCAtype=PCAtype, n_components=n_components)
            components['expmeta'] = expmeta


        if preload_timecourses:
            print('Preloading timecourses...')
            if 'timecourses' not in components:
                components = wbpca.getTimecourse(components)
                if savedata:
                    print('saving timecourses...')
                    f.save({'timecourses':components['timecourses']})
                    if 'pc_xcorr' in components.keys():
                        f.save({'pc_xcorr':components['pc_xcorr']})
            else:
                print('Timecourses already rebuilt.')

        if (preload_thresholds) and (PCAtype == 'ica'):
            print('Preloading domain thresholds...')
            if 'thresholds' not in components:
                try:
                    eb_filter = wbp.removeOutliers(components['eig_vec'].copy())
                    _, thresholds = wbp.getIcaDomains(eb_filter, return_thresh=True)
                    components['thresholds'] = thresholds
                    print('done!')
                    
                    print('Calculating domain flipping...')
                    flipped = flipICAcomponents(eig_vec, thresholds)

                    if savedata:
                        print('saving thresholds...')
                        f.save({'thresholds':thresholds, 'flipped':flipped})
                except:
                    print('Failed!! Error in threshold calculation!!')

        if savefigures:
            figpath = os.path.dirname(self.path[0]) + os.path.sep + \
            self.name + '_components'
            wbpca.PCfigure(components, figpath)

        if gui:
            components = wbpca.runPCAgui(components, savepath=savepath)
            self.filtered = wbpca.PCA_rebuild(components)
            components['filter']['noise_components'] = components['noise_components']

        return components

    def binarizeMovie(self, method = 'Otsu'):

        print('\nMovie Binarization\n-----------------------')

        if hasattr(self, 'filtered_movie'):
            print('binarizing on PCA-filtered movie')
            A = self.filtered_movie
        else:
            print('No PCA-filtered movie')
            print('binarizing bound movie')
            A = self.boundMovie()
            A = dFoF(A)

        if hasattr(self, 'roimask'):
            print('Using roi mask to mask body')
            A = self.maskedMovie(A)
        else:
            print('No mask found')

        print('\n')
        binarymask, mask_params = wbf.binarizeMovie(A, method)
        self.binary_params = mask_params
        self.binary = binarymask
        print('\n')

    def save(self, path, saveraw = False):

        if saveraw is False:
            data = self.__dict__
            if 'movie' in data:
                print('Not saving raw movie data')
                del data['movie']
        else:
            data = self.__dict__

        f = hdf5manager(path)
        f.save(self)

# Used for testing purposes.  This section is called only when file is run 
# through command line, not imported.
if __name__ == '__main__': 
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--movie', type = argparse.FileType('r'), 
        nargs = '+', required = False, 
        help = 'path to the image to be scanned.')
    ap.add_argument('-e', '--experiment', 
        nargs = 1, required = False, 
        help = 'name of experiment (YYYYMMDD_EE) for loading associated files.\
            Requires folder argument -f')
    ap.add_argument('-f', '--folder', 
        nargs = 1, required = False, 
        help = 'name of experiment to load associated files.  \
            Requires experiment argument -e')
    ap.add_argument('-o', '--output', type = argparse.FileType('a+'),
        nargs = 1, required = False, 
        help = 'path to the output experiment file to be written.  '
        'Must be .hdf5 file.')
    ap.add_argument('-r', '--rois', type = argparse.FileType('r'),
        nargs = 1, required = False,
        help = 'path to .zip file with .rois, or .roi file containing '
        'ROIs to associate with video object.')
    ap.add_argument('-s', '--save', type = argparse.FileType('w'),
        nargs = 1, required = False,
        help = 'path at which video file will be saved')
    ap.add_argument('-d', '--downsample', type = int,
        nargs = 1, required = False,
        help = 'factor to downsample videos by for initial loading')
    args = vars(ap.parse_args())

    # Find all movie and roi paths, load them
    if (args['folder'] is not None) and (args['experiment'] is not None):
        print(args['folder'][0])
        files = fm.experimentSorter(args['folder'][0], args['experiment'][0])
        pathlist = files['movies']

        if len(files['roi']) > 0:
            roipath = files['roi'][0]
        else:
            roipath = None
            print('No roipath found.')

        if len(files['meta']) > 0:
            metapath = files['meta'][0]
        else:
            metapath = None
            print('No metadata found.')
    else:
        pathlist = [path.name for path in args['movie']]
        
        if args['rois'] is not None:
            roipath = args['rois'][0].name
        else:
            roipath = None

        metapath = None

    print('\n{0} video files found:'.format(len(pathlist)))
    for path in pathlist:
        print('\t' + path)
    
    print('Rois:\n\t'+roipath)
    print('Metadata:\n\t'+metapath)

    # load files if they're given
    if len(pathlist) > 0:
        if args['downsample'] is not None:
            downsample = args['downsample'][0]
        else:
            downsample = False

        exp = wholeBrain(pathlist, downsample=downsample)

        if roipath is not None:
            print('Roi path found at:', roipath)
            exp.loadROIs(roipath)
            exp.defineMaskBoundaries()
        else:
            print('No roi path found.')

        if metapath is not None:
            print('Metadata path found at:', metapath)
            exp.loadMeta(metapath)
        else:
            print('No metadata path found.')
    else:
        exp = None

    # find output file for saving exp, make sure it's a valid .hdf5 file
    output = args['output']
    if output is not None:
        output = output[0].name
        print('Found output file at:', output)
        assert(output.endswith('.hdf5')), 'Output was not .hdf5 file'
    else:
        print('No output file found')

    # find savepath for use in saving video files
    if args['save'] is not None:
        save = save[0].name
        print('Found save path at:', output)
    else:
        print('No save path found')
    


    # For testing different features
    #-----------------------------------------------------------
    if exp is not None:
        savepath = args['save']
        if savepath is not None:
            playMovie(exp.movie)
            saveFile(savepath, 
                applyVideoColormap(colormapBoundary(exp.movie)))
            saveFile(savepath, exp.movie)

        if output is not None:
            exp.drawBoundingBox()
            exp.movie = exp.boundMovie()
            exp.PCAfilter(exp.movie)
            exp.save(output)
        else:
            exp.makeGridRois()
            # ds_rois = testvideo.ds_rois
            exp.save('test.hdf5')
