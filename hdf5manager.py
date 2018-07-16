#!/usr/bin/env python3

'''
Tools for standardized saving/loading a class or dictionary to a .hdf5 file.
Not all formats are supported, feel free to add a set of save/load functions for 
unsupported formats. 

Strings are saved as attributes of the file; lists of strings are saved as tab 
delimited strings; arrays are saved as datasets.  Dicts are saved as a new folder, 
with data saved as numpy datasets.

Useage:
* listing objects in an hdf5 file:
    f = hdf5manager(mypath)
    f.print()
* saving data to file:
    f = hdf5manager(mypath)
    f.save(mydict)
  OR:
    f.save(myClass)
* loading data from file:
    f = hdf5manager(mypath)
    data = f.load()

Authors: Sydney C. Weiser
Date: 2017-01-13
'''

import h5py
import numpy as np
import os
import pickle

def main():
    '''
    If called directly from command line, take argument passed, and try to read 
    contents if it's an .hdf5 file.
    '''
    import sys
    import os

    print('\nHDF5 Reader\n-----------------------')

    if len(sys.argv) > 1: # if argument was passed
        path = sys.argv[1]

        if (os.path.isfile(path)) & (path.endswith('.hdf5')):
            print('Found .hdf5 file')
            f = hdf5manager(path)
            f.print()
        else:
            print('Not a valid hdf5 file.\nExiting.\n')
    else:
        print('No argument was given.  To read a file, pass an '
            '.hdf5 file as an argument.\n')


class hdf5manager:
    def __init__(self, path, verbose=False):

        assert (path.endswith('.hdf5') | path.endswith('.mat'))
        path = os.path.abspath(path)

        if not os.path.isfile(path):
            # Create the file
            print('Creating file at:', path)
            f = h5py.File(path, 'w')
            f.close()

        self.path = path
        self.verbose = verbose

        if verbose:
            self.print()


    def print(self):
        path = self.path
        print()

        # If not saving or loading, open the file to read it
        if not hasattr(self, 'f'):
            print('Opening File to read...')
            f = h5py.File(path, 'r')
        else:
            f = self.f
    
        if len(list(f.keys())) > 0:
            print('{0} has the following keys:'.format(path))
            for file in f.keys():
                print('\t-',file)
        else:
            print('{0} has no keys.'.format(path))

        if len(list(f.attrs)) > 0:
            print('{0} has the following attributes:'.format(path))
            for attribute in f.attrs:
                print('\t-', attribute)
        else:
            print('{0} has no attributes.'.format(path))

        # If not saving or loading, close the file after finished
        if not hasattr(self, 'f'):
            print('Closing file...')
            f.close()
        print()

    def keys(self):
        # If not saving or loading, open the file to read it
        if not hasattr(self, 'f'):
            f = h5py.File(self.path, 'r')
        else:
            f = self.f

        keys = [key for key in f.attrs]
        keys.extend([key for key in f.keys()])

        if not hasattr(self, 'f'):
            f.close()

        return keys


    def open(self):
        path = self.path
        verbose = self.verbose

        f = h5py.File(path, 'a')
        self.f = f

        self.print() # print all variables

        if verbose:
            print('File is now open for manual accessing.\n'
                'To access a file handle, assign hdf5manager.f.[key] to a handle'
                ' and pull slices: \n'
                '\t slice = np.array(handle[0,:,1:6])\n'
                'It is also possible to write to a file this way\n'
                '\t handle[0,:,1:6] = np.zeros(x,y,z)\n')

    def close(self):
        self.f.close()
        del self.f

    def load(self, target=None, ignore=None):
        path = self.path
        verbose = self.verbose

        def loadDict(f, key):
            # Load dict to key from its folder
            print('\t\t-', 'loading', key, 'from file...')
            g = f[key]

            print('\t\t-', key, 'has the following keys:')
            print('\t\t  ', ', '.join([gkey for gkey in g.keys()]))

            data = {}
            if g.keys().__len__() > 0:
                for gkey in g.keys():
                    if type(g[gkey]) is h5py.Group:
                        data[gkey] = loadDict(g, gkey)
                    elif type(g[gkey]) is h5py.Dataset:
                        print('\t\t-', 'loading', key, 'from file...')
                        data[gkey] = np.array(g[gkey])
                    else:
                        print('key was of unknown type', type(gkey))

            print('\t\t-', key, 'has the following attributes:')
            print('\t\t  ', ', '.join([gkey for gkey in g.attrs]))

            for gkey in g.attrs:
                print('\t\t\t', gkey + ';', type(g.attrs[gkey]).__name__)
                print('\t\t\t-', 'loading', gkey, 'from file...')
                if type(g.attrs[gkey]) is str:
                    data[gkey] = g.attrs[gkey]
                elif type(g.attrs[gkey] is np.void):
                    out = g.attrs[gkey]
                    data[gkey] = pickle.loads(out.tostring())
                else:
                    print('INVALID TYPE!!')

            return data

        f = h5py.File(path, 'a') # Open file for access
        self.f = f # set to variable so other functions know file is open

        if target is None:
            print('No target key specified; loading all datasets')
            keys = f.keys()
            attrs = f.attrs
        else:
            assert (type(target) is str) or (type(target) is list), 'invalid target'
            if type(target) is str:
                target = [target]

            keys = []
            attrs = []

            for item in target:

                if (type(item) is str) & (item in f.keys()):
                    print('Target key found:', item)
                    keys.append(item)

                elif (type(item) is str) & (item in f.attrs):
                    print('Target attribute found:', item)
                    attrs.append(item)

                else:
                    print('Target was not valid:', target)
                    return None

        print('\nLoading datasets from hdf5 file:')
        data = {}
        for key in keys:
            print('\t', key + ';', type(f[key]).__name__)

            if key == ignore:
                print('\t\t- ignoring key:', key)
            else:
                if type(f[key]) is h5py.Group:
                    data[key] = loadDict(f, key)
                elif type(f[key]) is h5py.Dataset:
                    print('\t\t-', 'loading', key, 'from file...')
                    data[key] = np.array(f[key])
                else:
                    print('\t\t- attribute was unsupported type:', 
                        type(f[key]).__name__)

        for key in attrs:
            print('\t', key + ';', type(f.attrs[key]).__name__)

            if key == ignore:
                print('ignoring attribute:', key)
            else:
                print('\t\t-', 'loading', key, 'from file...')
                if type(f.attrs[key]) is str:
                    data[key] = f.attrs[key]
                elif type(f.attrs[key] is np.void):
                    out = f.attrs[key]
                    data[key] = pickle.loads(out.tostring())

        print('Keys extracted from file:')
        print('\t', ', '.join([key for key in data.keys()]))
        print('\n\n')

        del self.f
        f.close()

        if (type(target) is list) and (len(target) == 1):
            data = data[target[0]]

        return data

    def save(self, data):
        # data is a class file or dict of keys/data
        path = self.path
        verbose = self.verbose

        '''
        Saves a class or dict to hdf5 file.

        Note that lists of numbers are not supported, only np arrays or 
        lists of strings.
        '''

        # Functions to save each type of data:
        # --------------------------------------------------------------

        def saveDict(f, fdict, key):
            # Write dict to key as its own folder
            print('\t\t-', 'writing', key, 'to file...')

            # Delete if it exists
            if key in f:
                print('\t\t-', 'Removing', key, 'from file')
                del f[key]

            g = f.create_group(key)
            data_d = fdict

            for dkey in fdict:

                if (type(fdict[dkey]) is str):
                    saveString(g, fdict[dkey], dkey)
                elif type(fdict[dkey]) is np.ndarray:
                    saveArray(g, fdict[dkey], dkey)
                elif type(fdict[dkey]) is dict:
                    saveDict(g, fdict[dkey], dkey)
                else:
                    print('\t\t- attribute was unsupported type:', 
                        type(fdict[dkey]).__name__)
                    print('\t\tAttempting to save pickle dump of object')
                    try:
                        saveOther(g, fdict[dkey], dkey)
                        print('\t\tSaved succesfully!')
                    except:
                        print('\t\tFailed..')

            print('\t\t-', key, 'has the following keys:')
            print('\t\t  ', ', '.join([dkey for dkey in g.keys()]))

            print('\t\t-', key, 'has the following attributes:')
            print('\t\t  ', ', '.join([dkey for dkey in g.attrs]))            

        def saveString(f, string, key):
            # Write all strings as attributes of the dataset
            print('\t\t-', 'writing', key, 'to file...')
            f.attrs[key] = string

        def saveArray(f, array, key):
            # Check if key exists, and if entry is the same as existing value
            if key in f.keys():
                if (not np.array_equal(array, 
                        f[key])):
                    print('\t\t-', key, 'in saved file is inconsistent '
                        'with current version')
                    print('\t\t-', 'deleting', key, 'from file')
                    del f[key]
                    print('\t\t-', 'writing', key, 'to file...')
                    f.create_dataset(key, data=array, chunks=None)
                else:
                    print('\t\t-', key, 'in saved file is the same as '
                        'the current version')
            else:
                print('\t\t-', 'writing', key, 'to file...')
                f.create_dataset(key, data=array, chunks=None)

        def saveOther(f, obj, key):
            # Compress to bytestring using pickle, save similar to string
            # Write all strings as attributes of the dataset
            print('\t\t-', 'writing', key, 'to file...')

            bstring = pickle.dumps(obj)
            f.attrs[key] = np.void(bstring)

        # Check input data type, open file:
        # --------------------------------------------------------------

        # If data is not a dictionary, assume 
        if type(data) is not dict:
            # Get dictionary of all keys in class type
            data = data.__dict__

        if verbose:
            print('Attributes found in data file:')
            for key in data.keys():
                print('\t', key, ':', type(data[key]))

        f = h5py.File(path, 'a')
        self.f = f

        if verbose:
            self.print()

        # Loop through keys and save them in hdf5 file:
        # --------------------------------------------------------------
        print('\nSaving class attributes:')
        for key in data.keys():
            print('\t', key + ';', type(data[key]).__name__)
            if (type(data[key]) is str):
                saveString(f, data[key], key)
            elif type(data[key]) is np.ndarray:
                saveArray(f, data[key], key)
            elif type(data[key]) is dict:
                saveDict(f, data[key], key)
            else:
                print('\t\t- attribute was unsupported type:', 
                    type(data[key]).__name__)
                print('\t\tAttempting to save pickle dump of object')
                try:
                    saveOther(f, data[key], key)
                    print('\t\tSaved succesfully!')
                except:
                    print('\t\tFailed..')
        
        self.print()

        del self.f
        f.close()

if __name__ == '__main__':
    main()