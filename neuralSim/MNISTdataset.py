#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:56:50 2018

@author: jkpark
"""

import numpy as np
import os.path
import pickle as pickle
import time as time
import struct
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

def getLabeledData(filename, training=True):
    '''
    Read input-vector (image) and target class (label, 0-9) and return it as list of tuples.
    '''
    MNIST_data_path = './MNIST/'
    start = time.time()
    if os.path.isfile(MNIST_data_path + '%s.pickle' % filename):
        data = pickle.load(open(MNIST_data_path + '%s.pickle' % filename, 'rb'))
        print("Generated files already exist. Data will be loaded.")
    else:
        if training:
            images = open(MNIST_data_path + 'train-images-idx3-ubyte', 'rb')
            labels = open(MNIST_data_path + 'train-labels-idx1-ubyte', 'rb')
            print("Training data set is decoded.")
        else:
            images = open(MNIST_data_path + 't10k-images-idx3-ubyte', 'rb')
            labels = open(MNIST_data_path + 't10k-labels-idx1-ubyte', 'rb')
            print("Test data set is decoded.")
            
        # Get metadata for images
        images.read(4) # skip the magic number
        number_of_images = struct.unpack('>I', images.read(4))[0]
        rows = struct.unpack('>I', images.read(4))[0]
        cols = struct.unpack('>I', images.read(4))[0]
        
        # Get metadata for labels
        labels.read(4) # skip the magic_number
        N = struct.unpack('>I', labels.read(4))[0]
        
        if number_of_images != N:
            raise Exception('The number of labels did not match the number of images')
            
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8) # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8) # Initialize numpy array
        
        for i in range(N):
            if i % 1000 == 0:
                print("i: ", i)
            x[i] = [[struct.unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = struct.unpack('>B', labels.read(1))[0]
            
        data = {'image': x, 'label': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open(MNIST_data_path + '%s.pickle' % filename, 'wb'))
        
    print("Get labeled data takes", round(time.time()-start, 5), "seconds.")  
    
    return data

def getDataSet():
    testSet = getLabeledData('MNISTtestset', training=False)
    trainingSet = getLabeledData('MNISTtrainingset', training=True)
    
    return testSet, trainingSet

def getDataDistribution(dataset, nClass):
    datasetMean = np.zeros(nClass)
    datasetStd = np.zeros(nClass)
    
    datasetLength = len(dataset['label'])
    datasetSum = np.zeros(datasetLength)
    
    for i in range(datasetLength):
        data = np.array(dataset['image'][i].flatten(), dtype=int)
        datasetSum[i] = sum(data)
        
    for i in range(nClass):
        temp = datasetSum[np.where(dataset['label'] == i)[0]]
        datasetMean[i] = np.mean(temp)
        datasetStd[i] = np.std(temp)
        
    return datasetMean, datasetStd

def plot2DImage(dataset, nrow, ncol, obj='all', axis='on'):
    
    if obj == 'all':
        indices = np.random.randint(0, len(dataset['label'])-1, nrow*ncol)
    else:
        indices = np.where(dataset['label'].flatten() == obj)[0]
    
    a = np.array([63, 70, 121, 104, 152, 50, 2200, 2600, 2950, 2954])
    plt.figure()
    for index in range(nrow*ncol):
        plt.subplot(nrow, ncol, index+1)
        #image = plt.imshow(dataset['image'][indices[index]], interpolation='nearest', vmin=0, vmax=255, cmap=cmap.get_cmap('Greys'))
        image = plt.imshow(dataset['image'][a[index]], interpolation='nearest', vmin=0, vmax=255)
        plt.axis(axis)
        #plt.colorbar(image)
    plt.show()
    