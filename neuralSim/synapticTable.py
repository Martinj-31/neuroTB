#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:53:20 2018

@Author: Jongkil Park
@Company: KIST
@Description: This class is for generating a synaptic connectivity table. 
              This class supports the creation of random connectivity.
              Also, it is possible to read connection from a file.
              
              A synapse is = [source, destination, weight, delay, trainable]
"""
import numpy as np
import time
import pickle
import os.path
import matplotlib.pyplot as plt

import neuralSim.parameters as param

class synapticTable:
    
    objType = 'SynTable'
    
    # %% Synaptic table analysis
    def plot(self, post, pre='all', mode='dist', axisSize=28):
        """
        This function visualize synaptic weights. If the number of target postsynaptic
        neurons are greater than 100, it raises error for avoiding unwanted computer resource problem.
        
        Parameters
        -----------------------------------------------------------------------
        post : array_like
               Specific target neuron where synaptic connections connected to.
        
        pre : array_like, optional
              If the parameter is not specified, the function plots all presynaptic connections 
              connected to target postsynaptic neuron. It can be specified with an array of 
              presynaptic neuron IDs.
                
        mode : str, optional
               'dist' : plotting synaptic distribution
               'image' : plotting in 2D-image
        """
        post = np.array(post)
        
        if len(post) > 257:
            print("Error : it might cause significant perfomance degradation.")
            print("        Please change range of target neurons. (synapticTable.py)")
            return False
        
        for index in range(len(post)):
            if len(post) == 1:
                plt.figure()
            else:
                
                subplotx = int(np.ceil(np.sqrt(len(post))))
                subploty = int(len(post) / subplotx)
                plt.subplot(subplotx, subploty, index+1)
            
            targetNeuron = np.array(post[index], dtype=int)
            postIndices = np.array(np.array(self.postIndexPointer)[targetNeuron], dtype=int)
            synapses = np.array([self.synapses.T[0][postIndices], self.synapses.T[2][postIndices]])
            
            if type(pre) != str:
                pre = np.array(pre)
                adjustIndex = np.array([x == y for y in synapses[0] for x in [pre]]) + 0
                adjustIndex = np.where(adjustIndex == 1)[0]
                synapses = np.array([synapses[0][adjustIndex], synapses[1][adjustIndex]])
                
                synIndexMin = np.min(pre)
                synIndexMax = np.max(pre)
            else:
                if pre != 'all':
                    print("Error : optional argument is wrong.")
                    return False
                else: # pre == 'all'
                    synIndexMin = 0
                    synIndexMax = axisSize * axisSize
                    
            if mode == 'dist':
                plt.plot(synapses[0], synapses[1], 'b.')
            elif mode == 'image':
                '''
                Normally, this is for plotting receptive fileds.
                Need to check cordination and size of synaptic connections.
                '''
                synapses[0] = synapses[0] - synIndexMin
                #synIndexRange = synIndexMax - synIndexMin + 1
                
                #synFieldXSize = int(np.ceil(np.sqrt(synIndexRange)))
                #synFieldYSize = int(synIndexRange/synFieldXSize)
                
                #synField = np.zeros(synFieldXSize * synFieldYSize)
                synField = np.zeros(1024)
                np.add.at(synField, np.array(synapses[0], dtype=int), synapses[1])
                
                synField = np.reshape(synField[0:axisSize*axisSize], [axisSize, axisSize])

                plt.imshow(synField, interpolation='nearest', vmin=0, vmax=64)    
                plt.axis('off')
                
        plt.show()
        
        return True
    
    
    def hist(self, pre='all', post='all', b=20):
        plt.figure()
        
        if pre != 'all' or post != 'all':
            if pre != 'all':
                preIndices = np.array(np.concatenate(np.array(self.preIndexPointer)[np.array(pre, dtype=int)]), dtype=int)
            if post != 'all':
                postIndices = np.array(np.concatenate(np.array(self.postIndexPointer)[np.array(post, dtype=int)]), dtype=int)
            
            Indices = np.append(preIndices, postIndices)
    
            plt.hist(self.synapses.T[2][Indices], b)
        else:
            plt.hist(self.synapses.T[2], b)
            
        plt.ylabel('Count', size=13)
        plt.xlabel('Weight', size=13)
                    
    # %% Create synaptic connections
    def createConnections(self, source, destination, mode='all', probability=1, weight='random', wmax = param.synMax, delay=0, dmax = param.dlyMax, synType='exc', trainable=False):
        '''
        mode : 'all' all-to-all
               'one' one-to-one
        source : 
        destination :
        probability :
        weight :
        delay :
        synType : 
        trainable :
        '''
        
        if type(source) not in [np.ndarray, list, range, np.float64]:
            if source.objType == 'inputSpikes':
                source = np.array(source.ID())
            elif source.objType == 'neuPort' or source.objType == 'neuGroup':
                source = np.array(source.ID())
                
        if type(destination) not in [np.ndarray, list, range, np.float64]:
            if destination.objType == 'neuPort' or destination.objType == 'neuGroup':
                destination = np.array(destination.ID())
                        
        
        if synType == 'exc':
            wGain = 1
        elif synType == 'inh':
            wGain = -1
            
        SynStartTime = time.time()
        destination = np.array(destination, dtype=int)
        source = np.array(source, dtype=int)

        if mode == 'all':
            for sourceItem in source :
                
                refArray = np.ones(len(destination))
                sourceID = refArray * sourceItem
                train = np.ones(len(destination)) * trainable
                
                if type(weight) == str:
                    if weight == 'random':
                        w = np.random.random_sample(len(destination)) * wmax * wGain
                elif type(weight) == np.ndarray:
                    w = weight * wGain
                else: # integer input
                    w = refArray * weight * wGain
                
                if delay == 'random' :
                    d = np.random.random_sample(len(destination)) * dmax
                else:
                    d = refArray * delay
                
                if probability == 1:
                    if len(self.synapses) == 0:
                        self.synapses = np.array([sourceID, destination, w, d, train]).T
                    else:
                        self.synapses = np.append(self.synapses, np.array([sourceID, destination, w, d, train]).T, axis=0)
                    
                    self.preIndexPointer[int(sourceItem)] = np.append(self.preIndexPointer[int(sourceItem)], np.linspace(0, len(destination)-1, len(destination), dtype=int) + self.synapseCount)

                    if trainable  == True:
                        self.preIndexPointerPlastic[int(sourceItem)] = np.append(self.preIndexPointerPlastic[int(sourceItem)], np.linspace(0, len(destination)-1, len(destination), dtype=int) + self.synapseCount)
                        self.PreToPost[int(sourceItem)] = np.append(self.PreToPost[int(sourceItem)], destination)                        
                        
                    for index in range(len(destination)):
                        self.postIndexPointer[int(destination[index])] = np.append(self.postIndexPointer[int(destination[index])], int(self.synapseCount + index))
                        
                        if trainable == True:
                            self.postIndexPointerPlastic[int(destination[index])] = np.append(self.postIndexPointerPlastic[int(destination[index])], int(self.synapseCount + index))
                            self.PostToPre[int(destination[index])] = np.append(self.PostToPre[int(destination[index])], int(sourceItem))
                            
                    self.synapseCount += len(destination)
                else :
                    indices = np.where(np.random.random_sample(len(destination)) < probability)[0]
    
                    if len(self.synapses) == 0:
                        self.synapses = np.array([sourceID[indices], destination[indices], w[indices], d[indices], train[indices]]).T
                    else:
                        self.synapses = np.append(self.synapses, np.array([sourceID[indices], destination[indices], w[indices], d[indices], train[indices]]).T, axis=0)
                        
                    self.preIndexPointer[int(sourceItem)] = np.append(self.preIndexPointer[int(sourceItem)], np.linspace(0, len(indices)-1, len(indices), dtype=int) + self.synapseCount)    
                    
                    if trainable == True:
                        self.preIndexPointerPlastic[int(sourceItem)] = np.append(self.preIndexPointerPlastic[int(sourceItem)], np.linspace(0, len(indices)-1, len(indices), dtype=int) + self.synapseCount)    
                        self.PreToPost[int(sourceItem)] = np.append(self.PreToPost[int(sourceItem)], destination[indices])
                        
                    for index in range(len(indices)):
                        self.postIndexPointer[int(destination[indices[index]])] = np.append(self.postIndexPointer[int(destination[indices[index]])], int(self.synapseCount + index))
                        
                        if trainable == True:
                            self.postIndexPointerPlastic[int(destination[indices[index]])] = np.append(self.postIndexPointerPlastic[int(destination[indices[index]])], int(self.synapseCount + index))
                            self.PostToPre[int(destination[indices[index]])] = np.append(self.PostToPre[int(destination[indices[index]])], int(sourceItem))
                        
                    self.synapseCount += len(indices)
                    
        elif mode == 'one':
            # check dimension
            if len(source) != len(destination) :
                print("WARNINGS : The number of source does not match with the number of destination.")
                return False
            else:
                refArray = np.ones(len(destination))
                train = np.ones(len(destination)) * trainable
                
                if type(weight) == str:
                    if weight == 'random':
                        w = np.random.random_sample(len(destination)) * wmax * wGain
                elif type(weight) == np.ndarray:
                    w = weight * wGain
                else: # integer input
                    w = refArray * weight * wGain
                    
                    
                if delay == 'random' :
                    d = np.random.random_sample(len(destination)) * dmax
                else:
                    d = refArray * delay
                
                if probability == 1:
                    if len(self.synapses) == 0:
                        self.synapses = np.array([source, destination, w, d, train]).T
                    else:
                        self.synapses = np.append(self.synapses, np.array([source, destination, w, d, train]).T, axis=0)
                    
                    for index in range(len(source)):
                        self.preIndexPointer[int(source[index])] = np.append(self.preIndexPointer[int(source[index])], int(self.synapseCount + index))
                        self.postIndexPointer[int(destination[index])] = np.append(self.postIndexPointer[int(destination[index])], int(self.synapseCount + index))
                        
                        if trainable == True:
                            self.preIndexPointerPlastic[int(source[index])] = np.append(self.preIndexPointerPlastic[int(source[index])], int(self.synapseCount + index))
                            self.postIndexPointerPlastic[int(destination[index])] = np.append(self.postIndexPointerPlastic[int(destination[index])], int(self.synapseCount + index))
                            
                            self.PreToPost[int(source[index])] = np.append(self.PreToPost[int(source[index])], int(destination[index]))
                            self.PostToPre[int(destination[index])] = np.append(self.PostToPre[int(destination[index])], int(source[index]))
                        
                    self.synapseCount += len(destination)
                else :
                    indices = np.where(np.random.random_sample(len(destination)) < probability)[0]
                    
                    if len(self.synapses) == 0:
                        self.synapses = np.array([source[indices], destination[indices], w[indices], d[indices], train[indices]]).T
                    else:
                        self.synapses = np.append(self.synapses, np.array([source[indices], destination[indices], w[indices], d[indices], train[indices]]).T, axis=0)
                        
                    for index in range(len(indices)):
                        self.preIndexPointer[int(source[indices[index]])] = np.append(self.preIndexPointer[int(source[indices[index]])], int(self.synapseCount + index))
                        self.postIndexPointer[int(destination[indices[index]])] = np.append(self.postIndexPointer[int(destination[indices[index]])], int(self.synapseCount + index))
                        
                        if trainable == True:
                            self.preIndexPointerPlastic[int(source[indices[index]])] = np.append(self.preIndexPointerPlastic[int(source[indices[index]])], int(self.synapseCount + index))
                            self.postIndexPointerPlastic[int(destination[indices[index]])] = np.append(self.postIndexPointerPlastic[int(destination[indices[index]])], int(self.synapseCount + index))
                        
                            self.PreToPost[int(source[indices[index]])] = np.append(self.PreToPost[int(source[indices[index]])], int(destination[indices[index]]))
                            self.PostToPre[int(destination[indices[index]])] = np.append(self.PostToPre[int(destination[indices[index]])], int(source[indices[index]]))
                        
                    self.synapseCount += len(indices)
                
        print("Synaptic connection table is created. (Elapsed ", round(time.time()-SynStartTime, 3), "seconds.)")
    
    # %% create a synapse table from a weight matrix
    def createFromWeightMatrix(self, source, destination, weights, trainable=False):
        '''
        This function generates a synapse table using a predefined weight matrix
        generated other environments such as a tensorflow simulation.
        
        This function does not change portCount and maxPortNeuronCount.
        
        Warning : It is not yet tested in a simulation environment. 
                  Some pointers might be generated incorrectly, and need to be verified. 

        Parameters
        ----------
        source : ndarray
            The neuron IDs of source neurons.
        destination : ndarray
            The neuron IDs of destination neurons.
        weights : ndarray
            A weight matrix.
        trainable : bool, optional
            The default is False.

        Returns
        -------
        None.

        '''
        print("Warning : trainable is not importing. Need to check the code later.")
        
        
        # Check the dimension of a weight matrix to the size of source and destination
        source = np.array(source, dtype=np.dtype(int))
        destination = np.array(destination, dtype=np.dtype(int))
        
        destLen = len(destination)
        srcLen = len(source)
        wMatrixShape = np.shape(weights)
        
        if (wMatrixShape[0] != srcLen or wMatrixShape[1] != destLen):
            print("Error : need to check the dimension of a weight matrix to the size of source and destination")

        
        # Generate synapses
        sourceID = np.repeat(np.array(source), destLen)
        destinationID = np.tile(np.array(destination), srcLen)
        delay = np.zeros(destLen * srcLen)
        
        if trainable == False:
            train = np.zeros(destLen * srcLen)
        else:
            train = np.ones(destLen * srcLen)
        
        SynCount_base = len(self.synapses)
        
        
        if SynCount_base == 0:
            self.synapses = np.array([sourceID, destinationID, weights.flatten(), delay, train]).T
        else:
            self.synapses = np.append(self.synapses, np.array([sourceID, destinationID, weights.flatten(), delay, train]).T, axis=0)
        
        # preIndexPointer, preIndexPointerPlastic, and PreToPost
        for index in range(srcLen):
            srcIndex = SynCount_base + np.where(sourceID == source[index])[0]
            
            self.preIndexPointer[source[index]] = np.append(self.preIndexPointer[source[index]], srcIndex)
            
            
            '''
            if trainable == True:
                self.preIndexPointerPlastic[source[index]] = np.append(self.preIndexPointerPlastic[source[index]], srcIndex)    
                self.PreToPost[source[index]] = np.append(self.PreToPost[source[index]], self.synapses.T[1][srcIndex])
            '''
        # postIndexPointer, postIndexPointerPlastic, and PostToPre
        for index in range(destLen):
            destIndex = SynCount_base + np.where(destinationID == destination[index])[0]
            
            self.postIndexPointer[destination[index]] = np.append(self.postIndexPointer[destination[index]], destIndex)
            '''
            if trainable == True:
                self.postIndexPointerPlastic[destination[index]] = np.append(self.postIndexPointerPlastic[destination[index]], destIndex)
                self.PostToPre[destination[index]] = np.append(self.PostToPre[destination[index]], self.synapses.T[0][destIndex])
            '''    

        # synapseCount
        self.synapseCount += destLen * srcLen
        
    # %% create a synapse table from weight memory
    def createFromSynapseList(self, inputList):
        
        print("Warning : trainable is not importing. Need to check the code later.")
        
        source = np.array(inputList[0], dtype=np.dtype(int))
        destination = np.array(inputList[1], dtype=np.dtype(int))
        weights = np.array(inputList[2])
        train = np.array(inputList[3])
        
        srcLen = len(source)
        #destLen = len(destination)
        
        delay = np.zeros(srcLen)
        
        SynCount_base = len(self.synapses)
        
        if SynCount_base != 0:
            print("Error : the synapse table was already created. The function does not support appending a new synapse table to an old one.")
            #self.synapses = np.append(self.synapses, np.array([source, destination, weights.flatten(), delay, train]).T, axis=0)
        else: # len(self.synapses) == 0
            self.synapses = np.array([source, destination, weights.flatten(), delay, train]).T

            print("Synapses are added to the table.")
            # preIndexPointer, preIndexPointerPlastic, and PreToPost
            for index in np.unique(source):
                srcIndex = np.where(self.synapses.T[0] == index)[0]
                
                self.preIndexPointer[index] = np.append(self.preIndexPointer[index], srcIndex)
                '''
                if train[index] == 1:
                    self.preIndexPointerPlastic[source[index]] = np.append(self.preIndexPointerPlastic[source[index]], srcIndex)    
                    self.PreToPost[source[index]] = np.append(self.PreToPost[source[index]], self.synapses.T[1][srcIndex])
                '''
                    
            # postIndexPointer, postIndexPointerPlastic, and PostToPre
            for index in np.unique(destination):
                destIndex = np.where(self.synapses.T[1] == index)[0]
                
                self.postIndexPointer[index] = np.append(self.postIndexPointer[index], destIndex)
                '''
                if train[index] == 1:
                    self.postIndexPointerPlastic[index] = np.append(self.postIndexPointerPlastic[index], destIndex)
                    self.PostToPre[index] = np.append(self.PostToPre[index], self.synapses.T[0][destIndex])
                '''    
            # synapseCount
            self.synapseCount = srcLen
        
    # %% Read and save synaptic table
    def readSynapseTableFromFile(self, filename):
        if os.path.isfile('%s.pkl'%filename):
            data = pickle.load(open('%s.pkl'%filename, 'rb'))
            
            self.synapses = data['synapses']
            self.preIndexPointer = data['preIndexPointer']
            self.postIndexPointer = data['postIndexPointer']
            self.preIndexPointerPlastic = data['preIndexPointerPlastic']
            self.postIndexPointerPlastic = data['postIndexPointerPlastic']
            self.PreToPost = data['pretopost']
            self.PostToPre = data['posttopre']
            self.portCount = data['portCount']
            self.maxPortNeuronCount = data['maxPortNeuronCount']
            self.synapseCount = data['synapseCount']
            
            print("A synapse table is loaded.")
        else:
            print("Warning : file does not exist.")
            
    
    def saveSynapseTableToFile(self, filename):
        if os.path.isfile('%s.pkl'%filename):
            print("Warning : Same file name exist in the folder.")
        else:
            synapseTable = {'synapses' : self.synapses,
                            'preIndexPointer' : self.preIndexPointer,
                            'postIndexPointer' : self.postIndexPointer,
                            'preIndexPointerPlastic' : self.preIndexPointerPlastic,
                            'postIndexPointerPlastic' : self.postIndexPointerPlastic,
                            'pretopost' : self.PreToPost,
                            'posttopre' : self.PostToPre,
                            'portCount' : self.portCount,
                            'maxPortNeuronCount' : self.maxPortNeuronCount,
                            'synapseCount' : self.synapseCount
                            }
            pickle.dump(synapseTable, open('%s.pkl'%filename, 'wb'))
            print("A synapse table is saved.")
            
    # %% Synapse table lookup
    def synapticTableLookUp(self, source):
        '''
        source : An array of source ID. [[sourceID], ...]
        '''
        synEvents = []
        for i in range(0,self.portCount) :
            synEvents.append(np.array([]))
        
        #LookupStartTime = time.time()
        if self.synapseCount != 0 and len(source) != 0:
            
            SynTranspose = self.synapses.T
            
            indices = np.array(self.preIndexPointer)[np.array(source, dtype=int)]
            
            desEvents = SynTranspose[1:4].T[np.concatenate(indices)]
            
            for portIndex in range(0, self.portCount):
                synEvents[portIndex] = desEvents[np.array(desEvents.T[0]/self.maxPortNeuronCount, dtype=int) == portIndex]
                
                if len(synEvents[portIndex]) != 0:
                    synEvents[portIndex].T[0] = synEvents[portIndex].T[0] % self.maxPortNeuronCount
        
        return synEvents # List of arrays [destination, weight, delay]
    
    # %% Learning rule apply
    def synapticUpdate(self, indices, update):
        if len(indices) > 0 and len(update) > 0:
            # Select trainable synapse
            self.synapses.T[2][indices] = np.clip(self.synapses.T[2][indices] + update, 0, param.synMax)
    
    # %% Configure parameters
    def setPortCount(self, Count):
        # The number of ports assigned to the SRT
        self.portCount = Count
        
    def setMaxNeuronCount(self, maxCount):
        self.maxPortNeuronCount = maxCount
        
    def getPortCount(self):
        return self.portCount
    
    def getMaxNeuronCount(self):
        return self.maxPortNeuronCount
    
    def getSynapseCount(self):
        return self.synapseCount
    
    # %% Get index pointers
    def getIndexPointer(self):
        return self.preIndexPointer, self.postIndexPointer
    
    # %% Fan-in and fan-out analysis
    def averageFanIn(self, indexRange = 'all'):
        
        if indexRange == 'all':
            FanInCount = np.zeros(self.TotalNeuronCount)
            
            for i in range(self.TotalNeuronCount):
                FanInCount[i] = len(self.postIndexPointer[i])
                
        else: # np.ndarray
            indexRange = np.array(indexRange, dtype=np.int32)
            FanInCount = np.zeros(len(indexRange))
            
            count=0
            for i in indexRange:
                FanInCount[count] = len(self.postIndexPointer[i])
                count += 1
        
        
        averageFanIn = np.average(FanInCount)
        
        
        return averageFanIn, FanInCount
    
    def averageFanOut(self, indexRange = 'all'):
        
        if indexRange == 'all':
            FanOutCount = np.zeros(self.TotalNeuronCount)
            
            for i in range(self.TotalNeuronCount):
                FanOutCount[i] = len(self.preIndexPointer[i])
                
        else: # np.ndarray
            indexRange = np.array(indexRange, dtype=np.int32)
            FanOutCount = np.zeros(len(indexRange))
            
            count=0
            for i in indexRange:
                FanOutCount[count] = len(self.preIndexPointer[i])
                count += 1
        
        averageFanOut = np.average(FanOutCount)
            
        return averageFanOut, FanOutCount
    
    
    # %% Initialization
    def __init__(self, pCount=1, maxPNCount=1024, inputNCount=1024, filename=False):
        self.synapses = np.array([])

        self.preIndexPointer = []
        self.postIndexPointer = []
        
        self.preIndexPointerPlastic = []
        self.postIndexPointerPlastic = []

        self.PreToPost = []
        self.PostToPre = []
        
        if inputNCount == 0:
            print("Warning : input neuron count is zero. It might indicates synaptic table is created before input neuron definition.")
        
        self.TotalNeuronCount = pCount * maxPNCount + inputNCount
        
        for index in range(self.TotalNeuronCount):
            self.preIndexPointer.append(np.array([], dtype=int))
            self.postIndexPointer.append(np.array([], dtype=int))
            self.preIndexPointerPlastic.append(np.array([], dtype=int))
            self.postIndexPointerPlastic.append(np.array([], dtype=int))
            self.PreToPost.append(np.array([], dtype=int))
            self.PostToPre.append(np.array([], dtype=int))
            
        # The number of port assigned to the SRT
        self.portCount = pCount
        
        # The maximun number of neurons in a port
        self.maxPortNeuronCount = maxPNCount
        
        self.synapseCount = 0
        
        if filename != False:
            self.readSynapseTableFromFile(filename)
        
        
def layer(start, count):
    return np.linspace(start, start + count - 1, num = count, dtype=int)