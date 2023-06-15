# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:39:15 2022

@author: jongkil
"""
import random
import numpy as np
from datetime import datetime


class poissonSpike:
    def SingleNeuronPoissonSpikeTrain(neuronID, rate, dt, period):
        '''
        neuronID : Neuron ID being stimulated by the spike train
        rate : The firing rate of the poisson spike train (in Hz)
        dt : The timewindow of unit timestep (in seconds)
        period : The time period (in seconds)
        
        return (neuron, spikeTime)
        '''
        random.seed(datetime.now())
        spikeTime = np.array([x for x in np.linspace(0, period, period/dt + 1) if random.random() < rate * dt])
        neuron = np.ones(len(spikeTime)) * neuronID
        
        return np.vstack((neuron, spikeTime))
        
    def MultipleNeuronPoissonSpikeTrain(neuronNum, rate, dt, period, offset):
        
        random.seed(datetime.now())

        sampleCount = int(period/dt + 1)

        spikeTime = np.zeros(sampleCount * len(neuronNum))
        neuron = np.zeros(sampleCount * len(neuronNum))
                
        uBound = 0
        
        if type(neuronNum) == int:
            if type(rate) == int:
                for indexNeuronNum in range(neuronNum):
                    spikeTimeTemp = np.where(np.random.random_sample(sampleCount) < rate * dt)[0] * dt + offset
                    neuronTemp = np.ones(len(spikeTimeTemp)) * indexNeuronNum
                    
                    spikeTime[uBound : uBound + len(spikeTimeTemp)] = spikeTimeTemp
                    neuron[uBound : uBound + len(neuronTemp)] = neuronTemp
                    
                    uBound += len(spikeTimeTemp)
            else: # [rate]
                for indexNeuronNum in range(neuronNum):
                    spikeTimeTemp = np.where(np.random.random_sample(sampleCount) < rate[indexNeuronNum] * dt)[0] * dt + offset
                    neuronTemp = np.ones(len(spikeTimeTemp)) * indexNeuronNum
                    
                    spikeTime[uBound : uBound + len(spikeTimeTemp)] = spikeTimeTemp
                    neuron[uBound : uBound + len(neuronTemp)] = neuronTemp
                    
                    uBound += len(spikeTimeTemp)
        else: # [neuronNum]
            if type(rate) == int:
                for indexNeuronNum in neuronNum:
                    spikeTimeTemp = np.where(np.random.random_sample(sampleCount) < rate * dt)[0] * dt + offset
                    neuronTemp = np.ones(len(spikeTimeTemp)) * indexNeuronNum
                    
                    spikeTime[uBound : uBound + len(spikeTimeTemp)] = spikeTimeTemp
                    neuron[uBound : uBound + len(neuronTemp)] = neuronTemp
                    
                    uBound += len(spikeTimeTemp)
            elif len(neuronNum) == len(rate):
                for indexNeuronNum in range(len(neuronNum)):
                    spikeTimeTemp = np.where(np.random.random_sample(sampleCount) < rate[indexNeuronNum] * dt)[0] * dt + offset
                    neuronTemp = np.ones(len(spikeTimeTemp)) * neuronNum[indexNeuronNum]
                    
                    spikeTime[uBound : uBound + len(spikeTimeTemp)] = spikeTimeTemp
                    neuron[uBound : uBound + len(neuronTemp)] = neuronTemp
                    
                    uBound += len(spikeTimeTemp)
            else:
                print("Failed : The number of neuron does not match with the number of rate")
                return False
            
        spikeTime = spikeTime[0:uBound]
        neuron = neuron[0:uBound]
        
        arg = spikeTime.argsort()

        return neuron[arg], spikeTime[arg]
    
    def PoissonSpikeTrains_Time_Neuron(neuronNum, rate, dt, period, offset):
        result = poissonSpike.MultipleNeuronPoissonSpikeTrain(neuronNum, rate, dt, period, offset)
        
        return [result[1], result[0]]
    
    