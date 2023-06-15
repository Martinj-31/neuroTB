# -*- coding: utf-8 -*-
"""
Created on Thr Jan 17 11:19:25 2019

@Author: Jongkil Park
@Company: KIST

Description: 
"""

import numpy as np
import os.path
import matplotlib.pyplot as plt
import csv

class eventAnalysis:
    
    def plot(self, gCount=1, nCount=1024):    
        plt.figure()
        plt.plot(self.spikes[0], self.spikes[1], 'b.', markersize=3)
        plt.ylabel("NeuronID", size=15)
        plt.xlabel("Time (s)", size=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if gCount > 1:
            for index in range(gCount):
                plt.axhline(y=(index+1) * nCount, color='black', linestyle='-')
        plt.tight_layout()
        plt.show()
        
        return plt
        
    def getSpikeCount(self):
        
        return self.spikeCount
    
    def getAverageFiringRate(self):
        # estimating neuron counts
        nCount = int(np.ceil(np.max(self.spikes[1]/1024))*1024)
        
        firingRate = np.zeros(nCount)
        
        for i in range(nCount):
            
            firingRate[i] = len(np.where(self.spikes[1] == i)[0]) / (self.eventInTime)
        
        return firingRate
        
    def getConfiguration(self, value):
        while True:
            try:
                idx = np.array(self.configuration).T.tolist()[0].index(value)
                return np.array(self.configuration).T.tolist()[1][idx]
            except ValueError:
                return -1
        
        
    # %% event traffic analysis
    def eventTrafficStatPlot(self, group='all'):
    
        if self.statistics_mode == False:
            print("(eventAnalysis.py) Warning: statistics mode was not selected. Need to set '-statistic' option when run NeuPLUSRun().")
        elif self.statistics_mode == True:
            if group == 'all':
                indexGroup = np.linspace(0, 15, 16, dtype=np.int32)
            else:
                indexGroup = np.array(group, dtype=np.int32)
                
            for i in indexGroup:
                plt.plot(self.statistics[i]['time'], self.statistics[i]['count'])
               
            
            plt.ylabel("Spike count", size=15)
            plt.xlabel("Time (s)", size=15)
            
            plt.tight_layout()
            plt.show()
        
        
        
    # %% Power analysis
    def dynamicPowerAnalysis(self, group='all', scale='pJ'):
        if self.statistics_mode == False:
            print("(eventAnalysis.py) Warning: statistics mode was not selected. Need to set '-statistic' option when run NeuPLUSRun().")
        elif self.statistics_mode == True:
            energy_in_pJ = 28.26 # pJ/spike
            
            if scale=='pJ':
                energy_synop = energy_in_pJ # pJ/spike
            elif scale=='nJ':
                energy_synop = energy_in_pJ / 1000
            elif scale=='uJ':
                energy_synop = energy_in_pJ / 1000 / 1000
    
            if group == 'all':
                indexGroup = np.linspace(0, 15, 16, dtype=np.int32)
            else:
                indexGroup = np.array(group, dtype=np.int32)
            
            energy_consum_time = {}
            energy_consum_group = np.zeros(16)
            for i in indexGroup:
                energy_consum_time[i] = self.statistics[i]['count'] * energy_synop
                
                energy_consum_group[i] = sum(energy_consum_time[i])
                
            energy_consum_total = sum(energy_consum_group)
        
        return energy_consum_total, energy_consum_group, energy_consum_time
    
    
    # %% Initialize    
    def __init__(self, foldername, folderInx = 0, fname="RecordingData"):
        self.spikeCount = 0;
        
        preFrameTime = 0;
        
        index = 1;
        
        spikeEvent = []
        spikeTime = []
        
        self.spikes = []
        self.Frames = []
        
        self.statistics_mode = False
        self.statistics={}
        for i in range(16): # there are 16 groups. Each group has 64k neurons
            self.statistics[i] = {}
            self.statistics[i]['time'] = np.array([0])
            self.statistics[i]['count'] = np.array([0])
            
        
        if folderInx == 0:
            folderIndex = 1
            
            while os.path.exists(foldername + str(folderIndex) + '/'):
                self.foldername = foldername + str(folderIndex) + "/"
                folderIndex = folderIndex + 1
        else:
            self.foldername = foldername + str(folderInx) + "/"

        print("---------------- Reading events log file ------------")
        print("Folder name is " + self.foldername)
        
        DataExtension=".nie"
        ConfExtension=".inf"
        
        DataFilename = self.foldername + fname + str(index) + DataExtension
        ConfFilename = self.foldername + fname + ConfExtension
        
        configurationRead = False
        
        # configuration file read
        if os.path.isfile(ConfFilename):
            with open(ConfFilename, "r") as cf:
                self.configuration = [row for row in csv.reader(cf, delimiter='=')]
                configurationRead = True
        
        if configurationRead == True:
            self.FPGAversion = self.getConfiguration("FPGA version")
            self.expTime = float(self.getConfiguration("Experiment time"))
            self.eventInTime = float(self.getConfiguration("Event in time"))
            self.actualTimestep = float(self.getConfiguration("FPGA actual timestep"))
            
            print("FPGA version : ", self.FPGAversion)
            print("Experiment time : ", self.expTime, " second")
            print("Event in time : ", self.eventInTime, " second")
            
        else:
            print("Configuration file read failed.")
            print("Configuration is set to default.")
            
            self.FPGAversion = 2000
            self.expTime = 1
            self.eventInTime = 10
            self.actualTimestep = 0.1 / 1000
           
        # Event analysis            
        while os.path.isfile(DataFilename):
        
            with open(DataFilename, "rb") as f:
                byte = f.read()
                
                eventLength = int(len(byte)/4)
                
                spikeEvent = np.zeros(eventLength)
                spikeTime = np.zeros(eventLength)
                
                tempFrameTime = np.zeros(eventLength)
                tempFrameNumber = np.zeros(eventLength)
                
                eventCount = 0
                frameCount = 0
                
                preFrameTime = 0
                preEventTime = 0
                
                for i in range(eventLength):
                    
                    if((byte[i * 4 + 0] >> 7) == 1):
                        if ((byte[i * 4 + 0] >> 5) == 4): # Time stamp
                            prepreFrameTime = preFrameTime

                            preFrameTime = ((byte[i * 4 + 0] % 32) * 2**24) + ((byte[i * 4 + 1]) * 2**16) + ((byte[i * 4 + 2]) * 2**8) + ((byte[i * 4 + 3]))
                            preEventTime = preFrameTime

                            delta = preFrameTime - prepreFrameTime
                            # print(preFrameTime)
                            if delta != 512:
                                print("(eventAnalysis.py) WARNING: Time stamp difference is not regular.", preFrameTime, delta)
                                print("(eventAnalysis.py) WARNING: It might means that the system lost events via USB communication.")
                                print("(eventAnalysis.py) WARNING: We recommend to have lower 'deltat' for input spike event generation..")
                                
                        elif ((byte[i * 4 + 0] >> 5) == 7): # Frame Count
                            print(frameCount)
                            preEventTimeUpper = preEventTime // 512
                            PartialTime = (byte[i * 4 + 0] % 32) * 2**4 + (byte[i * 4 + 1] >> 4)

                            tempFrameTime[frameCount] = (preEventTimeUpper * 512 + PartialTime) * self.actualTimestep / 1000
                            tempFrameNumber[frameCount] = (byte[i * 4 + 1] % 16) * 2**16 + byte[i * 4 + 2] * 2**8 + byte[i * 4 + 3]
                            
                            frameCount += 1
                        elif ((byte[i * 4 + 0] >> 5) == 6): # Statistics
                            Count = ((byte[4 * i + 1]) << 16) + ((byte[4 * i + 2]) << 8) + ((byte[4 * i + 3]));
                            Group = ((byte[4 * i + 0]) % 16);

                            statTime = preEventTime * self.actualTimestep / 1000
                            
                            self.statistics[Group]['time'] = np.append(self.statistics[Group]['time'], statTime)
                            self.statistics[Group]['count'] = np.append(self.statistics[Group]['count'], Count)
                            
                            self.statistics_mode = True
                            print("Time : ", statTime, " Data sent to group ", Group, " count ",  Count);
                            
                            

                     
                    elif ((byte[i * 4 + 0] >> 7) == 0): # Spike event
                        self.spikeCount += 1
                        
                        # New version
                        neuronID = ((byte[i * 4 + 1] % 16) * 2**16) + (byte[i * 4 + 2] * 2**8) + byte[i * 4 + 3]
                        
                        # 9-bit time stamp
                        neuronTime = (preFrameTime + (byte[i * 4 + 0] % 32) * 2**4 + (byte[i * 4 + 1] >> 4)) * self.actualTimestep / 1000
                        preEventTime = (preFrameTime + (byte[i * 4 + 0] % 32) * 2**4 + (byte[i * 4 + 1] >> 4))
                        
                        
                        # 11-bit time stamp
                        # neuronTime = (preFrameTime + (byte[i * 4 + 0] % 128) * 2**4 + (byte[i * 4 + 1] >> 4)) * self.actualTimestep / 1000
                        # preEventTime = (preFrameTime + (byte[i * 4 + 0] % 128) * 2**4 + (byte[i * 4 + 1] >> 4))
                        
                        
                        # Old version
                        #neuronID = ((byte[i * 4 + 1] % 4) * 2**16) + (byte[i * 4 + 2] * 2**8) + byte[i * 4 + 3]
                        #neuronTime = (preFrameTime + (byte[i * 4 + 0] % 128) * 2**6 + (byte[i * 4 + 1] >> 2)) * self.actualTimestep / 1000
                        
                        spikeEvent[eventCount] = neuronID
                        spikeTime[eventCount] = neuronTime
                        eventCount += 1
                        #print(neuronID, neuronTime)
                if len(self.spikes) == 0:
                    self.spikes = np.array([spikeTime[0:eventCount], spikeEvent[0:eventCount]])
                    self.Frames = np.array([tempFrameTime[0:frameCount], tempFrameNumber[0:frameCount]])
                else:
                    self.spikes = np.append(self.spikes, np.array([spikeTime[0:eventCount], spikeEvent[0:eventCount]]), axis=1)
                    self.Frames = np.append(self.Frames, np.array([tempFrameTime[0:frameCount], tempFrameNumber[0:frameCount]]), axis=1)
            
            index = index + 1
            DataFilename = foldername + "RecordingData" + str(index) + DataExtension
                    
        print("Total spike count is", self.spikeCount)