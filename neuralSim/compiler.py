 # -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:19:12 2019

@author: Jongkil Park
@Company: KIST
@Description:
"""

# %% import packages
import numpy as np
import neuralSim.number as number
import os as os
import time
from datetime import date
import neuralSim.parameters as param

# %% Input Event compiler (obsoluted)
class eventCompiler:
    
    def __init__(self, foldername, fname):
        '''
        The class is obsoluted.
        Please use FrameToPoissonCompiler class in inputSpikeGen.py module.
        '''
        print("(compiler.py) ERROR : Please use FrameToPoissonCompiler class in inputSpikeGen.py module.")
        
# %% Synaptic map compiler
class synMapCompiler:
    def generateSynMap(self, synTable, inputNeurons, linear=0, zeroWeightDelete=0):
        '''

        Parameters
        ----------
        synTable : TYPE
            DESCRIPTION.
        inputNeurons : TYPE
            DESCRIPTION.
        linear : TYPE, optional
            DESCRIPTION. The default is 1.
        zeroWeightDelete : TYPE, optional
            How to deal with zero weight.
            0 : set to minimum weight. It is good to save connection.
            1 : delete the event.
            The default is 0.
           
        Synapse table is encoded in 
        4 bytes : endian 
        4 bytes : The DRAM address to start write the synaptic routing table.
        4 bytes : The configuration mode of synaptic routing table (0: 256k or 1:1024k neurons)
        following bytes : Synaptic tables.
        

        Returns
        -------
        None.

        '''

        # external router
        #cRouter = 16 # 16 * 64k neurons
        
        endianCheck = np.array([1])
        
        if self.nChip == 1:
            allGroup = np.array([0, 1, 2, 3])
            # For external synaptic connections.
            ToGroupConPointer = [[], [], [], []]
            SRTMode = np.array([256], dtype=np.uint32)
            
        elif self.nChip == 4:
            allGroup = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            # For external synaptic connections.
            ToGroupConPointer = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
            SRTMode = np.array([1024], dtype=np.uint32)
            
        if len(inputNeurons) > 1: 
            print("Multiple source groups are detected.")

        
        print("The total number of neurons, which is defined in a synaptic table, is ", len(synTable.preIndexPointer))
        
        CountExtSynapses = 0
       
        for gIndex in range(len(inputNeurons)):
            
            StartTime = time.time()
            
            sourceNeurons = inputNeurons[gIndex]
            
            groupNumber = int(sourceNeurons[0] // 65536)

            print("Synapse table for group " + str(groupNumber) + " is generating.")
            
            # File IO open to write a SRT.
            fileWrite = open(self.foldername + self.fname + "_group_" + str(groupNumber) + ".dat", 'wb')

            # This is a DRAM address to start to write.   
            rawDRAMaddress = (sourceNeurons[0] % 65536) * (2**(np.log2(self.hashLen) + np.log2(self.FPGA_AXIChBW) - 3))
            
            if self.nChip == 1:
                router_offset = groupNumber << 29

            elif self.nChip == 4:
                router_offset = groupNumber << 27

            
            address = np.array([rawDRAMaddress + router_offset], dtype=np.uint32)
                        

            # First 12 bytes for a header.
            fileWrite.write(bytearray(endianCheck))
            fileWrite.write(bytearray(address))
            fileWrite.write(bytearray(SRTMode))
            
            # Condition check
            # Find min and max of input neuron numbers.
            min_input_neuron_num = min(sourceNeurons)
            max_input_neuron_num = max(sourceNeurons)
            
            # recreate input neuron index.
            sourceNeurons = np.linspace(min_input_neuron_num, max_input_neuron_num, num=max_input_neuron_num - min_input_neuron_num +1, dtype=np.int32)
            
            # Predefine space for data bytes.
            tempData = np.zeros(len(sourceNeurons) * self.hashLen * 4, dtype=np.uint32)
            
            genCount = 0
            # This is an iterative operation for presynaptic neurons selected in indexRange.
            for index in sourceNeurons:
                
                # Postsynaptic connection count
                # Check total postsynaptic counts to avoid hash overflow
                synapse_index = synTable.preIndexPointer[index]
                
                if len(synapse_index) == 0:
                    cCount = 0
                else:
                    
                    cCount = len(synTable.synapses[synapse_index])
                    
                
                if cCount > self.maxSynCount:
                    print("Error : Hash overflow. Need to add a function to generate another hash table", "Input neuron number : ", index)
                elif cCount == 0:
                    tempData[genCount*self.hashLen*4] = 2**31 + 2**21
                    genCount = genCount + 1
                else:
                    # List up destination of postsynaptic neurons and weight
                    destination = np.array(synTable.synapses[synapse_index].T[1], dtype=np.uint32)
                    weight = np.round(synTable.synapses[synapse_index].T[2], 7)
                    train = np.array(synTable.synapses[synapse_index].T[4], dtype=np.uint32)
                    
                    #####################################################################
                    # need to check whether a source neuron has external destination
                    groupDestination = destination // 65536
                    
                    extDestIndex = np.where(groupDestination != groupNumber)[0]

                    extDestination = []

                    if len(extDestIndex) > 0:
                        for checkGroup in np.delete(allGroup, groupNumber):
                            if len(np.where(groupDestination == checkGroup)[0]) > 0:
                                extDestination.extend([[checkGroup, len(ToGroupConPointer[checkGroup])]])
                                ToGroupConPointer[checkGroup].append(np.array(synapse_index[np.where(groupDestination == checkGroup)[0]]))
                                
                                CountExtSynapses += len(np.where(groupDestination == checkGroup)[0])
                   
                    
                    # delete synapses connected to external groups
                    destination = np.delete(destination, extDestIndex)
                    weight = np.delete(weight, extDestIndex)
                    train = np.delete(train, extDestIndex)
                    
                    # convert synapses into a slot
                    datum = self.convertSynToSlot(destination, weight, train, extDestination, linear, zeroWeightDelete)

                    tempData[genCount * self.hashLen * 4 : (genCount + 1) * self.hashLen * 4] = datum

                    genCount = genCount + 1

            print("Start to write synaptic routing table")
            fileWrite.write(bytearray(tempData))
            
            # Dummy slots for next 1024 neurons
            tempDummy = np.zeros(self.hashLen * 4 * 1024, dtype=np.uint32)
            tempDummy[np.linspace(0, (128*4*1023), num=1024, dtype=np.uint32)] = 2**31 + 1*2**21
            fileWrite.write(bytearray(tempDummy))
            
            fileWrite.close()
            print("Memory compiler generated a synaptic routing table. (Elapsed ", round(time.time()-StartTime, 3), "seconds.)\n")

        
        # generate extenral synapse tables.
        if self.nChip == 4 and self.hashLen == 128:
            print("Warning: external synapse connection will be ignored.")
            print("Total ", CountExtSynapses, "external synapses will be ignored.")
        else:
            # ext mapping table
            for extGIndex in range(len(allGroup)):
    
                nExtSynapse = len(ToGroupConPointer[extGIndex])

                if nExtSynapse > 65536:
                    print("(compiler.py) Error: too many external slots are defined.")
                    
                elif nExtSynapse > 0:
                    sourceNeurons = np.linspace(0, nExtSynapse - 1, num=nExtSynapse, dtype=np.int32)
    
                    fileWrite = open(self.foldername + self.fname + "_group_" + str(extGIndex) + "_ext" + ".dat", 'wb')
            
    
                    # This is a DRAM address to start to write.              
                    rawDRAMaddress = sourceNeurons[0] * (2**(np.log2(self.hashLen) + np.log2(self.FPGA_AXIChBW) - 3))
                    if self.nChip == 1:
                        router_offset = extGIndex << 29
                        ext_offset = 0x1000_0000
                    elif self.nChip == 4:
                        router_offset = extGIndex << 27
                        ext_offset = 0x0400_0000
                       
                    address = np.array([rawDRAMaddress + router_offset + ext_offset], dtype=np.uint32)
                    
                    fileWrite.write(bytearray(endianCheck))
                    fileWrite.write(bytearray(address))
                    fileWrite.write(bytearray(SRTMode))
                    
                    # Predefine space for data bytes.
                    tempData = np.zeros(len(sourceNeurons) * self.hashLen * 4, dtype=np.uint32)
    
                    genCount = 0
                    # This is an iterative operation for presynaptic neurons selected in indexRange.
                    for index in sourceNeurons:
                        
                        # Postsynaptic connection count
                        # Check total postsynaptic counts to avoid hash overflow
                        synapse_index = ToGroupConPointer[extGIndex][index]
                        
                        if len(synapse_index) == 0:
                            cCount = 0
                        else:
                            cCount = len(synTable.synapses[synapse_index])
                            
                        if cCount > self.maxSynCount:
                            print("Error : Hash overflow. Need to add a function to generate another hash table", "Input neuron number : ", index)
                        elif cCount == 0:
                            tempData[genCount*self.hashLen*4] = 2**31 + 2**21
                            genCount = genCount + 1
                        else:
                            # List up destination of postsynaptic neurons and weight
                            destination = np.array(synTable.synapses[synapse_index].T[1], dtype=np.uint32)
                            weight = np.round(synTable.synapses[synapse_index].T[2], 7)
                            
                            
                            if len(synTable.synapses[synapse_index].T[4]) > 0:
                                train = np.zeros(len(synTable.synapses[synapse_index].T[4]))
                                print("Warning : external synaptic connection could not have trainable synapses.")
                            
                            # convert synapses into a slot
                            datum = self.convertSynToSlot(destination, weight, train, [], linear, zeroWeightDelete)
    
                            tempData[genCount * self.hashLen * 4 : (genCount + 1) * self.hashLen * 4] = datum
    
                            genCount = genCount + 1
    
                    print("Start to write synaptic routing table")
                    fileWrite.write(bytearray(tempData))
                    
                    # Dummy slots for next 1024 neurons
                    tempDummy = np.zeros(self.hashLen * 4 * 1024, dtype=np.uint32)
                    tempDummy[np.linspace(0, (128*4*1023), num=1024, dtype=np.uint32)] = 2**31 + 1*2**21
                    fileWrite.write(bytearray(tempDummy))
                    
                    fileWrite.close()
                    print("Memory compiler generated a synaptic routing table. (Elapsed ", round(time.time()-StartTime, 3), "seconds.)\n")



    def convertSynToSlot(self, destination, weight, train, extDestination=[], linear=0, zeroWeightDelete=0):
        cCount = len(destination)
        
        if (len(np.where(train == 1)[0]) > 0):
            trainable_exist = 1
        else:
            trainable_exist = 0
            
        if cCount > 0:

    
            # Here we need to convert weight information to compiled weight value.
            # find index of inhibitory synapse
            inh = np.array(np.where(weight < 0)[0])
            
            # If a weight is linear sampled value in [-127, 127]
            if linear == 1: 
                floatTemp = number.FloatSplitToInt(np.array([self.max_syn_weight], dtype=np.float), 3, 4)
                
                max_value = floatTemp[0] * 16 + floatTemp[1]
                
                sanity = np.where((weight > max_value) | (weight < -1 * max_value))[0]
                if len(sanity > 0):
                    print("Warning : A weight value out of range was found.")
                    np.clip(weight, -1 * max_value, max_value)
            # If a weight is a real value between [-248, 248]
            else :    
                sanity = np.where((weight > self.max_syn_weight) | (weight < -1 * self.max_syn_weight))[0]
                if len(sanity > 0):
                    print("Warning : A weight value out of range was found.")
                    np.clip(weight, -1 * self.max_syn_weight, self.max_syn_weight)
                
                # Convert the weight into floating point representation
                weight_positonal = np.array(["{:0.5f}".format(x) for x in weight], dtype=np.float)
                
                floatTemp = number.FloatSplitToInt(np.abs(np.round(weight_positonal, 5), dtype=np.float), 3, 4)
                
                weight = floatTemp[0] * 16 + floatTemp[1]
            
            weight[inh] = np.abs(weight[inh]) + 128
        
            # Values are converted to unsinged integer format.
            weight = np.array(weight, dtype=np.uint32)
                            
            # Sort the arrays by destination address.
            sortIndex = np.argsort(destination)
            destination = destination[sortIndex]
            weight = weight[sortIndex]
            
            zeroWeightIndex = np.where(weight == 0)[0]
            
            ## Zero weight event
            if zeroWeightDelete == 0:
                weight[zeroWeightIndex] = 1
            elif zeroWeightDelete == 1:
                weight = np.delete(weight, zeroWeightIndex)
                destination = np.delete(destination, zeroWeightIndex)
                train = np.delete(train, zeroWeightIndex)
                
                cCount -= len(zeroWeightIndex)
            
            ## DeltaOffset
            deltaOffset = np.append(np.array([0]), (np.append(destination, np.array([0])) - np.append(np.array([0]), destination))[1:len(destination)])
    
            for iteration in range(10):
                deltaOffsetOutIndex = np.where(deltaOffset > self.maxDeltaOffset)[0]
                if cCount + len(deltaOffsetOutIndex) > self.maxSynCount:
                    print("Error : too many dummy events are generated.")
    
                if len(deltaOffsetOutIndex) > 0:
                    # insert dummy events (zero events)
                    deltaOffset[deltaOffsetOutIndex] = deltaOffset[deltaOffsetOutIndex] - self.maxDeltaOffset
                    deltaOffset = np.insert(deltaOffset, deltaOffsetOutIndex, self.maxDeltaOffset)
                    weight = np.insert(weight, deltaOffsetOutIndex, 0)
                    train = np.insert(train, deltaOffsetOutIndex, 0)
                    
                    cCount += len(deltaOffsetOutIndex)
                    print("Dummy events are generated for offset calibration.")
                    #print("Offset error. Need to add a function to generate another header. (compiler.py)", index)
            
            # baseAddress is based on 64k neurons group
            baseAddress = destination[0] & 0xFFFF
        else: # no internal synapse
            baseAddress = 0
            weight = []
            deltaOffset = []
            train = []
            

        if len(extDestination) > 0:
            # pad bridge event
            weight = np.append(weight, np.zeros(1))
            deltaOffset = np.append(deltaOffset, np.zeros(1))
            train = np.append(train, np.zeros(1))
            cCount += 1

            # external events
            for i in range(len(extDestination)):
                weight = np.append(weight, [extDestination[i][1] & 0xFF])
                deltaOffset = np.append(deltaOffset, [((extDestination[i][1] >> 8) & 0xF) + (extDestination[i][0] & 0x7) * 16])
                train = np.append(train, [extDestination[i][0] >> 3])
                cCount += 1

        if cCount > self.maxSynCount:
            print("Error : too many events.")

        # zero padding for no events.
        if cCount < self.maxSynCount:
            weight = np.append(weight, np.zeros(self.maxSynCount - cCount))
            deltaOffset = np.append(deltaOffset, np.zeros(self.maxSynCount - cCount))
            train = np.append(train, np.zeros(self.maxSynCount - cCount))
        

        # Header [31] : 1
        # Header [30] : trainable synapse exist
        # Header [29] : 0 (2nd hash does not exist)
        # Header [28:16] : 0 (Reserved, 2nd hash slot number)
        # Header [15:00] : base address
        
        events = train * 2**15 + deltaOffset * 2**8 + weight
        events = np.reshape(events, (int(len(events)/2), 2))
        
        synapse = np.array(events.T[0] * 2**16 + events.T[1], dtype=np.uint32)
        
        hash_len_the_neuron = np.ceil((32 + cCount * 16) / self.FPGA_AXIChBW)
        
        header = np.array([1 * 2**31 + trainable_exist * 2**30 + hash_len_the_neuron * 2**21 + baseAddress], dtype=np.uint32)
        datum = np.append(header, synapse)

        return datum
        
    def set_max_syn_weight(self, value):
        floatTemp = number.FloatSplitToInt(np.array([value], dtype=np.float), 3, 4)
        
        if floatTemp[2][0] > 0:
            self.max_syn_weight = number.ToFloat(floatTemp[0], floatTemp[1], 4)
            print("Warning : maximum synaptic weight is set to ", self.max_syn_weight)
        else:
            self.max_syn_weight = value
        
        
        
    def __init__(self, fname, foldername=param.NeuPLUSSynFolder):    
        # FPGA internal variable
        self.FPGA_AXIChBW = 128
        self.synEventBWidth = 16 #16 bits
        
        # max synaptic weight 
        self.max_syn_weight = 248.0 # real value.
        
        
        self.HeaderCount = 2 #32 bits
        
        # compiler variables
        self.hashLen = 128
        self.maxSynCount = np.array(self.hashLen * (self.FPGA_AXIChBW / self.synEventBWidth) - self.HeaderCount, dtype=np.uint32)
        
        self.maxDeltaOffset = 127

        self.fname = fname
        self.foldername = foldername

        # the number of chip. (1 or 4)
        self.nChip = 1


# %% Experimental setup compiler
class expSetupCompiler:
    def setExperimentConf(self, *args):
        
        if self.mask_changed > 0 or self.FPGA_inst_count > 0 or self.LE_inst_count > 0 or self.arrayCount > 0:
            print("(compiler.py) WARNING : experiment set up should be defined before learning core and neuron array definition.")
            

        # mode selection
        if(args[0] == "EVENT_SAVE_MODE"):
            self.EventSaveMode = args[1]
        elif(args[0] == "INTERNAL_ROUTING_MODE"):
            self.InternalRoutingMode = args[1]
        elif(args[0] == "LOOPBACK_MODE"):
            self.LoopbackMode = args[1]
        elif(args[0] == "REALTIME_INPUT_MODE"):
            self.RealtimeInputMode = args[1]
        elif(args[0] == "DVS_INPUT_MODE"):
            self.DVS_inputMode = args[1]
        elif(args[0] == "TIME_ACCEL_MODE"):
            self.TimeAccelMode = args[1]
        elif(args[0] == "AUX_PATH"):
            self.AuxPath = args[1]
            
        # variables
        elif(args[0] == "TIME_ACCEL_TIMESTEP"):
            self.timeAccelTimestep = args[1]
            
            # if there is predefined time constant and refractory period..
            if self.arrayCount != 0:
                for index in range(self.arrayCount):
                    self.arrayNum = []
                    self.timeConstant = []
                    self.refractory = []
                    
                    ActualTimeconst, ActualRefractory, propUnitstep = self.checkAccurateTimeConst(self.timeConstant[index], self.refractory[index])
                    
                    print("Info : array number ", self.arrayNum[index])
                    print("Info : timeconstant ", self.timeConstant[index], " ms will be effectively applied as ", ActualTimeconst, " ms.")
                    print("Info : refractory ", self.refractory[index], " ms will be effectively applied as ", ActualRefractory, " ms.")
                    print("Info : For better accurate temporal precision, TimeAcceleration time step and resolution should be updated.")
                    print("Info : To have exact time constant with the current resolution, timestep should be set to ", propUnitstep)
            
            
        elif(args[0] == "TIME_ACCEL_RESOLUTION"):
            self.timeAccelResolution = args[1]

            if self.arrayCount != 0:
                for index in range(self.arrayCount):
                    self.arrayNum = []
                    self.timeConstant = []
                    self.refractory = []
                    
                    ActualTimeconst, ActualRefractory, propUnitstep = self.checkAccurateTimeConst(self.timeConstant[index], self.refractory[index])
                    
                    print("Info : array number ", self.arrayNum[index])
                    print("Info : timeconstant ", self.timeConstant[index], " ms will be effectively applied as ", ActualTimeconst, " ms.")
                    print("Info : refractory ", self.refractory[index], " ms will be effectively applied as ", ActualRefractory, " ms.")
                    print("Info : For better accurate temporal precision, TimeAcceleration time step and resolution should be updated.")
                    print("Info : To have exact time constant with the current resolution, timestep should be set to ", propUnitstep)


        elif(args[0] == "EXP_TIME"):
            self.ExpTime = args[1]
            
        elif(args[0] == "FPGA_TIMESTEP"):
            self.FPGATimestep = args[1]
            
            
        # Files
        elif(args[0] == "INPUT_SPIKES_FILE"):
            self.InputSpikesFilename = args[1]
        elif(args[0] == "SYN_TABLE_FILE_PREFIX"):
            self.synTableFilePrefix = args[1]
        elif(args[0] == "SYN_TABLE_READ_FILE"):
            self.synTableReadFile = args[1]
        elif(args[0] == "SYN_TABLE_READ_START"):
            self.synTableReadStart = args[1]
        elif(args[0] == "SYN_TABLE_READ_COUNT"):
            self.synTableReadCount = args[1]
        else:
            print("(compiler.py) ERROR : " + args[0] + "command is not defined.")
            
        print("Setup experiment configuration.")
        
    def setNeuronCoreConf(self, arrayNum, timeConstant, refractory, threshold, synapticGain, stochasticity):
        """
        

        Parameters
        ----------
        arrayNum : TYPE
            DESCRIPTION.
        timeConstant : TYPE
            DESCRIPTION.
        refractory : TYPE
            DESCRIPTION.
        threshold : TYPE
            DESCRIPTION.
        synapticGain : TYPE
            DESCRIPTION.
        stochasticity : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # all input should be in as numpy array
        
        #uint[] arrayNum
        #double[] timeConstant (in ms)
        #float[] refractory (in ms)
        #uint[] resetVoltage = 0
        #uint[] threshold
        #uint[] synapticGain
        #uint[] arrayStochasticity
        
        # max range
        array_max = 1024
        timeConstant_max = 20000
        refractory_max = 100
        threshold_max = 65504 # 1111_1111111111

        synapticGain_max = 0
        stochasticity_max = 0
        
        
        arrayNum = np.array(arrayNum, dtype=np.uint32)
        timeConstant = np.array(timeConstant, dtype=np.double)
        refractory = np.array(refractory, dtype=np.single)
        threshold = np.array(threshold, dtype=np.float)
        synapticGain = np.array(synapticGain, dtype=np.float)
        stochasticity = np.array(stochasticity, dtype=np.float)
        
        # Validity check
        array_false = np.where(arrayNum > array_max)[0]
        if len(array_false) > 0:
           print("Array number ", arrayNum[array_false], "is (are) out of range.")
        
        timeConstant_false = np.where(timeConstant > timeConstant_max )[0]
        if len(timeConstant_false) > 0:
            print("Timeconstant ", timeConstant[timeConstant_false], "is (are) out of range.")

        refractory_false = np.where(refractory > refractory_max )[0]
        if len(refractory_false) > 0:
            print("Refractory ", refractory[refractory_false], "is (are) out of range.")
        
        threshold_false = np.where(threshold > threshold_max )[0]
        if len(threshold_false) > 0:
            print("Threshold ", threshold[threshold_false], "is (are) out of range.")
        
        synapticGain_false = np.where(synapticGain > synapticGain_max )[0]
        if len(synapticGain_false) > 0:
            print("Synaptic gain ", synapticGain[synapticGain_false], "is (are) out of range.")
        
        stochasticity_false = np.where(stochasticity > stochasticity_max )[0]
        if len(stochasticity_false) > 0:
            print("Stochasticity ", stochasticity[stochasticity_false], "is (are) out of range.")

        print("Neuron array ", arrayNum, "configuration file will be generated.")
        
        # Generate information for time constant calculation and refractory period
        if self.TimeAccelMode == True:
            ActualTimeconst, ActualRefractory, propUnitstep = self.checkAccurateTimeConst(timeConstant, refractory)
            
            print("Info : timeconstant ", timeConstant, " ms will be effectively applied as ", ActualTimeconst, " ms.")
            print("Info : refractory ", refractory, " ms will be effectively applied as ", ActualRefractory, " ms.")
            print("Info : For better accurate temporal precision, TimeAcceleration time step and resolution should be updated.")
            print("Info : To have exact time constant with the current resolution, timestep should be set to ", propUnitstep)                           
                
                
        # need to find existing array num
        intersectItem = np.intersect1d(self.arrayNum, arrayNum)
        
        if len(intersectItem) > 0:
            for index in range(len(intersectItem)):
                overlapIndex = np.where(self.arrayNum == intersectItem[index])[0]
                
                print(intersectItem[index], " is already defined. It will be overwitten.")
                
                self.arrayNum = np.delete(self.arrayNum, overlapIndex)
                
                self.timeConstant = np.delete(self.timeConstant, overlapIndex)
                self.refractory = np.delete(self.refractory, overlapIndex)
                self.threshold = np.delete(self.threshold, overlapIndex)
                self.synapticGain = np.delete(self.synapticGain, overlapIndex)
                self.stochasticity = np.delete(self.stochasticity, overlapIndex)
        
        
        # real value split to exponent and mantissa
        threshold_exponent, threshold_mantissa, threshold_error = number.FloatSplitToInt(threshold, 6, 10)
        synapticGain_exponent, synapticGain_mantissa, synapticGain_error = number.FloatSplitToInt(synapticGain, 3, 4)
        stochasticity_exponent, stochasticity_mantissa, stochasticity_error = number.FloatSplitToInt(stochasticity, 3, 4)
        
        # threshold value error check
        th_error_index = np.where(threshold_error > 0)[0]
        if len(th_error_index) > 0:
            for index in range(len(th_error_index)):
                print(threshold[th_error_index[index]], " will be set to ", number.ToFloat(threshold_exponent[th_error_index[index]], threshold_mantissa[th_error_index[index]], 10), ".")
        
        # synaptic gain value error check
        synG_error_index = np.where(synapticGain_error > 0)[0]
        if len(synG_error_index) > 0:
            for index in range(len(synG_error_index)):
                print(synapticGain[synG_error_index[index]], " will be set to ", number.ToFloat(synapticGain_exponent[synG_error_index[index]], synapticGain_mantissa[synG_error_index[index]], 4), ".")
            
        # stocasticity value error check
        stochasticity_error_index = np.where(stochasticity_error > 0)[0]
        if len(stochasticity_error_index) > 0:
            for index in range(len(stochasticity_error_index)):
                print(stochasticity[stochasticity_error_index[index]], " will be set to ", number.ToFloat(stochasticity_exponent[stochasticity_error_index[index]], stochasticity_mantissa[stochasticity_error_index[index]], 4), ".")

        
        # real value to integer value.
        threshold_int = threshold_exponent * 1024 + threshold_mantissa
        synapticGain_int = synapticGain_exponent * 16 + synapticGain_mantissa
        stochasticity_int = stochasticity_exponent * 16 + stochasticity_mantissa
        
        
        self.arrayNum = np.append(self.arrayNum, arrayNum)
        
        self.timeConstant = np.append(self.timeConstant, timeConstant)
        self.refractory = np.append(self.refractory, refractory)
        
        self.threshold = np.append(self.threshold, threshold_int)
        
        self.synapticGain = np.append(self.synapticGain, synapticGain_int)
        self.stochasticity = np.append(self.stochasticity, stochasticity_int)
        
        ## configured array count
        self.arrayCount = self.arrayCount - len(intersectItem) + len(arrayNum)
        
        

        ## router time add event enable and disable.
        ## we want to enable time add event only when the core is defined.
        targetPort = arrayNum // 64
        targetGroup = (arrayNum % 64) >> 3
        
        
        self.time_add_event_enable[targetPort] = (int)(self.time_add_event_enable[targetPort]) | (2**targetGroup)
        
            
        
    def setLearningEngine(self, command, core, *args, verbose="True"):
        '''
        # What kind of value could be set for an experiment?
        # It would be better to saved in a byte instruction..
        
        # core : a number of router, which is routing events for a neuron core in 64k neurons. (0 ~ 15)
        # memArray : 0~8 ( spike trace memory ) 9 ( registers in a learning engine)
        # 1M NeuPLUS version only support 4 spike trace memory due to FPGA resource.
        '''
        
        prescaler = 15
        sysClockCycle = 6.66
        
        if (self.TimeAccelMode == True):
            unitTimestep = self.timeAccelTimestep * 1000 * 1000
        else:
            unitTimestep = prescaler * sysClockCycle
        
        
        ###### command interprete #######
        if command == "LC_TIMECONSTANT":
            # args[0] : time constant (in ms)
            
            if len(args) != 1:
                print("(compiler.py) ERROR : The number of argument does not match. The LC_TIMECONSTANT command requires 2 arguments. (core, time constant)")
            
            timeconstant = args[0]
            
            mantissabit = 4
            
            timeconstantConv =  (timeconstant/(2**mantissabit) * 1000 * 1000) / unitTimestep
            
            timeconstant_lower = timeconstantConv % 256
            timeconstant_upper = timeconstantConv / 256
            
            self.LEcommandToUint32(core, 9, 0, timeconstant_lower)
            self.LEcommandToUint32(core, 9, 1, timeconstant_upper)
            
            self.LE_core_timeconstant.append([core, timeconstant])
        
        elif command == "ADDR_LOOKUP":
            # args[0] : memory (0~3)
            # args[1] : value
            
            if len(args) != 2:
                print("(compiler.py) ERROR : The number of argument does not match. The ADDR_LOOKUP command requires 3 arguments. (core, memory, value)")
                return False

            if args[0] < 4:
                address = args[0] + 7
            else:
                print("(compiler.py) ERROR : address range is out of range.")
                return False

            if args[1] > 63:
                print("Warning")
                
            target_array = args[1]
            
            self.LEcommandToUint32(core, 9, address, target_array)


        elif command == "LEARNING_RULE":
            if len(args) != 5:
                print("(compiler.py) ERROR : The number of argument does not match. The LEARNING_RULE command requires 5 arguments. (core, stochastic_rounding, STDP type, modulator place, modulation )")
                return False
            
            
            # learning rule
            # array 9, address 15
            # [0] : modulation
            # [1] : place to apply modulator (0 : pre, 1: post) // not yet implemented.
            # [2] : STDP type (0 : additive, 1 : multiplicative)
            # [3] : stochastic rounding
            # [4] : constant modulation

            
            int_value = args[4] * 2**4 + args[3] * 2**3 + args[2] * 2**2 + args[1] * 2**1 + args[0]
            
            print("Learning rule is defined as ", int_value)
            
            self.LEcommandToUint32(core, 9, 15, int_value)

        elif command == "MAX_SYN_WEIGHT": # mode 0 address 16
            if len(args) != 1:
                print("(compiler.py) ERROR : The number of argument does not match. The MAX_SYN_WEIGHT command requires 2 arguments. (core, value)")
                return False

            # convert a real value to binary
            exponent, mantissa, error = number.FloatSplitToInt([args[0]], 3, 4)
            int_value = exponent * 16 + mantissa
            
            self.LEcommandToUint32(core, 9, 16, int_value)
            
            if error > 0:
                print("Floating point conversion error is ", error)
            
            
        elif command == "MAX_SYN_UPDATE": # mode 0  address 17
            if len(args) != 1:
                print("(compiler.py) ERROR : The number of arguments does not match. The MAX_SYN_UPDATE command requires 2 arguments. (core, value)")
                
                return False

            # convert a real value to binary
            exponent, mantissa, error = number.FloatSplitToInt([args[0]], 3, 4)
            int_value = exponent * 16 + mantissa
            
            
            self.LEcommandToUint32(core, 9, 17, int_value)
            
            if error > 0:
                print("Floating point conversion error is ", error)

        
        elif command == "MOD_AREA_DEFINE": # mode 0 address 18 and 19
            
            
            '''
            There are 1,024 modulators.
            # input
            - number_of_neuron : It is the number of neurons which are affected by a modulator.
                                 It should be 1, 2, 4, 8, 16, 32, and 64.
                                 This will be used to calculate offset of input address in the router.
                                 number_of_neuron : offset
                                        1         :   0
                                        2         :   1
                                        4         :   2
                                        8         :   3
                                        16        :   4 
                                        32        :   5
                                        64        :   6
            - start_neuron_num : It is a target start neuron number where the modulator memory module supports to.
                                 Actual start neuron number could be changed based on number_of_neuron.
                                 The number could be 0 ~ 1M
            '''
            ## Sanity check
            if len(args) != 2:
                print("(compiler.py) ERROR : The number of arguments does not match. The MOD_AREA_DEFINE command requires 3 arguments. (core, number_of_neuron to be affected by a modulator, start_neuron_num)")
                return False
    
            num_of_neurons = args[0]
            start_neuron_num = args[1]
            
            if num_of_neurons not in [1, 2, 4, 8, 16, 32, 64]:
                print("(compiler.py) ERROR : The number of neurons to be affected by a modulator should be 1, 2, 4, 8, 16, 32 or 64.")
                return False
            
            if start_neuron_num // (64*1024) != core:
                print("(compiler.py) ERROR : Target start address is not in the learning core in the group. Need to check.")
                return False
            
            ############################
            offset = int(np.log2(num_of_neurons))
            
            neuron_num_in_router = start_neuron_num % (64*1024)
            
            site_num = neuron_num_in_router // (2**(10+offset))

            start = core*(64*1024) + (site_num) * (2**(10+offset))
            end = start + 1024*num_of_neurons

            print("From neuron ", start, "to neuron ", end)
            print("Each modulator would affect ", num_of_neurons, " neurons.")
            
            if start_neuron_num != start :
                print("(compiler.py) WARNING : the start address of the target site is modifed because of offset. Please check it, if needed.")
                
                
            # modulation offset
            self.LEcommandToUint32(core, 9, 18, offset)
            
            # modulation group
            self.LEcommandToUint32(core, 9, 19, site_num)
            
            self.LEModArea.append([core, num_of_neurons, site_num])
            
            
            return num_of_neurons, site_num
            

        # %% Setting memory in each learning core
        
        # timing offset
        elif command == "LC_MEM_TIMECONSTANT":
            
            if len(args) != 2:
                print("(compiler.py) ERROR : The number of arguments does not match. The LC_MEM_TIMECONSTANT command requires 3 arguments. (core, memArray, target_timeconstant)")
                return False
            
            #need to check memory core timeConstant.
            exist = np.where(np.array(self.LE_core_timeconstant).T[0] == core)[0]
            
            if len(exist) == 0:
                print("Error : base timeconstant is not defined.")
                return False
            elif len(exist) > 1:
                print("Error : base timeconstant is defined multiple times.")
                return False
            
            memArray = args[0]
            target_timeconstant = args[1]
            
            # find base timeconstant
            base_timeconstant = np.array(self.LE_core_timeconstant).T[1][exist]
            
            mantissaBit = 4
            
            base_timeconstant_conv = (base_timeconstant/(2**mantissaBit) * 1000 * 1000) / unitTimestep
            target_timeconstant_conv = (target_timeconstant/(2**mantissaBit) * 1000 * 1000) / unitTimestep
            
            # calculate time offset
            
            offset = np.zeros(8)
            for offsetIndex in range(8):
                offset[offsetIndex] = int((base_timeconstant_conv * (offsetIndex * 16 + 15)  / target_timeconstant_conv) - (offsetIndex * 16 + 15))

            # offset values need to have same sign 
            offsetSign = np.sign(offset)
            
            for chk_i in range(7):
                if offsetSign[chk_i] != offsetSign[chk_i + 1] :
                    
                    print("(compiler.py) ERROR : Offset values need to have same sign.")
                    
                    return False

            mode = 2 * 262144
                        
            if sum(offset) == 0:
                for address in range(8):
                    int_value = mode + 0
                    self.LEcommandToUint32(core, memArray, address, int_value)
            else:
                for address in range(8):
                    if offset[address] < 0:
                        int_value = mode + abs(offset[address]) + 128
                    else:
                        int_value = mode + abs(offset[address])
                    
                    self.LEcommandToUint32(core, memArray, address, int_value)

            
        # update trace
        elif command == "TRACE_UPDATE_AMOUNT": #
            
            if len(args) != 2:
                print("(compiler.py) ERROR : The number of arguments does not match. The TRACE_UPDATE_AMOUNT command requires 3 arguments. (core, memArray, value)")
                return False
            
            memArray = args[0]
            
            exponent, mantissa, error = number.FloatSplitToInt([args[1]], 3, 4)
            
            #(set a configuration register)
            mode = 2* 262144
            
            int_value = mode + exponent * 16 + mantissa 
        
            self.LEcommandToUint32(core, memArray, 8, int_value)
            
            if error > 0:
                print("Floating point conversion error is ", error)
        
        elif command == "POST_TRACE_SCALE":
            if len(args) != 2:
                print("(compiler.py) ERROR : The number of arguments does not match. The POST_TRACE_SCALE command requires 3 arguments. (core, memArray, value)")
                return False
            
            memArray = args[0]
            
            if args[1] < 0:
                sign = 128
            else:
                sign = 0
                
            exponent, mantissa, error = number.FloatSplitToInt([abs(args[1])], 3, 4)
            
            #(set a configuration register)
            mode = 2 * 262144
            
            int_value = mode + exponent * 16 + mantissa + sign
            
            self.LEcommandToUint32(core, memArray, 9, int_value)
            
            if error > 0:
                print("Post scale factor conversion error is ", error)
                
        else:
            print("ERROR : the command does not exist in command list.")
            
# %%        
    def setFPGAConfiguration(self, command, *args):
        
        if command == "CHIP_CLOCK":
            if args[0] == 50:
                self.chip_clock = 0
            elif args[0] == 100:
                self.chip_clock = 1
            elif args[0] == 150:
                self.chip_clock = 2
            elif args[0] == 200:
                self.chip_clock = 3
            else:
                print("ERROR")
                self.chip_clock = 0
            
            self.FPGA_inst = np.insert(self.FPGA_inst, self.FPGA_inst_count, 258)
            self.FPGA_inst = np.insert(self.FPGA_inst, self.FPGA_inst_count+1, self.chip_clock)
            
            self.FPGA_inst_count = self.FPGA_inst_count + 2
        ###########################################
        ## core array output masking
        ## args[0] : core #
        elif command == "ARRAY_OUTPUT_ON":
            # args[0] : core
            
            if len(args) > 1:
                raise Exception("Error : the length of the arguments are wrong.")
                
            subIndex = args[0] // 16
            destCore = args[0] % 16
            
            self.mask_changed = 1
            self.output_mask[subIndex] = (int)(self.output_mask[subIndex]) | (2**destCore)
            
        elif command == "ARRAY_OUTPUT_OFF":
            if len(args) > 1:
                raise Exception("Error : the length of the arguments are wrong.")

            subIndex = args[0] // 16
            destCore = args[0] % 16
            
            self.mask_changed = 1
            self.output_mask[subIndex] = (int)(self.output_mask[subIndex]) & ((2**16-1) ^ 2**destCore)

        elif command == "ARRAY_OUTPUT_ALL_ON":
            self.mask_changed = 1
            for i in range(64):
                self.output_mask[i] = 65535
                
        elif command == "ARRAY_OUTPUT_ALL_OFF":
            self.mask_changed = 1
            for i in range(64):
                self.output_mask[i] = 0
        
        elif command == "ROUTER_SYN_OUTPUT_ENABLE":
            # args[0] : core number
            subIndex = args[0] // 16
            destCore = args[0] % 16
            
            self.router_syn_enable[subIndex] = (int)(self.router_syn_enable[subIndex]) | (2**destCore)
            
        elif command == "ROUTER_SYN_OUTPUT_DISABLE":
            # args[0] : core number
            subIndex = args[0] // 16
            destCore = args[0] % 16
            
            self.router_syn_enable[subIndex] = (int)(self.router_syn_enable[subIndex]) & ((2**16-1) ^ 2**destCore)
        elif command == "MAX_ITER_COUNT":
            groupNum = args[0]
            maxCount = args[1]
            
            if maxCount < 1:
                raise Exception("Error : The number of maximum iteration count should be grater and equal than 1.")
                
            self.max_iter_count_group[groupNum] = maxCount - 1;
            
            
        else:
            raise Exception("ERROR : the command does not exist in the command list.")
                
    def LEcommandToUint32(self, core, memArray, address, int_value):
        commandOffset = 3489660928
        
        inst = commandOffset + core * 16777216 + memArray * 1048576 + address * 256 + int(int_value)
            
        self.LE_inst = np.insert(self.LE_inst, self.LE_inst_count, inst)
        self.LE_inst_count = self.LE_inst_count + 1
        
        
    def genFPGAConfFile(self):
        fileWrite = open(self.foldername + self.filename, 'wb')
        
        # Write the number of instruction sets.
        if self.mask_changed > 0:
            mask_count = 64
        else:
            mask_count = 0
            
        router_enable_mask_count = 64
        time_add_event_enable_count = 16
        max_iter_count_setting_count = 16
            
        nByte = fileWrite.write(bytearray(np.array([mask_count], dtype=np.uint32)))
        if nByte != 4:
            raise Exception("(compiler.py) ERROR : the number of bytes is not correct. (651)")
            
        totalFPGAInstCount = self.FPGA_inst_count + router_enable_mask_count + time_add_event_enable_count + max_iter_count_setting_count
        
        nByte = fileWrite.write(bytearray(np.array([totalFPGAInstCount], dtype=np.uint32)))
        if nByte != 4:
            raise Exception("(compiler.py) ERROR : the number of bytes is not correct. (655)")
        nByte = fileWrite.write(bytearray(np.array([self.LE_inst_count], dtype=np.uint32)))
        if nByte != 4:
            raise Exception("(compiler.py) ERROR : the number of bytes is not correct. (658)")

        
        if mask_count > 0:            
            for index in range(64):
                nByte = fileWrite.write(bytearray(np.array([self.output_mask[index]], dtype=np.uint32)))
                if nByte != 4:
                    raise Exception("(compiler.py) ERROR : the number of bytes is not correct.")
        
        if self.FPGA_inst_count > 0:
            for index in range(self.FPGA_inst_count):
                nByte = fileWrite.write(bytearray(np.array([self.FPGA_inst[index]], dtype=np.uint32)))
                if nByte != 4:
                    raise Exception("(compiler.py) ERROR : the number of bytes is not correct.")

        if router_enable_mask_count > 0:
            for index in range(router_enable_mask_count):
                commandOffset = 16777216
                commandValue = commandOffset + index* 2**16 + self.router_syn_enable[index]
                
                nByte = fileWrite.write(bytearray(np.array([commandValue], dtype=np.uint32)))
                if nByte != 4:
                    raise Exception("(compiler.py) ERROR : the number of bytes is not correct.")
        
        if time_add_event_enable_count > 0:
            for index in range(time_add_event_enable_count):
                commandOffset = 20971520
                commandValue = commandOffset + index* 2**18 + self.time_add_event_enable[index]
                
                nByte = fileWrite.write(bytearray(np.array([commandValue], dtype=np.uint32)))
                if nByte != 4:
                    raise Exception("(compiler.py) ERROR : the number of bytes is not correct.")
        
        if max_iter_count_setting_count > 0:
            for index in range(max_iter_count_setting_count):
                commandOffset = 25165824
                commandValue = commandOffset + index* 2**18 + self.max_iter_count_group[index]
                
                nByte = fileWrite.write(bytearray(np.array([commandValue], dtype=np.uint32)))
                if nByte != 4:
                    raise Exception("(compiler.py) ERROR : the number of bytes is not correct.")
        
        
        if self.LE_inst_count > 0:
            for index in range(self.LE_inst_count):
                nByte = fileWrite.write(bytearray(np.array([self.LE_inst[index]], dtype=np.uint32)))
                if nByte != 4:
                    raise Exception("(compiler.py) ERROR : the number of bytes is not correct.")
                
        print("(compiler.py) INFO : FPGA configuration file is generated.")                
        fileWrite.close()
        
    def genNeuronConfFile(self):
        fileWrite = open(self.foldername + self.neuronFilename, 'wb')
        
        #uint[] arrayNum
        #double[] timeConstant (in ms)
        #float[] refractory (in ms)
        #uint[] resetVoltage = 0
        #uint[] threshold
        #uint[] synapticGain
        #uint[] arrayStochasticity
        
        resetVoltage = np.zeros(self.arrayCount, dtype=np.uint32)
        
        fileWrite.write(bytearray(np.array([self.arrayCount], dtype=np.uint32)))
        
        for index in range(self.arrayCount):
            nByte = fileWrite.write(bytearray(np.array([self.arrayNum[index]], dtype=np.uint32)))
            if nByte != 4:
                print("(compiler.py) ERROR : the number of bytes is not correct.")
            nByte = fileWrite.write(bytearray(np.array([self.timeConstant[index]], dtype=np.double)))
            if nByte != 8:
                print("(compiler.py) ERROR : the number of bytes is not correct.")
            nByte = fileWrite.write(bytearray(np.array([self.refractory[index]], dtype=np.single)))
            if nByte != 4:
                print("(compiler.py) ERROR : the number of bytes is not correct.")
            nByte = fileWrite.write(bytearray(np.array([resetVoltage[index]], dtype=np.uint32)))
            if nByte != 4:
                print("(compiler.py) ERROR : the number of bytes is not correct.")
            nByte = fileWrite.write(bytearray(np.array([self.threshold[index]], dtype=np.uint32)))
            if nByte != 4:
                print("(compiler.py) ERROR : the number of bytes is not correct.")
            nByte = fileWrite.write(bytearray(np.array([self.synapticGain[index]], dtype=np.uint32)))
            if nByte != 4:
                print("(compiler.py) ERROR : the number of bytes is not correct.")
            nByte = fileWrite.write(bytearray(np.array([self.stochasticity[index]], dtype=np.uint32)))
            if nByte != 4:
                print("(compiler.py) ERROR : the number of bytes is not correct.")
            
        fileWrite.close()
        
        print("(compiler.py) INFO : Total ", self.arrayCount, "neuron array will be configured.")
    
    def genExpConfFile(self, filename):
        
        if self.mask_changed > 0 or self.FPGA_inst_count > 0 or self.LE_inst_count > 0:
            self.genFPGAConfFile()
        
        if self.arrayCount > 0:
            self.genNeuronConfFile()
            
        fileWrite = open(self.foldername + filename, 'w')
        
        fileWrite.writelines("////////////////////////////////////////////////////////////////// \n")
        fileWrite.writelines("/// This file is configuring an experiment runset.\n")
        fileWrite.writelines("/// \n")
        fileWrite.writelines("/// " + date.today().strftime("%Y. %m. %d") + "\n")
        fileWrite.writelines("\n")
        fileWrite.writelines("/// experiment mode setup    \n")
        fileWrite.writelines("\n")
        
        fileWrite.writelines("EventSave=" + str(self.EventSaveMode) + "\n")
        fileWrite.writelines("InternalRouting=" + str(self.InternalRoutingMode) + "\n")
        fileWrite.writelines("LoopbackMode=" + str(self.LoopbackMode) + "\n")
        fileWrite.writelines("RealtimeInput=" + str(self.RealtimeInputMode) + "\n")
        fileWrite.writelines("DVS_input=" + str(self.DVS_inputMode) + "\n")
        fileWrite.writelines("TimeAccelMode=" + str(self.TimeAccelMode) + "\n")
        fileWrite.writelines("TimeAccelResolution=" + str(self.timeAccelResolution) + "\n")
        fileWrite.writelines("AUX_Path=" + str(self.AuxPath) + "\n")
        
        fileWrite.writelines("\n")
        fileWrite.writelines("ExpTime=" + str(self.ExpTime) + "\n")
        fileWrite.writelines("\n")
        fileWrite.writelines("\n")
        fileWrite.writelines("\n")
        
        fileWrite.writelines("////////////////////////////////////////////////////////////////\n")
        fileWrite.writelines("/// Neuron array configuration file\n")
        fileWrite.writelines("/// arraynumber, time constant, threshold, etc..\n")
        fileWrite.writelines("\n")

        fileWrite.writelines("NeuronConfigurationFile=" + self.neuronFilename + "\n");

        fileWrite.writelines("\n")
        
        fileWrite.writelines("////////////////////////////////////////////////////////////////\n")
        fileWrite.writelines("/// Input spike pattern\n")
        fileWrite.writelines("/// Spike pattern could be ...\n")
        fileWrite.writelines("/// 1. Poisson frame\n")
        fileWrite.writelines("/// 2. Temporal spike pattern encoded in byte events.\n")
        fileWrite.writelines("/// 3. Realtime input\n")
        fileWrite.writelines("\n")        
        
        if self.InputSpikesFilename != "":
            fileWrite.writelines("InputSpikesFile=" + self.InputSpikesFilename + "\n")
        fileWrite.writelines("\n")

        fileWrite.writelines("///////////////////////////////////////////////////////////////\n")
        fileWrite.writelines("/// synapse routing table file \n")
        fileWrite.writelines("/// If internalRouting mode is on, synapse routing table file should be defined.\n")
        fileWrite.writelines("/// \n")
        fileWrite.writelines("\n")
        
        if self.synTableFilePrefix != "":
            fileWrite.writelines("SynTableWriteFile=" + self.synTableFilePrefix + "\n")
        
        if self.synTableReadFile != "":
            fileWrite.writelines("SynTableReadFile=" + self.synTableReadFile + "\n")
            fileWrite.writelines("SynTableReadStart=" + str(self.synTableReadStart) + "\n")
            fileWrite.writelines("SynTableReadCount=" + str(self.synTableReadCount) + "\n")

        fileWrite.writelines("///////////////////////////////////////////////////////////////\n")
        fileWrite.writelines("///\n")
        fileWrite.writelines("///\n")
        
        fileWrite.writelines("FPGAExpConfFile=" + self.filename+"\n")

        fileWrite.close()
        
    def checkAccurateTimeConst(self, timeConstant, refractory):
        unitTimestep = self.timeAccelTimestep # ms
        updateCycle = 2**self.timeAccelResolution # count from MSB of mantissa, max 10 
        
        ChipPrescaler = (timeConstant * 0.693147 / updateCycle / unitTimestep) - 1
        if ChipPrescaler <= 0:
            ChipPrescaler = 0.1
        
        ActualTimeconst = (int(ChipPrescaler) + 1) * unitTimestep * updateCycle / 0.693147
        ActualArrayTimer = unitTimestep * (int(ChipPrescaler) + 1) / (1024/updateCycle)
        ActualRefractory = (int((refractory / ActualArrayTimer)) >> (10 - self.timeAccelResolution)) * 2**(10-self.timeAccelResolution) * ActualArrayTimer
        
        propUnitstep = timeConstant * 0.693147 / updateCycle
        
        return ActualTimeconst, ActualRefractory, propUnitstep
    
    def getLEModArea(self):
        
        group = np.array(self.LEModArea).T[0]
        num_of_neurons = np.array(self.LEModArea).T[1]
        site_num = np.array(self.LEModArea).T[2]
        
        return group, num_of_neurons, site_num
    
    
    def __init__(self, fname, nfname, foldername=param.NeuPLUSParamFolder):

        # Header version 1.0.1
        
        ####################
        # Neuron core configuration
        #
        self.arrayCount = 0
        self.arrayNum = []
        self.timeConstant = []
        self.refractory = []
        self.threshold = []
        self.synapticGain = []
        self.stochasticity = []
        
        
        ####################
        # Learning engine
        self.LE_inst_count = 0
        self.LE_inst = []
        
        self.LE_core_timeconstant = []
        
        # Modulation area
        self.LEModArea = []
        ######################
        # Time acceleration
        #
        self.timeAccelTimestep = 0.5 # in ms
        self.timeAccelResolution = 5 # 
        
        ######################
        # Experiment mode
        #
        self.EventSaveMode = False
        self.LoopbackMode = False
        self.InternalRoutingMode = True
        self.RealtimeInputMode = False
        self.DVS_inputMode = False
        self.TimeAccelMode = False
        self.AuxPath = False
        
        self.ExpTime = 3
        
        self.FPGATimestep = 0.1
        
        self.InputSpikesFilename = ""
        self.synTableFilePrefix = ""
        self.synTableReadFile = ""
        
        self.synTableReadStart = 0 #default
        self.synTableReadCount = 1024
        
        ######################
        # FPGA configuration
        #
        # Array clock speed
        self.FPGA_inst_count = 0
        self.chip_clock = 0 # FPGA defalut
        self.FPGA_inst = []

        # Output masking..        
        self.mask_changed = 0
        self.output_mask = np.ones(64) * 65535
        
        # router output enable
        self.router_syn_enable = np.ones(64) * 65535
        
        self.time_add_event_enable = np.zeros(16) * 255
        
        self.max_iter_count_group = np.ones(16) * 254
        
        ######
        self.foldername = foldername
        self.filename = fname
        self.neuronFilename = nfname
        
        
        
        
# %% synaptic map decompiler
class synMapDecompiler:
    '''
    This class is designed to decompile a synapse table written in a byte-coded .dat file to
    a readable synapse entries in numpy arrays.

    Returns
    -------
    None.

    '''
    def decompileSynMap(self):
        startTime = time.time()
        
        print("Reading compileed synapse tables.")
        print("This code is not yet complete. Use it very carefully.")
        print("The format of memory table is changed during development.")
        
        ########################
        # get the size of a file in byte counts.
        #
        byteSize = os.stat(self.foldername + self.fname).st_size
        print("Total ", byteSize, " bytes.")
        neuronCount = int(((byteSize-8))/(self.hashLen*16))
        
        if neuronCount > 1024 :
            print("Warning : it detects ", neuronCount, "in the file.")
            print("Warning : it might cause significant performance degradation.")
            print("Warning : it will sets the number of neuron to 1000.")
            neuronCount = 1024
        
        ########################
        # Masking informaiton
        #
        HEADER_INDICATION = 0x8000_0000
        TRAINABLE_EXIST = 0x4000_0000
        # HASH_ADDR_MASK = 
        DESTINATION_MASK = 0x0000_FFFF
        OFFSET_MASK = 0x7F00
        TRAINABLE_MASK = 0x8000
        WEIGHT_MASK = 0x00FF
        
        ########################
        # Read 4 bytes to check endian used to write the file.
        #
        endianByte = self.fileRead.read(4)
        
        endianLittle = int.from_bytes(endianByte, byteorder='little', signed=False)
        endianBig = int.from_bytes(endianByte, byteorder='big', signed=False)        
        
        if (endianLittle == 1 and endianBig != 1):
            endian = 'little'
        elif (endianLittle != 1 and endianBig == 1):
            endian = 'big'
        print("The file is encoded in ", endian, "endian.")
        
        ########################
        # Read 4 bytes 
        # This is an DRAM address 
        DRAMAddress = int.from_bytes(self.fileRead.read(4), byteorder=endian, signed=False)          
        print("Start DRAM Address is ", DRAMAddress, ".")
        ########################
        # Read 4 bytes 
        # This is SRT mode (256, 1024)
        SRTModeInt = int.from_bytes(self.fileRead.read(4), byteorder=endian, signed=False)          
                
        if (SRTModeInt == 1024):
            # 2048 is from 2**(log2(hashlen) + log2(FPGA_AXIChBW) - 3), where hashlen : 128 and FPGA_AXIChBW : 128
            source = DRAMAddress // 2048
        elif (SRTModeInt == 256):
            source = (DRAMAddress >> 29) * 65536 + (DRAMAddress // 2048) % 65536

        print("SRT is written in the mode of ", SRTModeInt, ".")
        #######################
        # Defined arrays to reduce time consumed by np.append function.
        #
        synCount = 0
        expectedSynapseCount = neuronCount * self.maxSynCount
        self.source = np.zeros(expectedSynapseCount)
        self.destination = np.zeros(expectedSynapseCount)
        self.weights = np.zeros(expectedSynapseCount)
        self.trainable = np.zeros(expectedSynapseCount)
        
        exponentArray = np.zeros(expectedSynapseCount)
        mentisaArray = np.zeros(expectedSynapseCount)
        signArray = np.zeros(expectedSynapseCount)
        
        ########################
        # Read synapses
        #
        for index_neuron in range(neuronCount):
        
            # read first bytes. It should be a header information
            byte = self.fileRead.read(4)   
            byte_to_int = int.from_bytes(byte, byteorder=endian, signed=False)
            
            #check a header
            header_checking = (byte_to_int & HEADER_INDICATION) >> 31
            trainable_checking = (byte_to_int & TRAINABLE_EXIST) >> 30
            destination = byte_to_int & DESTINATION_MASK
            

            if (header_checking != 1):
                #print("Warning : it is not a proper header information.")
                _ = 1
            else:
                #print(destination)
                _ = 1
                
            byte = self.fileRead.read(self.maxSynCount * 2)
            
            for index_dest in range(self.maxSynCount):
                
                if endian == 'big':
                    index_dest = index_dest
                elif endian == 'little':
                    if index_dest % 2 == 0:
                        index_dest += 1                        
                    elif index_dest % 2 == 1:
                        index_dest -= 1
                        
                byte_to_int = int.from_bytes(byte[index_dest*2 : index_dest*2 + 2], byteorder=endian, signed=False)
                
                offset = (byte_to_int & OFFSET_MASK) >> 8
                
                destination = destination + offset
                trainable = (byte_to_int & TRAINABLE_MASK) >> 15
                weight = byte_to_int & WEIGHT_MASK
                
                exponent = (weight & 0x70) >> 4
                mentisa = (weight & 0x0F)
                sign = ((weight & 0x80) >> 7) 
                
                if sign == 0:
                    sign = 1
                else:
                    sign = -1

                
                if byte_to_int != 0:
                    self.source[synCount] = source
                    self.destination[synCount] = destination
                    self.trainable[synCount] = trainable
                    
                    exponentArray[synCount] = exponent
                    mentisaArray[synCount] = mentisa
                    signArray[synCount] = sign
                    
                    synCount += 1
            
            if (source % 100 == 0):
                print("Neuron index ", source, "is decompiled.")

            source += 1 

        
        self.weights = number.ToFloat(np.array(exponentArray, dtype=np.uint32), np.array(mentisaArray, dtype=np.uint32), 4) * signArray
        #self.weights = (exponentArray * 16 + mentisaArray) * signArray
        
        self.source = self.source[0:synCount]
        self.destination = self.destination[0:synCount]
        self.weights = self.weights[0:synCount]
        self.trainable = self.trainable[0:synCount]
        
        print("Synaptic table decompile elapsed time is ", time.time() - startTime, " seconds.")
        
        return [self.source, self.destination, self.weights, self.trainable]
        
    def __init__(self, fname, foldername=param.NeuPLUSSynFolder):
        self.FPGA_AXIChBW = 128
        self.synEventBWidth = 16
        
        self.HeaderCount = 2
        
        self.hashLen = 128
        
        self.startAddress = 0
        
        self.maxSynCount = np.array(self.hashLen * (self.FPGA_AXIChBW / self.synEventBWidth) - self.HeaderCount, dtype=np.uint32)
        
        self.foldername = foldername
        self.fname = fname
        
        self.fileRead = open(foldername + fname, 'rb')
        
        self.source = np.array([])
        self.destination = np.array([])
        self.weights = np.array([])
        self.trainable = np.array([])
        