# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:15:23 2021

@author: kiryeong nam
"""

import struct
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import neuralSim.parameters as param
import neuralSim.number as number

from pathlib import Path

class MakeNeuPLUSByteFile():
    '''
    create binary file class
    
    Author : kiryeong nam 
    Date : 2020.10.07
    Description : Convert IBM dvsgesture raw aedat data to byte file data. 
    ''' 
    
    def __init__(self, 
                 header_version=1, 
                 attention=True,
                 default_frame_count=60,
                 window_size=[2, 32, 32], 
                 deltat=25,
                 timeAccelStep=0.5):

        self.header_version = header_version
        
        self.attention_window = attention

        self.default_frame_count = default_frame_count

        #print(self.data_end_point)
        self.window_size = window_size #[the number of polarity, the size of X, the size of Y]

        self.deltat = deltat # in milli seconds.
        
        self.timeAccelStep = timeAccelStep # in milli seconds.
        
        ## fixed variables
        self.unit_timestep = 100 # in micro seconds.
        self.timegap = 8
        self.n_att_events = 1000
   
        self.events = [] # [[t], [x], [y], [p]]
        
        self.confs = []
        
        self.weight_events = []
        
        self.default_strength = 20 # real value, need to be converted into FP to apply to the system.
    
    # %% Setting configuration after initialization

    def SetWindowSize(self, window_size):
        self.window_size = window_size
        
    def SetDeltaT(self, DeltaT):
        self.deltat = DeltaT    
    
        
      
    
    # %%
    def attention(self, tmad):
        
        window_size=np.array(self.window_size[1:], dtype=np.int64)
        
        # Find centroids of input events for attention window.
        centroids_x = np.convolve(tmad[:, 1], np.ones(self.n_att_events), 'valid') / self.n_att_events
        centroids_y = np.convolve(tmad[:, 2], np.ones(self.n_att_events), 'valid') / self.n_att_events
        
        tot_centroids_x = np.concatenate((tmad[:, 1][0:self.n_att_events-1], centroids_x)).astype('uint32')
        tot_centroids_y = np.concatenate((tmad[:, 2][0:self.n_att_events-1], centroids_y)).astype('uint32')
        
        tmad[:, 1] -= tot_centroids_x - window_size[0] // 2
        tmad[:, 2] -= tot_centroids_y - window_size[1] // 2
        
        attention_index = np.where((tmad[:, 1] >= 0) & (tmad[:, 1] < window_size[0]) & (tmad[:, 2] >= 0) & (tmad[:, 2] < window_size[1]))
        
        tmad = tmad[attention_index]
        
        return tmad, attention_index 
    
    # %% AEDAT to byte file
    def read_csv(self, filename):
        # read the csv file information
        
        label_filename = filename[:-6] + '_labels.csv'

        # check if there is .csv file 
        if os.path.isfile(label_filename):
            labels = np.loadtxt(label_filename,
                                skiprows=1,
                                delimiter=',',
                                dtype='uint32')
            
        else:
            print("Warning : there is no csv file.")
            labels = np.array([[-1, -1, -1]])

        return labels.T[1], labels.T[2] # start and end points.
        
            
    def unpack_aedat(self, filename, reference_time = 0):
        #read .aedat3.1 file       
        self.events = []
           
        with open(filename, 'rb') as f:  
            for i in range(5):  
                _ = f.readline()
            '''
            Each (aedat 3.1)event packet is made up of a common header, followed by its specific event data.
            This is not the same header as above, which is placed at the start of a file!
            This header is specific to each and every event packet.
            '''
            while True:
                header = f.read(28)
                if len(header) == 0:
                    break
                
                eventtype = struct.unpack('H', header[0:2])[0]
                #eventsource = struct.unpack('H', header[2:4])[0]
                eventsize = struct.unpack('I', header[4:8])[0]
                #eventoffset = struct.unpack('I', header[8:12])[0]
                #eventtsoverflow = struct.unpack('I', header[12:16])[0]
                #eventcapacity = struct.unpack('I', header[16:20])[0]
                eventnumber = struct.unpack('I', header[20:24])[0]
                #eventvalid = struct.unpack('I', header[24:28])[0]
                
                if (eventtype == 1):
                    event_bytes = np.frombuffer(f.read(eventnumber * eventsize),
                                                'uint32')
                    event_bytes = event_bytes.reshape(-1, 2)
    
                    x = (event_bytes[:, 0] >> 17) & 0x00001FFF
                    y = (event_bytes[:, 0] >> 2) & 0x00001FFF
                    p = (event_bytes[:, 0] >> 1) & 0x00000001
                    t = event_bytes[:, 1]
                    
                    #t = event_bytes[:, 1] - reference_time

                    self.events.append([t, x, y, p])
                    
                else:
                    f.read(eventnumber * eventsize) 
        
        self.events = np.column_stack(self.events)
        self.events = self.events.astype('uint32')
    
        
    def gather_aedat_file_name(self, start_id, end_id, root_directory='C:/SPB_Data/.spyder-py3/dvs/Dvsgesture_raw' , filename_prefix='user'):

        fns = []
        for i in range(start_id, end_id):
            search_mask = root_directory + os.sep + filename_prefix + "{0:02d}".format(i) + '*.aedat'
            glob_out = glob.glob(search_mask)
            if len(glob_out) > 0:
                fns += glob_out                  
        return fns
    
    # %% Final conversion to byte file.
    def plot(self):
        plt.figure()
        plt.plot(self.events[0] / 1000000, self.events[1], 'b.')
        plt.show()
        
        
    # %%
    def spike_to_byte_file(self, data_index):
        """
        input is an index of data. 
        
        converts inputs to bytes
        """
        eventnumber_header_list = []
        events_list = []
        dummy_event = []
        
       
        # Calculate t_start and t_end of data index.
        # t_start and t_end all in microsecond
        t_start = self.data_start_point[data_index] 
        t_end = self.data_end_point[data_index]
        
        # t_end_correct is the number of count in self.deltat*1000
        temp_t_end = t_end / (self.deltat*1000)
        if (temp_t_end == int(temp_t_end)):
            t_end_correct = np.array(np.floor(((t_end) - t_start)/ (self.deltat*1000) ), dtype=np.int64)
        else:
            t_end_correct = np.array(np.floor(((t_end + self.deltat*1000) - t_start)/ (self.deltat*1000) ), dtype=np.int64)
        
        time_window = range(t_start * 1000, t_start * 1000 + t_end_correct * self.deltat * 1000, self.deltat*1000)
        
        #####
        start_idx = np.searchsorted(self.events[0, :], t_start)
        end_idx = np.searchsorted(self.events[0, :], t_end, side='right')

        transposed_events = self.events[:, start_idx:end_idx].T             
        
        weight_events = self.weight_events[start_idx:end_idx]
        
        attention_index = []
        # Check attention window
        if self.attention_window is True:
            #print("attention window is defined")
            transposed_events, attention_index = self.attention(transposed_events) 
            weight_events = weight_events[attention_index]
        else:
            print("Warning : attention window is not defined.")
            pass
        
        #################################################
        ## spike events will be converted to 1-D address
        times = transposed_events[:, 0]
        data = transposed_events[:, 1:]
        
        event_type = np.full(len(times), 0)

        # convert events to 1-D address
        address = data[:,2] * self.window_size[2] * self.window_size[1] + self.window_size[1] * data[:,1] + data[:,0]
        
        # [t, address] events  
        transposed_events = np.column_stack((times, address, event_type))
        
        
        #################################################
        # %% Sort configuration events, if exists.
        #    It can provide modulation inputs
        if len(self.confs) > 0:
            
            conf_start_idx = np.searchsorted(self.confs[0, :], t_start)
            conf_end_idx = np.searchsorted(self.confs[0, :], t_end, side='right')
            
            transposed_conf_events = self.confs[:, conf_start_idx:conf_end_idx].T
            
            conf_time = transposed_conf_events[:,0]
            conf_event = transposed_conf_events[:, 1:]
            
            commandOffset = 3489660928 # 0xD000_0000
            
            # convert to 1-D events
            # conf_event[:,0] ## core
            # conf_event[:,1] ## addr
            # conf_event[:,2] ## value
            conf_event_1D = np.array(commandOffset + conf_event[:,0]*(2**24) + 4 * (2**20) + 1 * (2**18) + conf_event[:,1] * (2**8) + conf_event[:,2], dtype=np.uint32)
            
        else:
            conf_time = []
            conf_event_1D = []
            
            
        #################################################
        # %% For time acclearation mode
        # time_add_hex = 0xC020_0000; # event at every self.timeAccelStep 
        # end_of_events = 0xE01E_0000; # end of all events
        count_accel_time = np.array(np.ceil(t_end_correct * self.deltat*1000 / (self.timeAccelStep * 1000)), dtype=np.int64)
        
        accel_time = np.linspace(self.timeAccelStep * 1000, self.timeAccelStep*1000*(count_accel_time), count_accel_time) + (t_start * 1000)
        
        time_add_hex = np.full(len(accel_time), 0xC020_0000)
        
        #accel_time = []
        #time_add_hex = []
        
        #################################################
        ## Comebine transposed events to times and spike events

        # This address setting is for order of sorting.
        # dummy event should be the first at the same timestamp.
        dummy_event_address = -5
        time_add_event_address = -4
        
        # make default events and set target address -1 to find it using index later.
        dummy_event = np.vstack([np.array(time_window), np.full(len(time_window), dummy_event_address), np.full(len(time_window), 1)]).T 
        
        conf_event_transposed = np.vstack([np.array(conf_time), np.full(len(conf_time), -2), np.array(conf_event_1D) ]).T
        time_add_event_transposed = np.vstack([accel_time, np.full(len(accel_time), time_add_event_address), time_add_hex]).T
        
        # Merge dummy_events, events, and conf_events
        all_events = np.vstack((transposed_events,  conf_event_transposed, time_add_event_transposed, dummy_event )).tolist()
        
        # Sort by timestamp
        all_events.sort(key=lambda x : (x[0], x[1]))
        all_events = np.array(all_events)
        
        
        times = all_events.T[0]
        spikes = all_events.T[1]
        eventType = all_events.T[2]
        
        #################################################
        idx_start = 0
        idx_end = 0
        
        weight_count = 0
        
        for i, t in enumerate(time_window): 
 
           idx_end += np.searchsorted(times[idx_end:], t + self.deltat*1000)

           if idx_end > idx_start:
               # slicing elements from original arrays.
               timestamp = times[idx_start : idx_end  + 1]
               address = spikes[idx_start : idx_end + 1]
               eventTypeFrame = eventType[idx_start : idx_end + 1]
               
               without_timeadd_index = np.where(address != time_add_event_address)[0]
               
               with_timeadd_event = np.zeros(len(timestamp))
               
               if len(without_timeadd_index) > 0:
                   timestamp_ISI = timestamp[without_timeadd_index]
                   address_temp = address[without_timeadd_index]
               else:
                   timestamp_ISI = timestamp
                   address_temp = address
               
               # Strength is 8-bit FP
               if self.weight_events != []:
                   # self.weight_events need to be converted to 8-bit FP
                   # not yet implemented
                   Strength = self.weight_events[weight_count:weight_count + len(timestamp_ISI)]
                   weight_count += timestamp_ISI
               else:
                   strength_exponent, strength_mantissa, strength_error = number.FloatSplitToInt([self.default_strength], 3, 4)
                   Strength = np.full(len(timestamp_ISI), strength_exponent * 2**4 + strength_mantissa)
               
               
               # Caculatate ISI and strength
               past_timestamp = np.hstack([timestamp_ISI[0], timestamp_ISI[0:-1]])
               ISI = np.around((timestamp_ISI - past_timestamp) / self.unit_timestep).astype(int)
               
               # find ISI index where ISI >= 8ms
               ISI_big_index = np.where(ISI >= self.timegap)[0]
               ISI_big = ISI[ISI_big_index]
               ISI[ISI_big_index] = 0
               
               # fill 0 to ISI where ISI >= 8ms index and insert 0 to ISI where ISI >= 8ms index+1 because there will be filled with delay event
               direct_neuron_stimulus_event = 5*(2**29) + ISI*2**26 + Strength*2**18 + address_temp
               #direct_neuron_stimulus_event = np.insert(direct_neuron_stimulus_event, ISI_big_index, ISI_big + 1*2**31)
               
               with_timeadd_event[without_timeadd_index] = direct_neuron_stimulus_event
               with_timeadd_event = np.insert(with_timeadd_event, without_timeadd_index[ISI_big_index], ISI_big + 1*2**31)

               
               address = np.insert(address, without_timeadd_index[ISI_big_index], 0)
               eventTypeFrame = np.insert(eventTypeFrame, without_timeadd_index[ISI_big_index], 0)
               
               # Find conf event index
               # Replace modulate events
               conf_event_index = np.where(address == -2)[0]
               if len(conf_event_index) > 0 :
                   with_timeadd_event[conf_event_index] = eventTypeFrame[conf_event_index] 

               
               time_add_event_index = np.where(address == time_add_event_address)[0]
               if len(time_add_event_index) > 0:
                   with_timeadd_event[time_add_event_index] = eventTypeFrame[time_add_event_index]
               
               
               # dummy events index
               dummy_event_index = np.where(address == dummy_event_address)[0]
               
               # Delete dummy events
               with_timeadd_event = np.delete(with_timeadd_event, dummy_event_index, axis=0)
               
               # End of event stream in the last frame.
               if (i == len(time_window) - 1):
                   with_timeadd_event = np.insert(with_timeadd_event, len(with_timeadd_event), 0xE01E_0000)
               else:
                   with_timeadd_event = np.insert(with_timeadd_event, len(with_timeadd_event), 0xE01E_0001)
                   
               # timer off
               with_timeadd_event = np.insert(with_timeadd_event, len(with_timeadd_event), 0xC040_0000)
               # timer on
               with_timeadd_event = np.insert(with_timeadd_event, 0, 0xC040_0001)
               
               
               # num of eventnumber included in deltat(ex. 25ms)
               eventnumber = len(with_timeadd_event)
               
                   
               if eventnumber != 0: 
                   eventnumber_header_list.append(eventnumber)
                   #print("event count : ", eventnumber)
               else: 
                   print('No event')
               
               events_list.extend(with_timeadd_event)

               
           idx_start = idx_end
        
        
        return eventnumber_header_list, events_list 
  
    def collect_events(self):
        """
        Read each user event data
        """
        
        add_eventnumber_header_list = []
        add_events_list = []
        add_frame_count = []

        
        for i in range(len(self.data_start_point)):

             # convert event data to binary data
             eventnumber_header_list, events_list = self.spike_to_byte_file(i)

             frame_count = len(eventnumber_header_list)
             
             add_eventnumber_header_list.extend(eventnumber_header_list) 
             add_events_list.extend(events_list)                    
             add_frame_count.append(frame_count)
        
        if self.data_frame_count == None:
            self.data_frame_count = add_frame_count

        return add_eventnumber_header_list, add_events_list

    # %% Byte generation
    # Byte generation case 1 : convert an aedat file to a byte file for a NeuPLUS system.
    def aedat_to_bytefile(self, aedat_filename, 
                          default_frame_count=None, 
                          data_frame_count=None,
                          data_start_point=None):
   
        print("Processing aedat file conversion.")
         
        byte_file_name = Path(aedat_filename).stem
        
        self.unpack_aedat(aedat_filename)

        data_end_point = None
        
        self.SetDataPoints(data_frame_count, data_start_point, data_end_point)
        
        self.make_byte_file(byte_file_name)
        
    # Byte generation case 2 : Convert events with timestamp to a byte file for a NeuPLUS system.
    def events_to_bytefile(self, events, byte_file_name, 
                           conf_events=[],
                           weight_events=[],
                           weight_single=20,
                           dimension=1,
                           attention_mode=False,
                           data_frame_count=None,
                           data_start_point=None,
                           data_end_point=None
                           ):
        """
        convert event to byte file
        """
        # events = [[t], [x], [y], [p]]
        print("------------------------------------------------------------")
        print("(inputSpikesGen.py) Start to input spike files.\n")
        
        # Check the dimension of event stream.
        if dimension == 1:
            events.append([0] * len(events[0]))
            events.append([0] * len(events[0]))
            
        elif dimension == 2:
            events.append([0] * len(events[0]))
        
        self.events = np.array(events)
        self.events[0] = self.events[0] * 1000000 #convert time in second to micro second.
        self.events = self.events.astype('int64')
        
        if weight_events != []:
            self.weight_events = weight_events
            if len(self.events[0]) != len(self.weight_events):
                print("(ERROR : inputSpikesGen.py) the size of weight matrix does not match to the number of events.")
        
        self.default_strength = float(weight_single)
            
        if len(conf_events) > 0:
            self.confs = np.array(conf_events)
            self.confs[0] = self.confs[0] * 1000000
            self.confs = self.confs.astype('int64')
        
        self.attention_window = attention_mode
        
        self.SetDataPoints(data_frame_count, data_start_point, data_end_point)
        
        self.make_byte_file(byte_file_name)
        
    # %% Calibrate data points
    def SetDataPoints(self, data_frame_count, data_start_point, data_end_point):
        
        # Set data start point
        if data_start_point is not None:
            self.data_start_point = data_start_point 
        else:
            self.data_start_point = [0]

        # Set data end point
        if data_end_point is None:
            if data_frame_count is not None:
                self.data_frame_count = data_frame_count
                
                self.data_end_point = [self.events[0][-1]] # end of event time.
            else:
                if len(self.data_start_point) == 1: # all events
                    self.data_frame_count = None
                    
                    self.data_end_point = [self.events[0][-1]] # end of event time.
                else:
                    self.data_frame_count = [self.default_frame_count for i in range(len(self.data_start_point))]
                
                    self.data_end_point = (np.array(self.data_start_point) + np.array(self.data_frame_count) * self.deltat * 1000).tolist()
                                
        else:
            if data_start_point is not None:
                assert data_end_point > data_start_point, "Warning : Check the start point and end point." 
                assert len(data_start_point) == len(data_end_point), "Warning : Check the start point and end point."
            self.data_end_point = data_end_point
            
            # if data_end_point is defined, input data_frame_count is ignored and overwritten to 'None'.
            self.data_frame_count = None
        
        ####
        print("\nData point setting...")
        print('self.data_start_point:', self.data_start_point)
        print('self.data_end_point:', self.data_end_point)
        print('self.data_frame_count:', self.data_frame_count,"\n") #: None
        
    # %% Final conversion to byte file.
    def make_byte_file(self, byte_file_name):
        """
        make byte file 
        """
        
        eventnumber_header_list, events_list = self.collect_events()
        
        # convert list to byte file
        eventnumber_header_byte = np.array(eventnumber_header_list, dtype='uint32').tobytes() # convert list to numpy array        
        events_byte = np.array(events_list, dtype='uint32').tobytes() # convert list to numpy array
            
        byte_file_name = Path(byte_file_name).stem        
        
        byte_file_gen = open(param.NeuPLUSParamFolder + byte_file_name + '.nam', 'wb') 

        # make header of byte file
        header_version_byte = struct.pack('I',self.header_version) #header version
        time_resolution_byte = struct.pack('I',self.deltat) #time resolution
        num_of_data_byte = struct.pack('I',len(self.data_frame_count))        
        num_of_frame_byte = struct.pack('I',sum(self.data_frame_count))
        
        byte_file_gen.write(header_version_byte)
        byte_file_gen.write(time_resolution_byte)
        byte_file_gen.write(num_of_data_byte)
        byte_file_gen.write(num_of_frame_byte) 

        byte_file_gen.write(np.array(self.data_start_point, dtype='uint32').tobytes())
        byte_file_gen.write(np.array(self.data_frame_count, dtype='uint32').tobytes())
        byte_file_gen.write(eventnumber_header_byte) #num of events per deltat(25ms)        
        
        #print(eventnumber_header_list)
        #print(events_list)
        #print(sum(self.data_frame_count))
        # write neuron address information to byte file 
        byte_file_gen.write(events_byte)
        byte_file_gen.close()
        
        #print(self.data_frame_count)
        print("(inputSPikesGen.py) A input spike file is generated.\n")
        print("------------------------------------------------------------")
        
# %% Frame-to-Poisson 
class FrameToPoissonCompiler:
    '''
    This class is for generating an event file mapping to a Poisson spike generator.
    The event file has to be in a format with
    Header : [mode, frameCount, neuronCount, PoissonMode, Timescale]
    Frame 1 : frame data
    Frame 2 : frame data
           ...
           ...
           ...
    '''
    def generateHeader(self, frameCount, neuronCount, mode=0, poissonMode=1, timestep=0.0001):
        '''
        It generates a header defining a type of an event file.

        Parameters
        ----------
        frameCount : integer
            The number of frame.
        neuronCount : integer
            The number of neurons.
        mode : integer (0 : Direct to neuron, 1 : To DRAM)
            Default is 0.
        poissonMode : integer
            Set the mode of spike train. (1 : Poisson, 0 : regular)
        timestep : integer
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        timescale = timestep*1000000000/(6.66*15)
        
        self.neuronCount = neuronCount
        self.poissonMode = poissonMode
        self.timestep = timestep
        self.mode = mode
        
        header = np.array([self.mode, frameCount, self.neuronCount, self.poissonMode, timescale], dtype=np.uint32)
        self.f.write(bytearray(header))

        self.header = True
        
        return header
        
    def generateFrame(self, frametime, index, target, fin, weight=-1):
        '''
        It generates a frame of Poisson spike trains.

        Parameters
        ----------
        frametime : integer
            The duration of a timewindow for a frame in ms.
        index : np.array
            An array of addresses of an entry in an SRAM in a Poisson spike generator.
        target : np.array
            An array of addresses of target neurons in silicon neuron arrays.
        fin : np.array
            An array of frequency of spike trains.
        weight : np.array
            An array of weight of spike trains.
        dram : TYPE, optional
            Spike train targets DRAM (1) or SNA (0). The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if self.header == False:
            print("Error : Please define a header first.")
            return False
        
        # Correct data types.
        # Data type should be 32-bit unsigned integer 
        index = np.array(index, dtype=np.uint32)
        target = np.array(target, dtype=np.uint32)
        fin = np.array(fin, dtype=np.uint32)
        
        if type(weight) == int:
            if weight == -1:
                weight = np.array(np.ones(len(fin)) * 10, dtype=np.uint32)
            else:            
                weight = np.array(weight, dtype=np.uint32)
        else:
            weight = np.array(weight, dtype=np.uint32)
        
        self.f.write(bytearray(np.array([frametime], dtype=np.uint32)))
        command = np.ones(len(fin) + 1, dtype=np.uint32) * 196

        if self.mode == 1:
            tempAddress = 134217728 + target
        elif self.mode == 0:
            tempAddress = 67108864 + 262144 * weight + target
        

        data1 = np.array(index*4194304 + (tempAddress >> 6), dtype=np.uint32)
        
        zeroIndex = np.where(fin == 0)
        fin[zeroIndex] = 1
        
        ISI = np.clip(np.array((1/fin)/self.timestep, dtype=np.uint32), 0, 8191)
        ISI[zeroIndex] = 0
        fin[zeroIndex] = 0
        
        if self.poissonMode == 1: #Poisson
            data2 = np.array((tempAddress % 64) * 67108864 + fin * 6710.8863, dtype=np.uint32)
        else: #Regular
            data2 = np.array((tempAddress % 64) * 67108864 + ISI * 8192 + 1, dtype=np.uint32)
            
        # append ending event
        data1 = np.array(np.append(data1, np.array([0])), dtype=np.uint32)
        data2 = np.array(np.append(data2, np.array([0])), dtype=np.uint32)
        
        #print(len(command))
        #print(len(data1))
        #print(len(data2))
        merge = np.array([command, data1, data2]).T.flatten()
        #print(merge)
        self.f.write(bytearray(merge))
        
        return merge
        
    def fileGenerate(self):
        '''
        It generates an event file mapping to a Poisson spike generator.

        Returns
        -------
        None.

        '''
        if self.header == False:
            print("Error : Please define a header first.")
            return False
        
        self.f.close()
        
    def __init__(self, foldername, fname):
        '''
        This class for generating an event file mapping to a Poisson spike generator.

        Parameters
        ----------
        foldername : string like
            Folerder name where to create a file.
        fname : string
            File name which is to be created.

        Returns
        -------
        None.

        '''
        self.f = open(foldername + fname, 'wb')
        self.neuronCount = 0
        
        self.poissonMode = 1 # default is Poisson.
        
        self.header = False
        
        self.timestep = 0.0001
        
        self.mode = 0
        



def rewardGeneration(time, address, value, configuration):
    '''
    input
    - time     : second
    - address  : neuron address, where the modulator will be applied (0 ~ 1M)
    - value    : modulation strength in real value 
    - configuration : compiler class (from compiler)
    '''
    
    '''
    # input handling
    
    # need to get learning engine configuration
    # Group (0~16), np.array
    # site_num, np.array
    '''
    group, num_of_neurons, site_num = configuration.getLEModArea()
    
    # To ensure the type of return value to numpy array.
    group = np.array(group, dtype=np.int32)
    num_of_neurons = np.array(num_of_neurons, dtype=np.int32)
    site_num = np.array(site_num, dtype=np.int32)
    
    # To ensure the type of input array to numpy array. 
    time = np.array(time)
    address = np.array(address, dtype=np.int32)
    value = np.array(value, dtype=np.float)
    
    # Value conversion real value to 8-bit FP.
    floatTemp = number.FloatSplitToInt(value, 3, 4)
    value_float = floatTemp[0] * 16 + floatTemp[1]
                
    
    
    '''
    #  Address conversion
    
    # Input address is the address of a neuron in 1-M NeuPLUS system. 
    # This address need to be translated.
    # 
    # address conversion to modulation site
    # address of the neuron --> (core, mod_add)
    # mod_add is an address of SRAM in modulation memory module.
    '''
    # offset calculation
    offset = np.array(np.log2(num_of_neurons), dtype=np.int32) # np.array
    
    # Address list which modulator memory modules can be used to.
    start = group * (64*1024) + site_num * (2**(10 + offset))
    end = start + 1024*num_of_neurons
    
    
    # Modulator memory configuration check
    offset_array = np.zeros(16)
    site_num_array = np.zeros(16)
    
    for j in range(16):
        index = np.where(group == j)[0]
        if len(index) != 0:
            offset_array[j] = offset[index]
            site_num_array[j] = site_num[index]
    
    
    conv_count = 0
    
    mod_time = []
    mod_group = []
    mod_add = []
    value_sent = []
    
    for i in range(len(start)):
        within_router_index = np.where((address >= start[i]) & (address < end[i]))[0]
        
        
        if (len(within_router_index) == 0):
            print("(inputSpikesGen.py) Info : there is no modulation event in the event router of ", group[i])
        else:
            # find 
            address_within_router = address[within_router_index]

            # address_within_router (0 ~ 1M) --> group_in_system, mod_add_in_memory
            # group_in_system (0 ~ 15) : each group controls 64-k neurons
            # mod_add_in_memory (0 ~ 1024) : The address of SRAM for modulation.
            group_in_system = address_within_router // (64*1024)    
            
            mod_start_neuron_add = group_in_system * (64*1024) + site_num_array[group_in_system] * (2**(10 + offset_array[group_in_system]))
           
            mod_add_in_memory = (address_within_router - mod_start_neuron_add) // (2**offset_array[group_in_system])
            
            
            
            mod_time.extend(time[within_router_index].tolist())
            mod_group.extend(group_in_system.tolist())
            mod_add.extend(mod_add_in_memory.tolist())
            value_sent.extend(value_float[within_router_index].tolist())
        
            conv_count += len(address_within_router)
            
    if len(address) != conv_count:
        print("(inputSpikesGen.py) Warning : there are events which are not in modulation site. The modulation will not be applied.")
    
    return [mod_time, mod_group, mod_add, value_sent]


  