import sys, os
sys.path.append(os.getcwd())

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from tensorflow import keras
from datetime import date
import numpy as np
import configparser
import matplotlib.pyplot as plt

class networkGen:

    def __init__(self, parsed_model, config):
        self.config = config
        self.parsed_model = parsed_model
        self.num_classes = int(self.parsed_model.layers[-1].output_shape[-1])

        self.layers = []
        self.connections = []

        # mode On/Off

    def setup_layers(self, input_shape):
        self.add_input_layer(input_shape)
        for layer in self.parsed_model.layers[1:]:
            print(f"Building layer for {layer.name}")
            self.add_layer(layer)
            layer_type = layer.__class__.__name__
            if layer_type == 'Dense':
                self.Synapse_dense(layer)
            # There are two types of convolution layer. 1D or 2D
            elif 'Conv' in layer_type: 
                self.Synapse_convolution(layer)
            elif layer_type == 'AveragePooling2D':
                self.Synapse_pooling(layer)
            elif layer_type == 'Flatten':
                self.Synapse_flatten(layer)
            
            print("Layers : ", self.layers)
            print("Connections : ", self.connections)

    # Input will be made in neuralSim library.
    def add_input_layer(self, input_shape):
        neurongroup = {}
        neurongroup['N'] = np.prod(input_shape[1:])

        self.layers.append(neurongroup)

    def add_layer(self, layer):
        neurongroup = {}
        neurongroup['N'] = np.prod(layer.output_shape[1:])

        self.layers.append(neurongroup)

    def Synapse_dense(self, layer):
        weights, _ = layer.get_weights()

        connections = []

        for source in range(weights.shape[0]):
            for target in range(weights.shape[1]):
                connections.append([source, target, weights[source, target]]) # remove delay
        
        self.connections.append(connections)

    def Synapse_convolution(self, layer):
        """_summary_
        This method is for generating synapse connection from CNN layer to SNN layer with neuron index.

        Args:
            layer (Keras.model): Keras CNN model with weight information.
            weights (): Data shape is [filter_height, filter_width, input_channels, output_channels].
            height_fm (int): Height of feature map
            width_fm (int): Width of feature map
            height_kn, width_kn (int): Width and height of kernel
            stride_y, stride_x (int): strides
            numCols (): Number of columns in output filters (horizontal moves)
            numRows (): Number of rows in output filters (vertical moves)
            padding_y, paddding_x (int): Zero-padding rows, Zero-padding columns. It is (filter_size-1)/2

        Raises:
            NotImplementedError: _description_
        """
        print(f"Connecting layer...")

        w, _ = layer.get_weights()

        # 'channel_first' : [batch_size, channels, height, width]
        # 'channel_last' : [batch_size, height, width, channels]
        ii = 1 if keras.backend.image_data_format() == 'channels_first' else 0

        input_channels = w.shape[2]
        output_channels = w.shape[3]
        height_fm = layer.input_shape[1 + ii]
        width_fm = layer.input_shape[2 + ii]
        height_kn, width_kn = layer.kernel_size
        stride_y, stride_x = layer.strides

        fm = np.arange(width_fm*height_fm).reshape((width_fm, height_fm))
        if 'valid' == layer.padding:
            padding_y = 0
            padding_x = 0
            numCols = int((width_fm - width_kn)/stride_x + 1)
            numRows = int((height_fm - height_kn)/stride_y + 1)
            FM = fm
            for i in range(1, input_channels):
                FM = np.concatenate((FM, fm+(width_fm*height_fm*i)), axis=0)
        elif 'same' == layer.padding:
            padding_y = (height_kn - 1) // 2
            padding_x = (width_kn - 1) // 2
            numCols = int((width_fm - width_kn + 2*padding_x)/stride_x + 1)
            numRows = int((height_fm - height_kn + 2*padding_y)/stride_y + 1)
            fm_pad = np.pad(fm, ((padding_y, padding_y), (padding_x, padding_x)), mode='constant', constant_values=-1)
            FM = fm_pad
            for i in range(1, input_channels):
                FM = np.concatenate((FM, np.pad(fm+(width_fm*height_fm*i), ((padding_y, padding_y), (padding_x, padding_x)), mode='constant', constant_values=-1)), axis=0)

        height_fm = int(FM.shape[0]/input_channels)
        width_fm = FM.shape[1]

        FM = FM.flatten() # Make feature map flatten to indexing

        source = np.zeros(numCols*numRows*(height_kn*width_kn)*input_channels*output_channels)
        target = np.zeros(numCols*numRows*(height_kn*width_kn)*input_channels*output_channels)
        weights = np.zeros(numCols*numRows*(height_kn*width_kn)*input_channels*output_channels)

        idx = 0
        fin = 0
        target_cnt = 0
        reset_cnt = 0
        for fout in range(output_channels):
            row_idx = 0
            for i in range(numCols*numRows*input_channels):
                if 0 == i%(numCols*numRows) and i != 0:
                    if 'valid' == layer.padding:
                        row_idx += 2*width_fm + width_kn - stride_x
                    else:
                        row_idx += 2*padding_y*width_fm + width_kn - stride_x
                    fin += 1
                    target_cnt = reset_cnt
                    if fin == input_channels-1:    
                        fin = 0
                # When stride is more than 1,
                elif 0 == i%numCols and i != 0:
                    row_idx += width_fm*(stride_y-1) + width_kn - stride_x
                for j in range(height_kn):
                    source[idx:idx+width_kn] = FM[row_idx+i+(j*width_fm):row_idx+i+(j*width_fm)+width_kn]
                    target[idx:idx+width_kn] = np.zeros(len(source[idx:idx+width_kn])) + target_cnt
                    weights[idx:idx+width_kn] = w[j, 0:width_kn, fin, fout]
                    idx += width_kn
                row_idx += (stride_x-1)
                target_cnt += 1
            reset_cnt = target_cnt
                
        if 'same' == layer.padding:
            padding_idx = np.where(source == -1)[0]
            source = np.delete(source, padding_idx)
            target = np.delete(target, padding_idx)
            weights = np.delete(weights, padding_idx)

        source = source.astype(int)
        target = target.astype(int)
        weights = weights.astype(int)

        self.connections.append([source, target, weights])

    def Synapse_pooling(self, layer):
        print(f"Connecting layer...")

        ii = 1 if keras.backend.image_data_format() == 'channels_first' else 0

        width_fm = layer.input_shape[2 + ii]  # Width of feature map
        height_fm = layer.input_shape[1 + ii]  # Height of feature map
        numFm = layer.input_shape[3 - 2 * ii]  # Number of feature maps
        width_pl = layer.pool_size[1]  # Width of pool
        height_pl = layer.pool_size[0]  # Height of pool
        sx = layer.strides[1]
        sy = layer.strides[0]

        weight = 1 / (width_pl * height_pl)

        connections = []

        for output_channel in range(numFm):
            for y in range(0, height_fm - height_pl + 1, sy):
                for x in range(0, width_fm - width_pl + 1, sx):
                    target = int(x / sx + y / sy * ((width_fm - width_pl) / sx + 1) + output_channel * width_fm * height_fm / (width_pl * height_pl))
                    for k in range(height_pl):
                        source = x + (y + k) * width_fm + output_channel * width_fm * height_fm
                        for j in range(width_pl):
                            connections.append([source + j, target, weight]) # remove delay

        self.connections.append(connections)
    
    def Synapse_flatten(self, layer):

        connections = []

        self.connections.append(connections)

    def Evaluate(self, datasetname):

            if datasetname == 'cifar10':
                pass
            elif datasetname == 'mnist':
                pass

    def build(self):

        return self.layers, self.connections