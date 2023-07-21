import sys, os
os.chdir("C:/work/neuroTB")
sys.path.append(os.getcwd())

from tensorflow import keras
from datetime import date
import numpy as np
import configparser
import matplotlib.pyplot as plt

class networkGen:

    def __init__(self, config, parsed_model):
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

        Args:
            layer (_type_): _description_
            weights (): Data shape is [filter_height, filter_width, input_channels, output_channels].
            height_fm (int): Height of feature map
            width_fm (int): Width of feature map
            height_kn, width_kn (int): Width and height of kernel
<<<<<<< Updated upstream
            sy, sx (int): strides
            numCols (): Number of columns in output filters (horizontal moves)
            numRows (): Number of rows in output filters (vertical moves)
            py, px (int): Zero-padding rows, Zero-padding columns. It is (filter_size-1)/2
=======
            stride_y, stride_x (int): strides
            numCols (): Number of columns in output filters (horizontal moves)
            numRows (): Number of rows in output filters (vertical moves)
            padding_y, paddding_x (int): Zero-padding rows, Zero-padding columns. It is (filter_size-1)/2
>>>>>>> Stashed changes

        Raises:
            NotImplementedError: _description_
        """
        print(f"Connecting layer...")

        weights, _ = layer.get_weights()
        # print(weights)

        # 'channel_first' : [batch_size, channels, height, width]
        # 'channel_last' : [batch_size, height, width, channels]
        ii = 1 if keras.backend.image_data_format() == 'channels_first' else 0

        height_fm = layer.input_shape[1 + ii]
        width_fm = layer.input_shape[2 + ii]
        height_kn, width_kn = layer.kernel_size
<<<<<<< Updated upstream
        sy, sx = layer.strides
        py = (height_kn - 1) // 2
        px = (width_kn - 1) // 2
=======
        stride_y, stride_x = layer.strides
        padding_y = (height_kn - 1) // 2
        padding_x = (width_kn - 1) // 2
>>>>>>> Stashed changes

        if layer.padding == 'valid':
            # In padding 'valid', the original sidelength is reduced by one less
            # than the kernel size.
<<<<<<< Updated upstream
            numCols = (width_fm - width_kn + 1) // sx
            numRows = (height_fm - height_kn + 1) // sy
            x0 = px
            y0 = py
        elif layer.padding == 'same': # "Output image and input image are same."
            numCols = width_fm // sx
            numRows = height_fm // sy
=======
            numCols = (width_fm - width_kn + 1) // stride_x
            numRows = (height_fm - height_kn + 1) // stride_y
            x0 = padding_x
            y0 = padding_y
        elif layer.padding == 'same':
            # In padding 'same', output image and input image are same.
            numCols = width_fm // stride_x
            numRows = height_fm // stride_y
>>>>>>> Stashed changes
            x0 = 0
            y0 = 0
        else:
            raise NotImplementedError("Border_mode {} not supported".format(layer.padding))
        
        connections = []

<<<<<<< Updated upstream
        for output_channels in range(weights.shape[3]):
            for y in range(y0, height_fm - y0, sy):
                for x in range(x0, width_fm - x0, sx):
                    target = int((x - x0) / sx + (y - y0) / sy * numCols + output_channels * numCols * numRows)
                    for input_channels in range(weights.shape[2]):
                        for k in range(-py, py + 1):
=======
        for output_channel in range(weights.shape[3]):
            for y in range(y0, height_fm - y0, stride_y):
                for x in range(x0, width_fm - x0, stride_x):
                    target = int((x - x0) / stride_x + (y - y0) / stride_y * numCols + output_channel * numCols * numRows)
                    for input_channel in range(weights.shape[2]):
                        for k in range(-padding_y, padding_y + 1):
>>>>>>> Stashed changes
                            if not 0 <= y + k < height_fm:
                                continue
                            for p in range(-padding_x, padding_x + 1):
                                if not 0 <= x + p < width_fm:
                                    continue
<<<<<<< Updated upstream
                                source = (x + p) + (y + k) * width_fm + input_channels * width_fm * height_fm
                                connections.append([source, target, weights[py - k, px - p, input_channels, output_channels]]) # remove delay
        for a in zip(*connections):
            print(a)
=======
                                source = x + p + ((y + k) * width_fm) + (input_channel * width_fm * height_fm)
                                connections.append([source, target, weights[padding_y - k, padding_x - p, input_channel, output_channel]]) # remove delay
        # cons = np.array(connections)
        # print(cons)
        # np.savetxt('connections', cons, fmt='%d')
        print("Length : ", len(connections))
>>>>>>> Stashed changes
        self.connections.append(connections)

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
<<<<<<< Updated upstream
                    target = int(x / sx + y / sy * ((width_fm - width_pl) / sx + 1) + fout * width_fm * height_fm / (width_pl * height_pl))
=======
                    target = int(x / sx + y / sy * ((width_fm - width_pl) / sx + 1) + output_channel * width_fm * height_fm / (width_pl * height_pl))
>>>>>>> Stashed changes
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