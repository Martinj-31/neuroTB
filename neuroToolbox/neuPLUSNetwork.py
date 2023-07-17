import sys, os
os.chdir("C:/work/neuroTB")
sys.path.append(os.getcwd())

from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime


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
            elif 'Conv' in layer_type: # There are two types of convolution layer. 1D or 2D
                self.Synapse_convolution(layer)
            elif layer_type == 'AveragePooling2D':
                self.Synapse_pooling(layer)
            elif layer_type == 'Flatten':
                self.Synapse_flatten(layer)

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
            for target in range(weights.shape(1)):
                connections.append((source, target, weights[source, target], delay))
        
        self.connections.append(connections)

    def Synapse_convolution(self, layer):
        print(f"Connecting layer...")

        weights, _ = layer.get_weights()

        # According to image data format, parameters of feature map is different.
        # 'channel_first' : [batch_size, channels, height, width]
        # 'channel_last' : [batch_size, height, width, channels]
        ii = 1 if keras.backend.image_data_format() == 'channels_first' else 0

        height_fm = layer.input_shape[1 + ii]  # Height of feature map
        width_fm = layer.input_shape[2 + ii]  # Width of feature map
        height_kn, width_kn = layer.kernel_size  # Width and height of kernel
        sy, sx = layer.strides  # Convolution strides
        py = (height_kn - 1) // 2  # Zero-padding rows
        px = (width_kn - 1) // 2  # Zero-padding columns

        if layer.padding == 'valid':
            # In padding 'valid', the original sidelength is reduced by one less
            # than the kernel size.
            numCols = (width_fm - width_kn + 1) // sx  # Number of columns in output filters
            numRows = (height_fm - height_kn + 1) // sy  # Number of rows in output filters
            x0 = px
            y0 = py
        elif layer.padding == 'same':
            numCols = width_fm // sx
            numRows = height_fm // sy
            x0 = 0
            y0 = 0
        else:
            raise NotImplementedError("Border_mode {} not supported".format(
                layer.padding))
        
        connections = []

        # Loop over output filters 'fout'
        for fout in range(weights.shape[3]):
            for y in range(y0, height_fm - y0, sy):
                for x in range(x0, width_fm - x0, sx):
                    target = int((x - x0) / sx + (y - y0) / sy * numCols +
                                fout * numCols * numRows)
                    # Loop over input filters 'fin'
                    for fin in range(weights.shape[2]):
                        for k in range(-py, py + 1):
                            if not 0 <= y + k < height_fm:
                                continue
                            for p in range(-px, px + 1):
                                if not 0 <= x + p < width_fm:
                                    continue
                                source = p + x + (y + k) * width_fm + fin * width_fm * height_fm
                                connections.append((source, target,
                                                    weights[py - k, px - p, fin,
                                                            fout], delay))

        self.connections.append(connections)

    def Synapse_pooling(self, layer, weights):
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

        for fout in range(numFm):
            for y in range(0, height_fm - height_pl + 1, sy):
                for x in range(0, width_fm - width_pl + 1, sx):
                    target = int(x / sx + y / sy * ((width_fm - width_pl) / sx + 1) +
                                fout * width_fm * height_fm / (width_pl * height_pl))
                    for k in range(height_pl):
                        source = x + (y + k) * width_fm + fout * width_fm * height_fm
                        for j in range(width_pl):
                            connections.append((source + j, target, weight, delay))

        self.connections.append(connections)
    
    def Synapse_flatten(self, layer):

        connections = []

        self.connections.append(connections)

    def Evaluate(self, datasetname):

            if datasetname == 'cifar10':
                pass
            elif datasetname == 'mnist':
                pass