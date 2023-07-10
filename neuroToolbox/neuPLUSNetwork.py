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

        self.setup_layers()

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

    def Synapse_convolution(self, layer):
        print(f"Connecting layer...")

        weights, _ = layer.get_weights()

        # According to image data format, parameters of feature map is different.
        # 'channel_first' : [batch_size, channels, height, width]
        # 'channel_last' : [batch_size, height, width, channels]
        ii = 1 if keras.backend.image_data_format() == 'channels_first' else 0

        ny = layer.input_shape[1 + ii]  # Height of feature map
        nx = layer.input_shape[2 + ii]  # Width of feature map
        ky, kx = layer.kernel_size  # Width and height of kernel
        sy, sx = layer.strides  # Convolution strides
        py = (ky - 1) // 2  # Zero-padding rows
        px = (kx - 1) // 2  # Zero-padding columns

        if layer.padding == 'valid':
            # In padding 'valid', the original sidelength is reduced by one less
            # than the kernel size.
            mx = (nx - kx + 1) // sx  # Number of columns in output filters
            my = (ny - ky + 1) // sy  # Number of rows in output filters
            x0 = px
            y0 = py
        elif layer.padding == 'same':
            mx = nx // sx
            my = ny // sy
            x0 = 0
            y0 = 0
        else:
            raise NotImplementedError("Border_mode {} not supported".format(
                layer.padding))
        
        connections = []

        # Loop over output filters 'fout'
        for fout in range(weights.shape[3]):
            for y in range(y0, ny - y0, sy):
                for x in range(x0, nx - x0, sx):
                    target = int((x - x0) / sx + (y - y0) / sy * mx +
                                fout * mx * my)
                    # Loop over input filters 'fin'
                    for fin in range(weights.shape[2]):
                        for k in range(-py, py + 1):
                            if not 0 <= y + k < ny:
                                continue
                            for p in range(-px, px + 1):
                                if not 0 <= x + p < nx:
                                    continue
                                source = p + x + (y + k) * nx + fin * nx * ny
                                connections.append((source, target,
                                                    weights[py - k, px - p, fin,
                                                            fout], delay))

        return connections

    def Synapse_pooling(self, layer, weights):
        print(f"Connecting layer...")

        ii = 1 if keras.backend.image_data_format() == 'channels_first' else 0

        nx = layer.input_shape[2 + ii]  # Width of feature map
        ny = layer.input_shape[1 + ii]  # Height of feature map
        nz = layer.input_shape[3 - 2 * ii]  # Number of feature maps
        dx = layer.pool_size[1]  # Width of pool
        dy = layer.pool_size[0]  # Height of pool
        sx = layer.strides[1]
        sy = layer.strides[0]

        weight = 1 / (dx * dy)

        connections = []

        for fout in range(nz):
            for y in range(0, ny - dy + 1, sy):
                for x in range(0, nx - dx + 1, sx):
                    target = int(x / sx + y / sy * ((nx - dx) / sx + 1) +
                                fout * nx * ny / (dx * dy))
                    for k in range(dy):
                        source = x + (y + k) * nx + fout * nx * ny
                        for j in range(dx):
                            connections.append((source + j, target, weight, delay))

        return connections
    
    def Synapse_dense(self, layer):

        connections = []

        return connections
    
    def Synapse_flatten(self, layer):

        connections = []

        return connections

    def Evaluate(self, datasetname):

            if datasetname == 'cifar10':
                pass
            elif datasetname == 'mnist':
                pass