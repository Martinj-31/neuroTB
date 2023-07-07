import sys, os
os.chdir("C:/work/neuroTB")
sys.path.append(os.getcwd())

from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime


class networkGen:

    def __init__(self, config, layers):
        self.config = config

    def Synapse_convolution(self, layers, weights):
        print(f"Connecting layer...")

        # According to image data format, parameters of feature map is different.
        # 'channel_first' : [batch_size, channels, height, width]
        # 'channel_last' : [batch_size, height, width, channels]
        ii = 1 if keras.backend.image_data_format() == 'channels_first' else 0

        ny = layers.input_shape[1 + ii]  # Height of feature map
        nx = layers.input_shape[2 + ii]  # Width of feature map
        ky, kx = layers.kernel_size  # Width and height of kernel
        sy, sx = layers.strides  # Convolution strides
        py = (ky - 1) // 2  # Zero-padding rows
        px = (kx - 1) // 2  # Zero-padding columns

        if layers.padding == 'valid':
            # In padding 'valid', the original sidelength is reduced by one less
            # than the kernel size.
            mx = (nx - kx + 1) // sx  # Number of columns in output filters
            my = (ny - ky + 1) // sy  # Number of rows in output filters
            x0 = px
            y0 = py
        elif layers.padding == 'same':
            mx = nx // sx
            my = ny // sy
            x0 = 0
            y0 = 0
        else:
            raise NotImplementedError("Border_mode {} not supported".format(
                layers.padding))
        
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

    def Synapse_pooling(self, layers, weights):
        print(f"Connecting layer...")

        connections = []

        return connections

    def Evaluate(self, datasetname):

            if datasetname == 'cifar10':
                pass
            elif datasetname == 'mnist':
                pass