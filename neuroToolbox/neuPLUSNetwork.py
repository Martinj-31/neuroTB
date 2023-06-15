import sys, os
os.chdir("C:/work/KIST_snn_toolbox-master")
sys.path.append(os.getcwd())

import neuralSim.parameters as param
import neuralSim.compiler as compiler
import neuralSim.synapticTable as synTable
import neuralSim.inputSpikesGen as inSpike
import neuralSim.eventAnalysis as eventAnalysis
import neuralSim.poissonSpike as spikes

from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime


class networkGen:

    def __init__(self, config, layers):
        self.config = config
        self.nCount = 1024
        self.inputSpikeFilename = "testByte.nam"
        self.synTableFilePrefix = "SynTableWrite"

        self.fname = "testExpConf.exp"
        self.nfname = "testNeuronConf.nac"
        self.conffname = "neuplusconf.txt"
        self.synfname = "testRead.dat"

        self.testSet = compiler.expSetupCompiler(self.fname, self.nfname)
        self.SynTable = synTable.synapticTable(pCount=config, maxPNCount=self.nCount, inputNCount=self.nCount)

        # Experiment setup
        self.testSet.setExperimentConf("EVENT_SAVE_MODE", config)
        self.testSet.setExperimentConf("INTERNAL_ROUTING_MODE", config)
        self.testSet.setExperimentConf("TIME_ACCEL_MODE", config)

        self.testSet.setExperimentConf("EXP_TIME", config)
        
        self.testSet.setExperimentConf("INPUT_SPIKES_FILE", self.inputSpikeFilename)
        self.testSet.setExperimentConf("SYN_TABLE_FILE_PREFIX", self.synTableFilePrefix)
        self.testSet.setExperimentConf("SYN_TABLE_READ_FILE", self.synfname)
        self.testSet.setExperimentConf("SYN_TABLE_READ_START", 0)
        self.testSet.setExperimentConf("SYN_TABLE_READ_COUNT", 1024)

    def Neurons(self, cnt):
        self.testSet.setNeuronCoreConf([cnt], [self.config], [self.config], [self.config], [0], [0])

    def Synapse_convolution(self, layers, weights):
        print(f"Connecting layer...")

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

        self.SynTable.createFromWeightMatrix(source=source, 
                                             destination=target, 
                                             weights=weights)

    def Synapse_pooling(self, layers, weights):
        print(f"Connecting layer...")

        self.SynTable.createConnections(source=layers, 
                                        destination=layers, 
                                        mode=self.config, 
                                        probability=self.config, 
                                        weight=weights, 
                                        snType=self.config, 
                                        trainable=self.config)

    def run(self):
        self.testSet.genExpConfFile(self.conffname)
        SynMapCompiled = compiler.synMapCompiler(self.synTableFilePrefix)
        SynMapCompiled.generateSynMap(synTable=self.SynTable, 
                                    inputNeurons=[range(int(self.config))], 
                                    linear=0)

        elapsed_time = param.NeuPLUSRun('-GUIModeOff', '-conf', self.conffname)
        print("Result time : ", elapsed_time)

    def Evaluate(self, datasetname):

            if datasetname == 'cifar10':
                pass
            elif datasetname == 'mnist':
                pass