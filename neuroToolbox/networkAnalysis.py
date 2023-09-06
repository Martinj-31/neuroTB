import os, sys, configparser

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from tensorflow import keras
import numpy as np

class networkAnalysis:

    def __init__(self, x_norm, parsed_model, spike_model, config):
        self.x_norm = x_norm
        self.parsed_model = parsed_model
        self.spike_model = spike_model

        self.config = config

        self.input_model.summary()
        self.parsed_model.summary()

    def conversionPlot(self):
        activation_dir = os.path.join(self.config['paths']['path_wd'], 'activations')
        
        for layer in self.parsed_model.layers:
            activations = np.load(os.path.join(activation_dir, layer.name))


    def batchnormEval(self):
        b = 1