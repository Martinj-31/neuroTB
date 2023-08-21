import sys, os, pickle
sys.path.append(os.getcwd())

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

from neuroToolbox.neuPLUSNetwork import networkGen as net

class evaluation:
    def __init__(self, spike_model, config):
        self.config = config
        self.model = spike_model
        self.filepath = self.config.get('paths', 'path_wd')
        self.filename = self.config.get('paths', 'filename_snn')

    def genLayer(self):
        activates = np.load(self.filepath + '/activations/' + layer.name + '.npz')

        for layer in self.model.layers[1:]:
            layer_type = layer.__class__.__name__
            print(f"Generate {layer_type} for evaluation.")
            if layer_type == 'Flatten':
                continue
            elif layer_type == 'Dense':
                net.Synapse_dense(layer, evaluation=True)
                with open(self.filepath + '/' + self.filename + layer_type + '_eval.pkl'):
                    pickle.dump()

            elif 'Conv' in layer_type:
                net.Synapse_convolution(layer, evaluation=True)
                with open(self.filepath + '/' + self.filename + layer_type + '_eval.pkl'):
                    pickle.dump()

    def evaluate(self, layer):
        # Activates from DNN.
        activates = np.load(self.filepath + '/activations/' + layer.name + '.npz')

        # Spike rate from SNN.
        spike_rate = np.load(self.filepath + '/spike_train/' + layer.name + 'npz')

        x=np.arange(10)
        y=x
        plt.plot(activates, spike_rate, 'b.')
        plt.plot(x, y, 'r', alpha=0.5)
        plt.show()
