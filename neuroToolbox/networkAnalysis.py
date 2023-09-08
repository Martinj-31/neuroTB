import os, sys, configparser

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt

class Analysis:

    def __init__(self, x_norm, parsed_model, spike_model, config):
        self.x_norm = x_norm
        self.parsed_model = parsed_model
        self.spike_model = spike_model

        self.config = config

        self.parsed_model.summary()

    def conversionPlot(self):
        activation_dir = os.path.join(self.config['paths']['path_wd'], 'activations')

        filepath = self.config.get('paths', 'converted_model')
        filename = self.config.get('paths', 'filename_snn')
        with open(filepath + filename + '_Converted_neurons.pkl', 'rb') as f:
            neurons = pickle.load(f)
        with open(filepath + filename + '_Converted_synapses.pkl', 'rb') as f:
            synapses = pickle.load(f)

        w_list = []
        synCnt = 0
        for i, layer in enumerate(neurons.keys()):
            if i == 0:
                continue

            print(f"Analysis for {layer}...")

            if 'pooling' in layer:
                pass
            else:
                activations_file = np.load(os.path.join(activation_dir, f'activation_{layer}.npz'))
                activations = activations_file['arr_0']

            source = np.array(list(synapses.values())[i-1][0]) - synCnt
            synCnt += 1024
            target = np.array(list(synapses.values())[i-1][1]) - synCnt
            weights = np.array(list(synapses.values())[i-1][2])
            src = len(np.unique(source))
            tar = len(np.unique(target))
            w = np.zeros(src * tar).reshape(src, tar)

            for j in range(len(weights)):
                w[source[j]][target[j]] = weights[j]
            w_list.append(w)

            acts_list = []
            fr_list = []
            for input_idx in range(len(self.x_norm)):
                firing_rate = self.x_norm[input_idx].flatten()
                for w in w_list:
                    firing_rate = np.dot(firing_rate, w)
                    neg_idx = np.where(firing_rate < 0)[0]
                    firing_rate[neg_idx] = 0
                fr_list = np.concatenate((fr_list, firing_rate))

            if 'pooling' in layer:
                continue

            if 'conv' in layer:
                for ic in range(activations.shape[0]):
                    for oc in range(activations.shape[-1]):
                        acts_list = np.concatenate((acts_list, activations[ic, :, :, oc].flatten()))
            elif 'dense' in layer:
                for ic in range(activations.shape[0]):
                    for oc in range(activations.shape[-1]):
                        acts_list = np.concatenate((acts_list, activations[ic, oc].flatten()))
            elif 'pooling' in layer:
                for ic in range(activations.shape[0]):
                    for oc in range(activations.shape[-1]):
                        acts_list = np.concatenate((acts_list, activations[ic, oc].flatten()))

            plt.xlabel(f"Activations in {layer}")
            plt.ylabel(f"Firing rate in {layer}")
            plt.scatter(acts_list, fr_list)
            plt.plot(np.arange(0, 1, 0.05), np.arange(0, 1, 0.05), 'r')
            plt.show()

    def batchnormEval(self):
        b = 1