import os, sys, pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import neuroToolbox.utils as utils

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


class Converter:
    """
    Class for converting weight for SNN conversion.
    """
    
    def __init__(self, spike_model, config):
        """
        Initialize the Converter instance.

        Args:
            spike_model (tf.keras.Model): The compiled model for SNN from networkCompiler.
            config (configparser.ConfigParser): Configuration settings for weight conversion.
        """
        self.neurons = spike_model[0]
        self.synapses = spike_model[1]
        self.config = config

        self.v_th = self.config.getint('conversion', 'threshold')
        self.t_ref = self.config.getint('conversion', 'refractory') / 1000
        lower_bound = self.config.getfloat('conversion', 'lower_x')
        upper_bound = self.config.getfloat('conversion', 'upper_x')

        self.lower_bound = 1/self.t_ref * lower_bound
        self.upper_bound = 1/self.t_ref * upper_bound

        self.filepath = self.config['paths']['models']
        self.filename = self.config['names']['snn_model']


    def convertWeights(self):
        model = self.synapses

        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']

        print("\n\n######## Converting weight ########\n")
        
        synCnt = 0
        input_activation = x_norm
        for layer, neuron in self.synapses.copy().items():
            print(f" >>> Weight conversion for {layer} layer")
            src = np.array(neuron[0]) - synCnt
            synCnt += 1024
            tar = np.array(neuron[1]) - synCnt
            w = np.array(neuron[2])
            source = len(np.unique(src))
            target = len(np.unique(tar))
            weights = np.zeros(source * target).reshape(source, target)
            for i in range(len(w)):
                weights[src[i]][tar[i]] = w[i]
            if 'conv' in layer or layer == 'dense':
                bias = neuron[3]
            else:
                pass

            firing_rate = self.get_spikes(model=self.synapses.copy(), layer_in=layer, layer_out=layer, x=input_activation)

            scaled_firing_rate = self.min_max_scaling(firing_rate, self.lower_bound, self.upper_bound)
            scaled_firing_rate_shifted = scaled_firing_rate - self.lower_bound

            normalization_factor = np.max(scaled_firing_rate_shifted) / np.max(firing_rate)
            print(f"  | Normalization factor : {normalization_factor} |\n")

            new_weight = w * normalization_factor
            new_bias = bias + self.lower_bound

            neuron[2] = new_weight
            if 'conv' in layer or layer == 'dense':
                neuron[3] = new_bias
            else:
                pass

            input_activation = firing_rate

        with open(self.filepath + self.filename + '_Converted_synapses.pkl', 'wb') as f:
            pickle.dump(self.synapses, f)

        print(f"Weight conversion DONE.<<<\n\n\n")
    

    def min_max_scaling(self, data, new_min=0, new_max=1):

        current_min = np.min(data)
        current_max = np.max(data)

        scaled_data = [((x - current_min) / (current_max - current_min)) * (new_max - new_min) + new_min for x in data]

        return np.array(scaled_data)
    

    def get_spikes(self, model, layer_in, layer_out, x):
        start_synCnt = self.remove_keys(model, layer_in)

        spikes = []
        for input_idx in range(len(x)):
            synCnt = start_synCnt
            firing_rate = x[input_idx].flatten()
            for layer, neuron in model.items():
                src = np.array(neuron[0]) - synCnt
                synCnt += 1024
                tar = np.array(neuron[1]) - synCnt
                w = np.array(neuron[2])
                source = len(np.unique(src))
                target = len(np.unique(tar))
                weights = np.zeros(source * target).reshape(source, target)
                for i in range(len(w)):
                    weights[src[i]][tar[i]] = w[i]
                firing_rate = np.dot(firing_rate, weights)
                if 'conv' in layer:
                    s = 0
                    for oc_idx, oc in enumerate(neuron[4]):
                        firing_rate[s:oc] = firing_rate[s:oc] + neuron[3][oc_idx]
                        s = oc
                else:
                    pass
                neg_idx = np.where(firing_rate < 0)[0]
                firing_rate[neg_idx] = 0
                if layer == layer_out:
                    break
            spikes.append(firing_rate)
        
        return np.array(spikes)
    

    def get_activations(self, layer_in, layer_out, x):
            
        print("Calculating activations of layer {}.".format(layer_out.name))

        activations = tf.keras.models.Model(inputs=layer_in.input, outputs=layer_out.output).predict(x)
        
        return np.array(activations)
    

    def remove_keys(self, dictionary, target_key):
        keys_to_remove = []
        found_target_key = False

        for key in list(dictionary.keys()):
            if key == target_key:
                found_target_key = True
            if not found_target_key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del dictionary[key]

        return len(keys_to_remove) * 1024