import os, sys, pickle, math
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
        self.input_model_name = config["names"]["input_model"]

        self.parsed_model = tf.keras.models.load_model(os.path.join(self.config["paths"]["models"], f"parsed_{self.input_model_name}.h5"))

        self.v_th = self.config.getint('conversion', 'threshold')
        self.t_ref = self.config.getint('conversion', 'refractory') / 1000
        lower_bound = self.config.getfloat('conversion', 'lower_x')
        upper_bound = self.config.getfloat('conversion', 'upper_x')

        self.min = 1/self.t_ref * lower_bound
        self.max = 1/self.t_ref * upper_bound
        self.min_bound = self.min / (self.v_th - self.min*self.t_ref)
        self.max_bound = self.max / (self.v_th - self.max*self.t_ref)

        self.filepath = self.config['paths']['models']
        self.filename = self.config['names']['snn_model']


    def convertWeights(self):
        activation_dir = os.path.join(self.config['paths']['path_wd'], f"parsed_model_activations")

        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']

        print("\n\n######## Converting weight ########\n")

        for layer in self.parsed_model.layers:

            if 'input' in layer.name or 'flatten' in layer.name:
                continue

            print(f" Weight conversion for {layer} layer...")

            neuron = self.synapses[layer.name]

            w = np.array(neuron[2])
            if 'conv' in layer.name or layer.name == 'dense':
                bias = neuron[3]
            else: pass

            cur_activation_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{layer.name}.npz"))
            cur_activation = cur_activation_file['arr_0']

            inbound = utils.get_inbound_layers_with_params(layer)
            pre_layer = inbound[0]
            pre_activation_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{pre_layer.name}.npz"))
            pre_activation = pre_activation_file['arr_0']

            log_scaled_activation = np.log10(cur_activation+1)
            log_scaled_activation = self.min_max_scaling(log_scaled_activation, 0, np.max(cur_activation))
            plt.plot(cur_activation[0].flatten(), 'b.')
            plt.plot(log_scaled_activation[0].flatten(), 'r.')
            plt.hlines(np.max(cur_activation), 0, len(cur_activation[0].flatten()), color='gray')
            plt.show()

            # weight calculation
            new_w = w
            new_bias = bias

            neuron[2] = new_w
            if 'conv' in layer.name or layer.name == 'dense':
                neuron[3] = new_bias
            else: pass

        with open(self.filepath + self.filename + '_Converted_synapses.pkl', 'wb') as f:
            pickle.dump(self.synapses, f)

        print(f"Weight conversion DONE.<<<\n\n\n")
    

    def get_spikes(self, model, layer_in, layer_out, x):
        self.remove_keys(model, layer_in)
        weights = utils.weightDecompile(self.synapses)

        spikes = []
        for input_idx in range(len(x)):
            firing_rate = x[input_idx].flatten()
            for layer, neuron in model.items():
                firing_rate = np.dot(firing_rate, weights[layer])
                if 'conv' in layer:
                    s = 0
                    for oc_idx, oc in enumerate(neuron[4]):
                        firing_rate[s:oc] = firing_rate[s:oc] + neuron[3][oc_idx]
                        s = oc + 1
                elif 'dense' in layer:
                    firing_rate = firing_rate + neuron[3]
                else: pass
                neg_idx = np.where(firing_rate < 0)[0]
                firing_rate[neg_idx] = 0
                if layer == layer_out:
                    break
            spikes.append(firing_rate)
        
        return np.array(spikes)
    
    
    def min_max_scaling(self, data, new_min=0, new_max=1):

        current_min = np.min(data)
        current_max = np.max(data)

        scaled_data = [((x - current_min) / (current_max - current_min)) * (new_max - new_min) + new_min for x in data]

        return np.array(scaled_data)
    

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
    
