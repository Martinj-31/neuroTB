import os, sys, pickle, math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

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
        
        bias_flag = config["model"]["bias"]
        if bias_flag == 'False':
            self.bias_flag = False
        elif bias_flag == 'True':
            self.bias_flag = True
        else: print(f"ERROR !!")

        self.parsed_model = tf.keras.models.load_model(os.path.join(self.config["paths"]["models"], f"parsed_{self.input_model_name}.h5"))
        
        if 'LIF' == config["conversion"]["neuron"]:
            self.v_th = config.getfloat('LIF', 'threshold')
            self.t_ref = config.getint('LIF', 'refractory') / 1000
            self.w_mag = config.getfloat('LIF', 'w_mag')
            max_ratio = config.getfloat('LIF', 'max_ratio')
            # self.alpha = (self.v_th*max_ratio) / ((1-max_ratio)*self.t_ref)
        elif 'IF' == config["conversion"]["neuron"]:
            self.v_th = config.getint('IF', 'threshold')

        self.filepath = self.config['paths']['models']
        self.filename = self.config['names']['snn_model']


    def convertWeights(self):
        activation_dir = os.path.join(self.config['paths']['path_wd'], f"parsed_model_activations")

        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']

        print("\n\n######## Converting weight ########\n")
        
        weights = utils.weightDecompile(self.synapses)
        
        if 'LIF' == self.config["conversion"]["neuron"]:
            
            print(f">>Conversion for LIF neuron.\n")
            
            # def log_lif(x, beta, T, V):
            #     if x == 0:
            #         return 0
            #     return (1/V)*x**(np.log(beta/(beta*T + V)) / np.log(beta))
            
            # def lif(x, T, V):
            #     return x/(x*T + V)
            
            # def difference_of_integrals(beta):
            #     integral_A, _ = quad(lambda x: lif(x, self.t_ref, self.v_th) - log_lif(x, beta, self.t_ref, self.v_th), 0, beta)
            #     integral_B, _ = quad(lambda x: log_lif(x, beta, self.t_ref, self.v_th) - lif(x, self.t_ref, self.v_th), beta, self.alpha)
            #     return abs(integral_B - integral_A)
            
            # v_th = {}
            for layer in self.parsed_model.layers:
                
                if 'input' in layer.name:
                    firing_rate = x_norm
                    continue
                elif 'flatten' in layer.name:
                    continue
                else: pass
                
                print(f" Weight conversion for {layer.name} layer...\n")

                neuron = self.synapses[layer.name]
                
                if self.bias_flag:
                    if 'conv' in layer.name or 'dense' == layer.name:
                        ann_weights = [weights[layer.name], neuron[3]]
                        print(ann_weights[1])
                    else: ann_weights = [weights[layer.name]]
                else: ann_weights = [weights[layer.name]]
                
                # min_value = minimize_scalar(difference_of_integrals, bounds=(0, self.alpha), method='bounded')
                # beta = min_value.x
                
                # Weight normalization
                max_ann_weights = np.max(abs(ann_weights[0]))
                snn_weights = ann_weights[0] / max_ann_weights * self.w_mag
                # snn_weights = ann_weights[0]
                
                if self.bias_flag:
                    if 'conv' in layer.name or 'dense' == layer.name:
                        neuron[2] = snn_weights
                        neuron[3] = ann_weights[1]
                    else: neuron[2] = snn_weights
                else: neuron[2] = snn_weights
                
            with open(self.filepath + self.filename + '_Converted_synapses.pkl', 'wb') as f:
                pickle.dump(self.synapses, f)
                
        # Need to edit
        elif 'IF' == self.config["conversion"]["neuron"]:
            
            print(f">>Conversion for IF neuron.\n")
            
            batch_size = self.config.getint('conversion', 'batch_size')
            norm_facs = {self.parsed_model.layers[0].name: 1.0}
            
            for layer in self.parsed_model.layers:
                
                if len(layer.weights) == 0:
                    continue
                
                activations = self.get_activations_layer(self.parsed_model, layer, x_norm, batch_size, activation_dir)
                
                print(f"Maximum activation: {np.max(activations):.5f}.")
                nonzero_activations = activations[np.nonzero(activations)]
                del activations
                perc = self.config.getfloat('IF', 'percentile')
                
                cliped_max_activation = self.get_percentile_activation(nonzero_activations, perc)
                norm_facs[layer.name] = cliped_max_activation
                print(f"Cliped maximum activation: {norm_facs[layer.name]:.5f}.\n")
                i += 1

            for layer in self.parsed_model.layers:

                if len(layer.weights) == 0:
                    continue

                ann_weights, bias = layer.get_weights()
                    
                if layer.activation.__name__ == 'softmax':
                    norm_fac = 1.0
                    print(f"\n Using norm_factor: {norm_fac:.2f}.")
                else:
                    norm_fac = norm_facs[layer.name]
                    
                inbound = self.get_inbound_layers_with_params(layer)
                
                if len(inbound) == 0:
                    snn_weights = [ann_weights * norm_facs[self.parsed_model.layers[0].name] / norm_fac, bias / norm_fac]
                    print("\n +++++ input norm_facs +++++ \n ", norm_facs[self.parsed_model.layers[0].name])
                    print("  ---------------")
                    print(" ", norm_fac)
                elif len(inbound) == 1:
                    snn_weights = [ann_weights * norm_facs[inbound[0].name] / norm_fac, bias / norm_fac]
                    print("\n +++++ norm_facs +++++\n ", norm_facs[inbound[0].name])
                    print("  ---------------")
                    print(" ", norm_fac)
                else: snn_weights = [ann_weights, bias]
                
                layer.set_weights(snn_weights)
                
        else: pass
        
        print(f"\nWeight conversion DONE.<<<\n\n\n")
    

    def get_spikes(self, x, weights, layer_name):
        neuron = self.synapses[layer_name]
        input_spikes = x
        spikes = []
        for input_idx in range(len(input_spikes)):
            firing_rate = input_spikes[input_idx].flatten()
            firing_rate = np.dot(firing_rate, weights)
            if 'conv' in layer_name:
                s = 0
                for oc_idx, oc in enumerate(neuron[4]):
                    firing_rate[s:oc] = firing_rate[s:oc] + neuron[3][oc_idx]
                    x = oc + 1
            elif 'dense' in layer_name:
                firing_rate = firing_rate + neuron[3]
            else: pass
            neg_idx = np.where(firing_rate < 0)[0]
            firing_rate[neg_idx] = 0
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
    
    
    def get_activations_layer(self, layer_in, layer_out, x, batch_size=None, path=None):

        # Set to 10 if batch_size is not specified
        if batch_size is None:
            batch_size = 10
        
        # If input sample x and batch_size are divided and the remainder is nonzero
        if len(x) % batch_size != 0:
            # Delete the remainder divided by input sample list x
            x = x[: -(len(x) % batch_size)]
        
        print("Calculating activations of layer {}.".format(layer_out.name))
        # Calculate the activation of the corresponding layer neuron by putting 
        # an input sample in the predict function
        activations = tf.keras.models.Model(inputs=layer_in.input, 
                                            outputs=layer_out.output).predict(x, batch_size)
        
        # Save activations as an npz file
        print("Writing activations to disk.")
        if path is not None:
            np.savez_compressed(os.path.join(path, f'activation_{layer_out.name}.npz'), activations)


        return np.array(activations)
    
    
    def get_percentile_activation(self, activations, percentile):

        return np.percentile(activations, percentile) if activations.size else 1
    
    
    def get_inbound_layers_with_params(self, layer):
        
        inbound = layer
        prev_layer = None
        # Repeat when a layer with weight exists
        while True:
            inbound = self.get_inbound_layers(inbound)
            
            if len(inbound) == 1 and not isinstance(inbound[0], 
                                                    tf.keras.layers.BatchNormalization):
                inbound = inbound[0]
                if self.has_weights(inbound):
                    return [inbound]
                
            # If there is no layer information
            # In the case of input layer, the previous layer does not exist, 
            # so it is empty list return
            else:
                result = []
                for inb in inbound:
                    if isinstance(inb, tf.keras.layers.BatchNormalization):
                        prev_layer = self.get_inbound_layers_with_params(inb)[0]
                            
                    if self.has_weights(inb):
                        result.append(inb)
                        
                    else:
                        result += self.get_inbound_layers_with_params(inb)
                        
                if prev_layer is not None:
                    return [prev_layer]
                
                else:
                    return result
    
    
    def get_inbound_layers(self, layer):
        
        # Check the previous layer of that layer through _inbound_nodes
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        
        if not isinstance(inbound_layers, (list, tuple)):
            inbound_layers = [inbound_layers]
            
        return inbound_layers
        
    
    def has_weights(self, layer):
        
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            return False
        
        else:    
            return len(layer.weights)