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
    
    def __init__(self, parsed_model, config):
        """
        Initialize the Converter instance.

        Args:
            spike_model (tf.keras.Model): The compiled model for SNN from networkCompiler.
            config (configparser.ConfigParser): Configuration settings for weight conversion.
        """
        self.config = config
        self.input_model_name = config["names"]["input_model"]

        self.parsed_model = parsed_model
        # self.parsed_model = tf.keras.models.load_model(os.path.join(self.config["paths"]["models"], f"parsed_{self.input_model_name}.h5"))
        
        if 'LIF' == config["conversion"]["neuron"]:
            self.v_th = config.getint('LIF', 'threshold')
            self.t_ref = config.getint('LIF', 'refractory') / 1000
            alpha = config.getint('LIF', 'alpha')
        elif 'IF' == config["conversion"]["neuron"]:
            self.v_th = config.getint('IF', 'threshold')

        # self.min = 1/self.t_ref * lower_bound
        # self.max = 1/self.t_ref * upper_bound
        # self.min_bound = self.min / (self.v_th - self.min*self.t_ref)
        # self.max_bound = self.max / (self.v_th - self.max*self.t_ref)

        self.filepath = self.config['paths']['models']
        self.filename = self.config['names']['snn_model']


    def convertWeights(self):
        activation_dir = os.path.join(self.config['paths']['path_wd'], f"parsed_model_activations")

        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']

        print("\n\n######## Converting weight ########\n")
        
        # weights = utils.weightDecompile(self.synapses)
        
        if 'LIF' == self.config["conversion"]["neuron"]:
            
            print(f">>Conversion for LIF neuron.\n")
            
            for layer in self.parsed_model.layers:
                
                if 'input' in layer.name or 'flatten' in layer.name:
                    continue
                else: pass
                
                print(f" Weight conversion for {layer.name} layer...")
                
                weights = layer.get_weights()
            
        elif 'IF' == self.config["conversion"]["neuron"]:
            
            print(f">>Conversion for IF neuron.\n")
            
            batch_size = self.config.getint('conversion', 'batch_size')
            
            norm_facs = {self.parsed_model.layers[0].name: 1.0}
            max_weight_values = {}
            
            i = 0
            for layer in self.parsed_model.layers:
                
                if 'input' in layer.name or 'flatten' in layer.name:
                    continue
                else: pass
                
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
                print(layer.name)
                if 'input' in layer.name or 'flatten' in layer.name:
                    continue
                else: pass

                parameters = layer.get_weights()
                if 'pooling' in layer.name:
                    ann_weights = parameters
                else:
                    ann_weights = parameters[0]
                    bias = parameters[1]
                    
                if layer.activation.__name__ == 'softmax':
                    norm_fac = 1.0
                    print(f"\n Using norm_factor: {norm_fac:.2f}.")
                else:
                    norm_fac = norm_facs[layer.name]
                    
                inbound = utils.get_inbound_layers_with_params(layer)
                
                if len(inbound) == 0:
                    if 'pooling' in layer.name:
                        snn_weights = [ann_weights * norm_facs[self.parsed_model.layers[0].name] / norm_fac]
                    else:
                        snn_weights = [ann_weights * norm_facs[self.parsed_model.layers[0].name] / norm_fac, bias / norm_fac]
                    print("\n +++++ input norm_facs +++++ \n ", norm_facs[self.model.layers[0].name])
                    print("  ---------------")
                    print(" ", norm_fac)
                elif len(inbound) == 1:
                    if 'pooling' in layer.name:
                        snn_weights = [ann_weights * norm_facs[inbound[0].name] / norm_fac]
                    else:
                        snn_weights = [ann_weights * norm_facs[inbound[0].name] / norm_fac, bias / norm_fac]
                    print("\n +++++ norm_facs +++++\n ", norm_facs[inbound[0].name])
                    print("  ---------------")
                    print(" ", norm_fac)
                else: snn_weights = [ann_weights, bias]
                
                layer.set_weights(snn_weights)
                
        else: pass
        
        print(f"\nWeight conversion DONE.<<<\n\n\n")

        for layer in self.parsed_model.layers:

            if 'input' in layer.name or 'flatten' in layer.name:
                continue

            print(f" Weight conversion for {layer.name} layer...")

            neuron = self.synapses[layer.name]

            w = np.array(neuron[2])
            if 'conv' in layer.name or layer.name == 'dense':
                bias = neuron[3]
            else: pass
            
            if 'pooling' in layer.name:
                neuron[2] = weights[layer.name]
                continue
            else: pass

            inbound = utils.get_inbound_layers_with_params(layer)
            pre_layer = inbound[0]
            pre_activation_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{pre_layer.name}.npz"))
            cur_activation_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{layer.name}.npz"))
            
            pre_activation = pre_activation_file['arr_0']
            cur_activation = cur_activation_file['arr_0']

            # median = np.median(pre_activation)
            # scale = self.max_bound / median
            # weights[layer.name] = weights[layer.name] * scale
            
            # weight calculation

            new_bias = bias
            neuron[2] = weights[layer.name]
            if 'conv' in layer.name or layer.name == 'dense':
                neuron[3] = new_bias
            else: pass

        with open(self.filepath + self.filename + '_Converted_synapses.pkl', 'wb') as f:
            pickle.dump(self.synapses, f)

        print(f"\nWeight conversion DONE.<<<\n\n\n")
    

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