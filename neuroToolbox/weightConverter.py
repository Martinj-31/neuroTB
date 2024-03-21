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
        
        bias_flag = config["options"]["bias"]
        if bias_flag == 'False':
            self.bias_flag = False
        elif bias_flag == 'True':
            self.bias_flag = True
        else: print(f"ERROR !!")
        
        self.input_trans = config["options"]["input_trans"]
        
        self.scaling_precision = config.getfloat('conversion', 'scaling_precision')
        self.firing_range = config.getfloat('conversion', 'firing_range')

        self.parsed_model = tf.keras.models.load_model(os.path.join(self.config["paths"]["models"], f"parsed_{self.input_model_name}.h5"))
        
        self.v_th = config.getfloat('spiking_neuron', 'threshold')
        self.t_ref = config.getint('spiking_neuron', 'refractory') / 1000
        self.w_mag = config.getfloat('spiking_neuron', 'w_mag')
        
        self.percentile = config.getfloat('options', 'percentile')

        self.filepath = self.config['paths']['models']
        self.filename = self.config['names']['snn_model']


    def convertWeights(self):
        activation_dir = os.path.join(self.config['paths']['path_wd'], f"parsed_model_activations")

        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']

        print("\n\n######## Converting weight ########\n")
        
        weights = utils.weightDecompile(self.synapses)
        
        print(f">>Conversion for IF neuron.\n")
        
        if self.t_ref == 0:
            self.t_ref = 0.0000001
            print(f"###################################################")
            print(f"# Refractory period is 0.")
            print(f"# Replaced by a very small value that can be ignored.\n")
        
        if 'False' == self.config['options']['max_norm']:
            pre_layer_acts = x_norm
            linear_spikes = x_norm
            log_spikes = utils.log_transfer(pre_layer_acts, self.input_trans)
            
            for layer in self.parsed_model.layers:               
                if 'input' in layer.name:
                    firing_rate = x_norm
                    continue
                elif 'flatten' in layer.name:
                    continue
                else: pass
                
                print(f" Weight conversion for {layer.name} layer...\n")

                neuron = self.synapses[layer.name]
                
                # Prepare activations from previous layer and current layer.
                inbound = utils.get_inbound_layers_with_params(layer)
                pre_layer = inbound[0]
                pre_layer_activations_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{pre_layer.name}.npz"))
                layer_activations_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{layer.name}.npz"))
                pre_layer_activations = pre_layer_activations_file['arr_0']
                layer_activations = layer_activations_file['arr_0']
                
                if self.bias_flag:
                    if 'conv' in layer.name or 'dense' == layer.name:
                        ann_weights = [weights[layer.name], neuron[3]]
                        print(ann_weights[1])
                    else: ann_weights = [weights[layer.name]]
                else: ann_weights = [weights[layer.name]]

                max_ann_weights = np.max(abs(ann_weights[0]))
                snn_weights = ann_weights[0] / max_ann_weights * self.w_mag
                
                firing_rate = self.get_output_spikes(log_spikes, snn_weights, layer.name)
                nonzero_firing_rate = firing_rate[np.nonzero(firing_rate)]
                mean = np.average(nonzero_firing_rate)
                max_firing_rate = np.max(nonzero_firing_rate)
                target_firing_rate = mean / max_firing_rate * (1/self.t_ref)*self.firing_range
                
                print(f"Target firing rate : {target_firing_rate}")
                
                while True:
                    firing_rate = self.get_output_spikes(log_spikes, snn_weights, layer.name)
                    nonzero_firing_rate = firing_rate[np.nonzero(firing_rate)]
                    print(np.average(nonzero_firing_rate.flatten()))
                    
                    if target_firing_rate*0.99 <= np.average(nonzero_firing_rate.flatten()) <= target_firing_rate*1.01:
                        print(f"  ==> Average firing rate : {np.average(nonzero_firing_rate.flatten())}")
                        print(f"  ==> Scaling factor : {np.max(snn_weights) / (np.max(ann_weights[0] / max_ann_weights * self.w_mag))}\n")
                        break
                    elif np.average(nonzero_firing_rate.flatten()) <= target_firing_rate:
                        snn_weights *= 1 + self.scaling_precision
                    elif np.average(nonzero_firing_rate.flatten()) >= target_firing_rate:
                        snn_weights *= 1 - self.scaling_precision
                    else: pass
                
                # Compare firing rate with activations.
                np.set_printoptions(threshold=np.inf, linewidth=np.inf)
                previous_layer_activations = utils.Input_Activation(pre_layer_activations, pre_layer.name)
                current_layer_activations = utils.Input_Activation(layer_activations, layer.name)
                log_domain_input = utils.log_transfer(previous_layer_activations, self.input_trans)
                
                linear_activations = self.get_activations(previous_layer_activations, ann_weights[0], layer.name)
                log_activations = self.get_activations(log_domain_input, ann_weights[0], layer.name)
                
                linear_input_spikes = self.get_input_spikes(linear_spikes, snn_weights, layer.name)
                log_input_spikes = self.get_input_spikes(log_spikes, snn_weights, layer.name)
                
                linear_output_spikes = self.get_output_spikes(linear_spikes, snn_weights, layer.name)
                log_output_spikes = self.get_output_spikes(log_spikes, snn_weights, layer.name)
                
                plt.plot(linear_activations.flatten(), log_output_spikes.flatten(), 'o', markersize=2, color='blue', linestyle='None')
                plt.title(f"{layer.name}", fontsize=20)
                plt.xlabel(f"Log Activations", fontsize=15)
                plt.ylabel(f"Log Spikes", fontsize=15)
                plt.yscale('symlog')
                plt.show()
                
                # plt.plot(linear_activations.flatten(), current_layer_activations.flatten(), 'o', markersize=2, color='black', linestyle='None', label='ReLU')
                # plt.plot(linear_input_spikes.flatten(), linear_output_spikes.flatten(), 'o', markersize=2, color='blue', linestyle='None', label='IF neuron')
                # plt.plot(log_input_spikes.flatten(), log_output_spikes.flatten(), 'o', markersize=2, color='red', linestyle='None', label='Log IF neuron')
                # plt.yscale('symlog')
                # plt.title(f"{layer.name}", fontsize=20)
                # plt.xlabel(f"Input Activation or Firing rate", fontsize=15)
                # plt.ylabel(f"Output Activation for Firing rate", fontsize=15)
                # plt.legend(fontsize='xx-large')
                # plt.show()
                
                log_spikes = utils.log_transfer(current_layer_activations, self.input_trans)
                log_spikes = self.min_max_scaling(log_spikes, 0, 255)
                
                linear_spikes = linear_output_spikes
                # log_spikes = log_output_spikes
                
                if self.bias_flag:
                    if 'conv' in layer.name or 'dense' == layer.name:
                        neuron[2] = snn_weights
                        neuron[3] = ann_weights[1]
                    else: neuron[2] = snn_weights
                else: neuron[2] = snn_weights

            with open(self.filepath + self.filename + '_Converted_synapses.pkl', 'wb') as f:
                pickle.dump(self.synapses, f)
                
        # Need to edit
        elif 'True' == self.config['options']['max_norm']:
            
            print(f">>Max-norm option.\n")
            
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
    

    def get_output_spikes(self, x, weights, layer_name):
        synapse = self.synapses[layer_name]
        input_spikes = x
        spikes = []
        for input_idx in range(len(input_spikes)):
            firing_rate = input_spikes[input_idx].flatten()
            firing_rate = utils.neuron_model(firing_rate, weights, self.v_th, self.t_ref, layer_name, synapse, self.bias_flag)
            spikes.append(firing_rate)
        
        return np.array(spikes)
    
    
    def get_input_spikes(self, x, weights, layer_name):
        synapse = self.synapses[layer_name]
        input_spikes = x
        spikes = []
        for input_idx in range(len(input_spikes)):
            firing_rate = input_spikes[input_idx].flatten()
            firing_rate = utils.neuron_model(firing_rate, weights, self.v_th, 0, layer_name, synapse, self.bias_flag)
            spikes.append(firing_rate)
        
        return np.array(spikes)
    
    
    def get_activations(self, x, weights, layer_name):
        synapse = self.synapses[layer_name]
        input_activations = x
        acts = []
        for input_idx in range(len(input_activations)):
            activation = input_activations[input_idx].flatten()
            activation = utils.neuron_model(activation, weights, 1.0, 0, layer_name, synapse, self.bias_flag, False)
            acts.append(activation)
        
        return np.array(acts)
    
    
    def min_max_scaling(self, data, new_min=0, new_max=1):

        current_min = np.min(data)
        current_max = np.max(data)

        scaled_data = [((x - current_min) / (current_max - current_min)) * (new_max - new_min) + new_min for x in data]

        return np.array(scaled_data)
    
    
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