import os, sys, pickle, math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import neuroToolbox.utils as utils
from tqdm import tqdm

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
        
        self.mac_operation = config.getfloat('result', 'input_model_mac')
        
        self.trans_domain = config["options"]["trans_domain"]
        
        self.scaling_precision = config.getfloat('conversion', 'scaling_precision')
        self.firing_range = config.getfloat('conversion', 'firing_range')
        self.fp_precision = config["conversion"]["fp_precision"]
        self.epoch = config.getint('conversion', 'epoch')
        self.optimizer = config["conversion"]["optimizer"]

        self.parsed_model = tf.keras.models.load_model(os.path.join(self.config["paths"]["models"], f"parsed_{self.input_model_name}.h5"))
        
        self.t_ref = config.getint('spiking_neuron', 'refractory') / 1000
        self.w_mag = config.getfloat('spiking_neuron', 'w_mag')
        
        self.init_v_th = config.getfloat('spiking_neuron', 'threshold')
        self.v_th = {}
        for layer in self.parsed_model.layers:
            if 'input' in layer.name or 'flatten' in layer.name:
                continue
            else: self.v_th[layer.name] = self.init_v_th
        
        self.percentile = config.getfloat('options', 'percentile')

        self.filepath = self.config['paths']['models']
        self.filename = self.config['names']['snn_model']


    def convertWeights(self):
        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']
        
        data_size = int(10000 / self.config.getint('test', 'data_size'))
        data_size = 50
        x_test_file = np.load(os.path.join(self.config["paths"]["dataset_path"], 'x_test.npz'))
        x_test = x_test_file['arr_0']
        x_test = x_test[::data_size]
        y_test_file = np.load(os.path.join(self.config["paths"]["dataset_path"], 'y_test.npz'))
        y_test = y_test_file['arr_0']
        y_test = y_test[::data_size]

        print("\n\n######## Converting weight ########\n")
        
        weights = utils.weightDecompile(self.synapses)
        
        print(f">>Conversion for IF neuron.\n")
        
        if self.t_ref == 0:
            self.t_ref = 0.0000001
            print(f"###################################################")
            print(f"# Refractory period is 0.")
            print(f"# Replaced by a very small value that can be ignored.\n")
            
        if 'off' == self.optimizer:
            for layer in self.parsed_model.layers:
                if 'input' in layer.name or 'flatten' in layer.name:
                    continue
                else: pass
                
                neuron = self.synapses[layer.name]
                
                if self.bias_flag:
                    if 'conv' in layer.name or 'dense' == layer.name:
                        ann_weights = [weights[layer.name], neuron[3]]
                    else: ann_weights = [weights[layer.name]]
                else: ann_weights = [weights[layer.name]]
                
                snn_weights = ann_weights[0]
                snn_weights = utils.weightFormat(snn_weights, self.fp_precision)
                
                if self.bias_flag:
                    if 'conv' in layer.name or 'dense' == layer.name:
                        neuron[2] = snn_weights
                        neuron[3] = ann_weights[1]
                    else: neuron[2] = snn_weights
                else: neuron[2] = snn_weights
                
                self.v_th[layer.name] = 1.0
                
                with open(self.filepath + self.filename + '_Converted_synapses.pkl', 'wb') as f:
                    pickle.dump(self.synapses, f)
                    
            return
        
        for layer in self.parsed_model.layers:
            if 'input' in layer.name or 'flatten' in layer.name:
                continue
            else: pass
            
            print(f" Weight Normalization for {layer.name} layer...\n")
            
            neuron = self.synapses[layer.name]
            
            if self.bias_flag:
                if 'conv' in layer.name or 'dense' == layer.name:
                    ann_weights = [weights[layer.name], neuron[3]]
                else: ann_weights = [weights[layer.name]]
            else: ann_weights = [weights[layer.name]]
            
            # Weight Normalization
            max_ann_weights = np.max(abs(ann_weights[0]))
            snn_weights = ann_weights[0] / max_ann_weights * self.w_mag
            snn_weights = utils.weightFormat(snn_weights, self.fp_precision)
            
            # Weight adaptation
            if self.bias_flag:
                if 'conv' in layer.name or 'dense' == layer.name:
                    neuron[2] = snn_weights
                    neuron[3] = ann_weights[1]
                else: neuron[2] = snn_weights
            else: neuron[2] = snn_weights
            
            with open(self.filepath + self.filename + '_Converted_synapses.pkl', 'wb') as f:
                pickle.dump(self.synapses, f)
            print('')
        
        err = 0
        for i in range(self.epoch):
            print(f"Epoch {i+1}")
            print(f"Target firing rate range : {self.firing_range}")
            
            for layer in self.parsed_model.layers:
                if 'input' in layer.name:
                    input_data = utils.Input_Activation(x_norm, layer.name)
                    continue
                elif 'flatten' in layer.name:
                    continue
                else: pass
                
                print(f" Threshold balancing for {layer.name} layer...\n")
                
                neuron = self.synapses[layer.name]
                
                snn_weights = neuron[2]
                    
                while True:
                    output_spikes = self.get_output_spikes(input_data, snn_weights, layer.name)
                    
                    max_output_spikes = np.max(output_spikes)
                    print(max_output_spikes)
                    
                    if self.firing_range*0.95 <= max_output_spikes <= self.firing_range*1.05:
                        break
                    
                    if max_output_spikes < self.firing_range:
                        self.v_th[layer.name] -= 1
                    elif max_output_spikes > self.firing_range:
                        self.v_th[layer.name] += 1
                    else: pass
                
                input_data = self.get_output_spikes(input_data, snn_weights, layer.name)
                
            score, synOps = self.score(x_test, y_test)
            print(score, synOps)
            
            error = float(self.config['result']['input_model_acc'])*100 - score
            print(error, error - err)
            if error - err > 0:
                sign = 1
            elif error - err < 0:
                sign = -1
            err = error
            
            alpha = 0.3
            acc_error = score / float(self.config['result']['input_model_acc'])*100
            cal_error = synOps / self.mac_operation
            sub_error = acc_error*alpha + cal_error*(1-alpha)
            print(sub_error)
            
            self.firing_range += sign*1
            
        print(f"THreshold :{self.v_th}")

        print(f"\nWeight conversion DONE.<<<\n\n\n")
    

    def get_output_spikes(self, x, weights, layer_name):
        synapse = self.synapses[layer_name]
        input_spikes = x
        spikes = []
        for input_idx in range(len(input_spikes)):
            firing_rate = input_spikes[input_idx].flatten()
            firing_rate = utils.neuron_model(firing_rate, weights, self.v_th[layer_name], self.t_ref, layer_name, synapse, self.fp_precision, self.bias_flag)
            spikes.append(firing_rate)
        
        return np.array(spikes)
    
    
    def get_input_spikes(self, x, weights, layer_name):
        synapse = self.synapses[layer_name]
        input_spikes = x
        spikes = []
        for input_idx in range(len(input_spikes)):
            firing_rate = input_spikes[input_idx].flatten()
            firing_rate = utils.neuron_model(firing_rate, weights, self.v_th[layer_name], 0, layer_name, synapse, self.bias_flag, False)
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
    
    
    def score(self, x, y):
        x_test = x
        y_test = y
        
        weights = {}
        for key in self.synapses.keys():
            weights[key] = self.synapses[key][2]
        
        score = 0
        syn_operation = 0
        for input_idx in tqdm(range(len(x_test)), ncols=70, ascii=' ='):
            firing_rate = x_test[input_idx].flatten()
            for layer, synapse in self.synapses.items():
                for neu_idx in range(len(firing_rate)):
                    fan_out = len(np.where(weights[layer][neu_idx][:] > 0))
                    syn_operation += firing_rate[neu_idx] * fan_out
                firing_rate = utils.neuron_model(firing_rate, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.fp_precision, self.bias_flag)

            if np.argmax(y_test[input_idx]) == np.argmax(firing_rate):
                score += 1
            else: pass
        score = (score/len(x_test))*100
        
        return score, syn_operation
    
    
    def get_threshold(self):
        
        return self.v_th
    
    
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