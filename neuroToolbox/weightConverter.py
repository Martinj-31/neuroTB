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
        
        self.firing_range = config.getfloat('conversion', 'firing_range')
        self.fp_precision = config["conversion"]["fp_precision"]
        self.normalization = config["conversion"]["normalization"]
        self.optimizer = config["conversion"]["optimizer"]
        self.timesteps = config.getfloat('conversion', 'timesteps')
        self.timesteps = math.log10(self.timesteps)
        
        self.error_list = []
        self.synops_error_list = []
        self.acc_error_list = []
        self.firing_range_list = []

        self.parsed_model = tf.keras.models.load_model(os.path.join(self.config["paths"]["models"], f"parsed_{self.input_model_name}.h5"))
        self.changename_layers()
        data_size = self.config.getint('test', 'data_size')
        
        x_test_file = np.load(os.path.join(self.config["paths"]["dataset_path"], 'x_test.npz'))
        x_test = x_test_file['arr_0']
        self.x_test = x_test[:1000]
        
        y_test_file = np.load(os.path.join(self.config["paths"]["dataset_path"], 'y_test.npz'))
        y_test = y_test_file['arr_0']
        self.y_test = y_test[:1000]
        
        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        self.x_norm = x_norm_file['arr_0']
        
        self.t_ref = config.getint('spiking_neuron', 'refractory') / 1000
        self.w_mag = config.getfloat('spiking_neuron', 'w_mag')
        self.init_v_th = config.getfloat('spiking_neuron', 'threshold')
        
        self.v_th = {}
        for layer in self.parsed_model.layers:
            if 'input' in layer.name or 'flatten' in layer.name or 'add' in layer.name or 'lambda' in layer.name:
                continue
            else: self.v_th[layer.name] = self.init_v_th

        self.filepath = self.config['paths']['models']
        self.filename = self.config['names']['snn_model']


    def convertWeights(self):
        """
        Convert ANN weights to SNN weights and set threshold.
        """

        print("\n\n######## Converting weight ########\n")
        
        weights = utils.weightDecompile(self.synapses)
        
        print(f">>Conversion for IF neuron.\n")
        
        if self.t_ref == 0:
            self.t_ref = 0.0000001
            print(f"###################################################")
            print(f"# Refractory period is 0.")
            print(f"# Replaced by a very small value that can be ignored.\n")
        
        weight_list = {}
        th_list = {}
        bias_weight_list = {}
        th_factor = 1.0
        bias_weight_factor = 1.0

        for layer in self.parsed_model.layers:
            if 'input' in layer.name or 'flatten' in layer.name or 'add' in layer.name or 'lambda' in layer.name or 'activation' in layer.name or 're_lu' in layer.name:
                continue
            else: pass
            
            neuron = self.synapses[layer.name]
            print(f"Convert weight and threshold for layer {layer.name}\n")
            if self.bias_flag:
                if 'conv' in layer.name or 'dense' in layer.name:
                    ann_weights = [weights[layer.name], neuron[4]]
                else: ann_weights = [weights[layer.name]]
            else: ann_weights = [weights[layer.name]]
            
            if 'on' == self.normalization:
                max_ann_weights = np.max(abs(ann_weights[0]))
                weight_factor = self.w_mag
                
                #weight_factor = self.w_mag/max_ann_weights
                snn_weights = ann_weights[0] * weight_factor
                weight_list[layer.name] = weight_factor 
                if self.bias_flag:
                    th_list[layer.name] = th_factor
                    th_factor = th_factor * self.v_th[layer.name]
                    bias_weight_factor = bias_weight_factor*weight_list[layer.name]
                    bias_weight_list[layer.name] = bias_weight_factor

                if 'conv' in layer.name or 'dense' in layer.name:
                    if self.bias_flag:
                        snn_bias = ann_weights[1]/th_list[layer.name]*bias_weight_list[layer.name]

            else:
                snn_weights = ann_weights[0]
                if 'conv' in layer.name or 'dense' in layer.name:
                    if self.bias_flag:
                        snn_bias = ann_weights[1]
            
            snn_weights = utils.weightFormat(snn_weights, self.fp_precision)
            
            if self.bias_flag:
                if 'conv' in layer.name or 'dense' in layer.name:
                    neuron[3] = snn_weights
                    neuron[4] = snn_bias
                else: neuron[3] = snn_weights
            else: neuron[3] = snn_weights
            
            with open(self.filepath + self.filename + '_Converted_synapses.pkl', 'wb') as f:
                pickle.dump(self.synapses, f)
                    
        if self.optimizer == 'off':
            return
        
        print(f" Threshold balancing for each layer...\n")
        
        score_list = []
        synOps_list = []
        threshold_list = []
        layer_threshold = {}
        for l in self.parsed_model.layers:
            if 'input' in layer.name or 'flatten' in layer.name or 'add' in layer.name or 'lambda' in layer.name or 'activation' in layer.name:
                continue
            layer_threshold[l.name] = []
            
        target_score = float(self.config['result']['input_model_acc'])*100
        for target_firing_rate in range(int(self.firing_range), 0, -1):
            
            print(f"Target firing rate range : {target_firing_rate}\n")
            shortcut = None
            
            for layer in self.parsed_model.layers:
                if 'input' in layer.name:
                    input_data = utils.Input_Activation(self.x_norm, layer.name)
                    continue
                elif 'flatten' in layer.name or 'lambda' in layer.name or 'activation' in layer.name:
                    continue
                else: pass

                first_layer_flag = 0

                if '_identity' in layer.name or 'add' in layer.name:
                    if layer.name == 'conv2d_identity':
                        first_layer_flag = 1
                        
                    if 'add' in layer.name:
                        input_data = input_data + shortcut
                        shortcut = input_data
                        continue

                neuron = self.synapses[layer.name]
                snn_weights = neuron[3]
                if 'conv' in layer.name or 'dense' in layer.name:
                    if self.bias_flag:
                        snn_bias = neuron[4]

                if '_conv' in layer.name:
                    shortcut = self.get_output_spikes(shortcut, snn_weights, layer.name)
                    continue
                
                cnt = 0
                while True:
                    output_spikes = self.get_output_spikes(input_data, snn_weights, layer.name)
                    nonzero_output_spikes = output_spikes[np.nonzero(output_spikes)]
                    avg_output_spikes = np.mean(nonzero_output_spikes)
                    
                    if target_firing_rate*0.95 <= avg_output_spikes <= target_firing_rate*1.05:
                        print(f"==> Average firing rate of {layer.name}: {avg_output_spikes}")
                        break
                    elif cnt == 50:
                        print(f"==> Average firing rate of {layer.name}: {avg_output_spikes}")
                        print(f"Skip point occurs")
                        break
                    
                    if abs(avg_output_spikes - target_firing_rate) > 10:
                        scaling_factor = 4
                    else: scaling_factor = 1
                    
                    if avg_output_spikes < target_firing_rate:
                        self.v_th[layer.name] -= 1 * scaling_factor
                    elif avg_output_spikes > target_firing_rate:
                        self.v_th[layer.name] += 1 * scaling_factor
                    else: pass
                    
                    if 'conv' in layer.name or 'dense' in layer.name:
                        if self.bias_flag:
                            # Load previous layer threshold
                            snn_bias = snn_bias # 코드 수정
                    
                    cnt += 1
                
                print(f"Threshold: {self.v_th[layer.name]}")
                
                # layer_threshold for threshold plot
                layer_threshold[layer.name].append(self.v_th[layer.name])
                
                input_data = self.get_output_spikes(input_data, snn_weights, layer.name)

                if first_layer_flag == 1:
                    shortcut = input_data
                    
            score, synOps = self.score(self.x_test, self.y_test)
            print(self.v_th)
            print(score, synOps, "\n")
            score_list.append(score)
            synOps_list.append(synOps)
            threshold_list.append(self.v_th.copy())
        
        cloesst_idx = np.argmin(np.abs(np.array(score_list) - target_score))
        print(f"Best score index : \n{cloesst_idx}\n")
        
        print(f"Score list : \n{score_list}\n")
        print(f"SunOps list : \n{synOps_list}\n")
        print(f"Layer threshold : \n{layer_threshold}")
        for layer in self.parsed_model.layers:
            if 'input' in layer.name or 'flatten' in layer.name:
                continue
            plt.plot(np.arange(len(layer_threshold[layer.name])), layer_threshold[layer.name], marker='o', markersize=3, label=f"{layer.name}")
        plt.title(f"Threshold trend", fontsize=20)
        plt.xlabel(f"average firing rates", fontsize=15)
        plt.ylabel(f"threshold", fontsize=15)
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(0, 20, 2), np.arange(20, 0, -2))
        plt.show()
        
        plt.plot(score_list, synOps_list, marker='x', markersize=10, linestyle='None')
        plt.title(f"Accuracy", fontsize=20)
        plt.xlabel(f"Accuracy (%)", fontsize=15)
        plt.ylabel(f"Input data size", fontsize=15)
        plt.xlim([0, 100])
        plt.show()
        
        self.v_th = threshold_list[cloesst_idx]
        
        '''
        pre_error = 0
        direction = -1
        for epoch in range(self.epoch):
            print(f"Epoch {epoch+1}")
            print(f"Target firing rate range : {self.firing_range}\n")
            
            for layer in self.parsed_model.layers:
                if 'input' in layer.name:
                    input_data = utils.Input_Activation(self.x_norm, layer.name)
                    continue
                elif 'flatten' in layer.name:
                    continue
                else: pass
                
                neuron = self.synapses[layer.name]
                
                snn_weights = neuron[2]
                
                cnt = 0
                while True:
                    output_spikes = self.get_output_spikes(input_data, snn_weights, layer.name)
                    nonzero_output_spikes = output_spikes[np.nonzero(output_spikes)]
                    avg_output_spikes = np.mean(nonzero_output_spikes)
                    
                    if self.firing_range*0.95 <= avg_output_spikes <= self.firing_range*1.05:
                        break
                    elif cnt == 50:
                        break
                    
                    if avg_output_spikes < self.firing_range:
                        self.v_th[layer.name] -= 1
                    elif avg_output_spikes > self.firing_range:
                        self.v_th[layer.name] += 1
                    else: pass
                    
                    cnt += 1
                
                input_data = self.get_output_spikes(input_data, snn_weights, layer.name)
                
            score, synOps = self.score(self.x_test, self.y_test)

            acc_error = (float(self.config['result']['input_model_acc'])*100) - score
            ops_error = synOps / self.mac_operation
            error = ops_error*self.loss_alpha + acc_error*(1-self.loss_alpha)
            
            print(f"Scaled threshold : \n{self.v_th}\n")
            print(f"{'Error table':<13} | {'SynOps error':<20} | {'Perf error':<30}   ")
            print(f"{'-'*65}")
            print(f"{'Pure':<13} | {ops_error:<20} | {acc_error:<30}")
            print(f"{'Alpha scaled':<13} | {ops_error*self.loss_alpha:<20} | {acc_error*(1-self.loss_alpha):<30}")
            print(f"{'-'*65}\n")
            
            if epoch == 0:
                direction = -1
            else:
                if error > pre_error:
                    direction *= -1
                elif error == pre_error:
                    direction = -1
                    
            print(f"Error change")
            print(f"{pre_error} --> {error}\n\n")

            self.error_list.append(error)
            self.synops_error_list.append(ops_error*self.loss_alpha)
            self.acc_error_list.append(acc_error*(1-self.loss_alpha))
            
            pre_error = error
            
            self.firing_range += direction * self.scaling_step
            if self.firing_range == 0:
                self.firing_range = 1.0
            
            self.firing_range_list.append(self.firing_range)
            '''
        print(f"THreshold :{self.v_th}")

        print(f"\nWeight conversion DONE.<<<\n\n\n")
    

    def get_output_spikes(self, x, weights, layer_name):
        """
        Get output spikes for a layer.

        Args:
            x (_type_): _description_
            weights (_type_): _description_
            layer_name (_type_): _description_

        Returns:
            _type_: _description_
        """
        synapse = self.synapses[layer_name]
        input_spikes = x
        spikes = []
        for input_idx in range(len(input_spikes)):
            firing_rate = input_spikes[input_idx].flatten()
            firing_rate = utils.neuron_model(firing_rate, weights, self.v_th[layer_name], self.t_ref, layer_name, synapse, self.bias_flag, self.timesteps)
            spikes.append(firing_rate)
        
        return np.array(spikes)
    
    
    def get_input_spikes(self, x, weights, layer_name):
        """
        Get weighted sum.

        Args:
            x (_type_): _description_
            weights (_type_): _description_
            layer_name (_type_): _description_

        Returns:
            _type_: _description_
        """
        synapse = self.synapses[layer_name]
        input_spikes = x
        spikes = []
        for input_idx in range(len(input_spikes)):
            firing_rate = input_spikes[input_idx].flatten()
            firing_rate = utils.neuron_model(firing_rate, weights, self.v_th[layer_name], 0, layer_name, synapse, self.bias_flag, self.timesteps)
            spikes.append(firing_rate)
        
        return np.array(spikes)
    
    
    def get_activations(self, x, weights, layer_name):
        """
        Get expected activations with output spikes.

        Args:
            x (_type_): _description_
            weights (_type_): _description_
            layer_name (_type_): _description_

        Returns:
            _type_: _description_
        """
        synapse = self.synapses[layer_name]
        input_activations = x
        acts = []
        for input_idx in range(len(input_activations)):
            activation = input_activations[input_idx].flatten()
            activation = utils.neuron_model(activation, weights, 1.0, 0, layer_name, synapse, self.bias_flag, self.timesteps)
            acts.append(activation)
        
        return np.array(acts)
    
    
    def score(self, x, y):
        """
        Measure score for SNN with current parameters.

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_test = np.floor(x)
        y_test = y

        weights = {}
        for key in self.synapses.keys():
            if 'add' in key:
                continue
            weights[key] = self.synapses[key][3]
        
        score = 0
        syn_operation = 0
        for input_idx in tqdm(range(len(x_test)), ncols=70, ascii=' ='):
            firing_rate = x_test[input_idx].flatten()
            shortcut = None
            for layer, synapse in self.synapses.items():
                for neu_idx in range(len(firing_rate)):
                    if 'add' in layer:
                        continue
                    fan_out = len(np.where(weights[layer][neu_idx][:] > 0))
                    syn_operation += firing_rate[neu_idx] * 10**self.timesteps * fan_out
                if '_identity' in layer or 'add' in layer:
                    if layer == 'conv2d_identity':
                        firing_rate = utils.neuron_model(firing_rate, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.bias_flag, self.timesteps)
                    if 'add' in layer:
                        firing_rate = firing_rate + shortcut
                    shortcut = firing_rate
                elif '_conv' in layer:
                    shortcut = utils.neuron_model(shortcut, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.bias_flag, self.timesteps)
                else:
                    firing_rate = utils.neuron_model(firing_rate, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.bias_flag, self.timesteps)

            if np.argmax(y_test[input_idx]) == np.argmax(firing_rate):
                score += 1
            else: pass
        score = (score/len(x_test))*100
        
        return score, syn_operation
    
    
    def get_threshold(self):
        
        return self.v_th


    def changename_layers(self):
        neurons_keys = list(self.neurons.keys())
        i = 0
        print(f"Change name of layers...")
        print(f"_________________________________________________________________")
        for layer in self.parsed_model.layers:
            old_name = layer.name
            
            if '_flatten' in old_name or 'lambda' in old_name:
                continue
            if i < len(neurons_keys):
                new_name = neurons_keys[i]
                if '_identity' in new_name or '_conv' in new_name:
                    layer._name = new_name
                    print(f"Update {old_name} to {new_name}")
                else:
                    layer._name = old_name
                i += 1
        
        print(f"_________________________________________________________________")