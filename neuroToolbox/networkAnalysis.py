import os, sys, configparser

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from tensorflow import keras
import numpy as np
import pickle, math
import matplotlib.pyplot as plt

import neuroToolbox.utils as utils

class Analysis:
    """
    Class for analysis converted SNN model.
    """

    def __init__(self, config):
        """
        Initialize the networkAnalysis instance.

        Args:
            x_norm (Numpy.ndarray): Input dataset for analysis.
            input_model_name (String): Input model name for SNN conversion.
            config (configparser.ConfigParser): Configuration settings for compiling.
        """
        self.config = config

        self.input_model_name = config["names"]["input_model"]
        self.input_model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"{self.input_model_name}.h5"))
        self.parsed_model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"parsed_{self.input_model_name}.h5"))
        
        self.fp_precision = config["conversion"]["fp_precision"]
        self.timesteps = config.getfloat('conversion', 'timesteps')
        self.timesteps = math.log10(self.timesteps)
        
        self.t_ref = config.getint('spiking_neuron', 'refractory') / 1000
            
        bias_flag = config["options"]["bias"]
        if bias_flag == 'False':
            self.bias_flag = False
        elif bias_flag == 'True':
            self.bias_flag = True
        else: print(f"ERROR !!")
        
        self.mac_operation = self.config["result"]["input_model_mac"]

        self.snn_filepath = os.path.join(self.config['paths']['models'], self.config['names']['snn_model'])

        with open(self.snn_filepath + '_Converted_neurons.pkl', 'rb') as f:
            self.neurons = pickle.load(f)
        with open(self.snn_filepath + '_Converted_synapses.pkl', 'rb') as f:
            self.synapses = pickle.load(f)
    

    def run(self, data_size):
        """
        Run SNN model.

        Args:
            data_size (_type_): _description_
        """
        print(f"Preparing for running converted snn.")
        print(f"Threshold : {self.v_th}")

        x_test_file = np.load(os.path.join(self.config["paths"]["dataset_path"], 'x_test.npz'))
        x_test = x_test_file['arr_0']
        x_test = x_test[:data_size]
        y_test_file = np.load(os.path.join(self.config["paths"]["dataset_path"], 'y_test.npz'))
        y_test = y_test_file['arr_0']
        y_test = y_test[:data_size]

        print(f"Input data length : {len(x_test)}")
        print(f"...\n")

        print(f"Loading synaptic weights ...\n")

        weights = {}
        for key in self.synapses.keys():
            if 'add' in key:
                continue
            weights[key] = self.synapses[key][3]

        score = 0
        self.syn_operation = 0
        for input_idx in range(len(x_test)):
            firing_rate = []
            for oc in range(x_test[input_idx].shape[-1]):
                firing_rate = np.concatenate((firing_rate, x_test[input_idx][:, :, oc].flatten()))
            shortcut = None
            for layer, synapse in self.synapses.items():
                # Calculate synaptic operations
                for neu_idx in range(len(firing_rate)):
                    if 'add' in layer:
                        continue
                    fan_out = len(np.where(weights[layer][neu_idx][:] > 0))
                    self.syn_operation += firing_rate[neu_idx] * 10**self.timesteps * fan_out
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

            print(f"Firing rate from output layer for #{input_idx+1} input")
            print(f"{firing_rate}")

            if np.argmax(y_test[input_idx]) == np.argmax(firing_rate):
                score += 1
            else: pass
            
            print(f"predict : {np.argmax(firing_rate)} | answer : {np.argmax(y_test[input_idx])}")
            print(f"Accuracy : {score/(input_idx+1)*100} %\n")

        self.accuracy = (score/len(x_test))*100
        print(f"______________________________________")
        print(f"Accuracy : {self.accuracy} %")
        print(f"Synaptic operation : {self.syn_operation}")
        print(f"Time steps : {10**self.timesteps} s")
        print(f"______________________________________\n")
        print(f"End running\n\n")


    def plot_compare(self):
        """
        Plot activation and expected firing rates for each layer.
        """
        os.makedirs(self.config['paths']['path_wd'] + '/fr_corr')
        
        activation_dir = os.path.join(self.config['paths']['path_wd'], f"parsed_model_activations")
        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']

        weights = {}
        for key in self.synapses.keys():
            if 'add' in key:
                continue
            weights[key] = self.synapses[key][3]

        firing_rate = []
        for i in range(x_norm.shape[0]):
            fr = []
            for oc in range(x_norm.shape[-1]):
                fr = np.concatenate((fr, x_norm[i, :, :, oc].flatten()))
            firing_rate.append(fr)
        
        output_layer = 0
        lambda_cnt = 0
        add_cnt = 0
        add_flag = 0

        for layer, synapse in self.synapses.items():
            output_layer += 1
            if 'add' in layer:
                add_flag = 1
                add_cnt += 1
                lambda_cnt += 1
                pass
            elif 'conv2d' in layer:
                add_flag = 0
                new_layer = layer.replace('conv2d', 'lambda')
                if '_identity' in new_layer:
                    new_layer = new_layer.replace('_identity','')
                elif '_conv' in new_layer:
                    new_layer = new_layer.replace('_conv','')
                
                if add_cnt == 0:
                    act_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{new_layer}.npz"))
                else:
                    base, old_number = new_layer.rsplit('_', 1)
                    new_layer = new_layer.replace(new_layer, f"lambda_{int(old_number)+add_cnt}")
                    act_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{new_layer}.npz"))

                lambda_cnt += 1
            elif 'dense' in layer:
                add_flag = 0
                if output_layer == len(self.synapses):
                    act_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{layer}.npz"))
                else:
                    new_layer = layer.replace(layer, f"lambda_{lambda_cnt}")
                    act_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{new_layer}.npz"))
                    lambda_cnt += 1
            else:
                add_flag = 0
                act_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{layer}.npz"))

            if add_flag == 0:
                acts = act_file['arr_0']            
                activations = utils.Input_Activation(acts, layer)
            
            fr = []

            if '_identity' in layer or 'add' in layer:
                if layer == 'conv2d_identity':
                    for idx in range(len(firing_rate)):
                        spikes = firing_rate[idx].flatten()
                        spikes = utils.neuron_model(spikes, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.bias_flag, self.timesteps)
                        fr.append(spikes)
                    firing_rate = np.array(fr)
                if 'add' in layer:
                    firing_rate = firing_rate + shortcut
                    shortcut = firing_rate
                    continue
                shortcut = firing_rate
            elif '_conv' in layer:
                for idx in range(len(shortcut)):
                    spikes = shortcut[idx].flatten()
                    spikes = utils.neuron_model(spikes, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.bias_flag, self.timesteps)
                    fr.append(spikes)
                shortcut = np.array(fr)
            else: 
                for idx in range(len(firing_rate)):
                    spikes = firing_rate[idx].flatten()
                    spikes = utils.neuron_model(spikes, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.bias_flag, self.timesteps)
                    fr.append(spikes)
                firing_rate = np.array(fr)

            plt.figure(figsize=(10, 10))
            if '_conv' in layer:
                plt.plot(activations, shortcut, 'o', markersize=2, color='red', linestyle='None')
            else:
                plt.plot(activations, firing_rate, 'o', markersize=2, color='red', linestyle='None')
            plt.title(f"DNN activation vs. Expected firing rates", fontsize=30)
            plt.xlabel(f"Activations in {layer}", fontsize=27)
            plt.ylabel(f"Expected firing rates in {layer} Hz", fontsize=27)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(True)
            # plt.yscale('symlog')
            # plt.ylim([1,200])
            plt.savefig(self.config['paths']['path_wd'] + '/fr_corr' + f"/{layer}", transparent=False)
            plt.show()
    
    
    def set_threshold(self, threshold):
        
        self.v_th = threshold
    
    def genResultFile(self):
        
        logfile = open(self.config['paths']['path_wd'] + '/LOG.txt', 'w')
        
        logfile.writelines(f"///////////////////////////////////////////////// \n")
        logfile.writelines(f"/// LOG file for experiment \n")
        logfile.writelines(f"/// \n")
        logfile.writelines(f"/// Experiment setup \n")
        logfile.writelines(f"\n")
        
        logfile.writelines(f"Input Model Name : {self.config['names']['input_model']} \n")
        logfile.writelines(f"Data set : {self.config['names']['dataset']} \n")
        logfile.writelines(f"Test data set size : {self.config['test']['data_size']} \n")
        logfile.writelines(f"\n")
        
        logfile.writelines(f"/// Model setup \n")
        if 'True' == self.config["options"]["bias"]:
            logfile.writelines(f"Bias : YES\n")
        elif 'False' == self.config["options"]["bias"]:
            logfile.writelines(f"Bias : NO\n")
        
        logfile.writelines(f"\n")
        
        logfile.writelines(f"/// Neuron setup \n")
        logfile.writelines(f"Neuron model : {self.config['conversion']['neuron']} neuron \n")
        if 'IF' == self.config["conversion"]["neuron"]:
            logfile.writelines(f"Refractory period : {self.config['spiking_neuron']['refractory']} ms \n")
            logfile.writelines(f"Threshold : {self.v_th} \n")
        
        logfile.writelines(f"\n")
        
        logfile.writelines(f"/// Conversion setup \n")
        logfile.writelines(f"Normalization : {self.config['conversion']['normalization']} \n")
        logfile.writelines(f"Optimizer : {self.config['conversion']['optimizer']} \n")
        logfile.writelines(f"Format : {self.config['conversion']['fp_precision']} \n")
        
        logfile.writelines(f"\n")
        
        logfile.writelines(f"/// RESULT \n")
        logfile.writelines(f"Input Model Accuracy : {float(self.config['result']['input_model_acc'])*100:.2f} %\n")
        logfile.writelines(f"Parsed Model Accuracy : {float(self.config['result']['parsed_model_acc'])*100:.2f} %\n\n")
        logfile.writelines(f"Accuracy for {self.config['names']['dataset']} {self.config['test']['data_size']} : {self.accuracy} % \n\n")
        logfile.writelines(f"MAC operation : {self.mac_operation} \n")
        logfile.writelines(f"Synaptic operation : {self.syn_operation} \n")
        logfile.writelines(f"\n")
        logfile.writelines(f"\n")
        logfile.writelines(f"///////////////////////////////////////////////// \n")
        
        logfile.close()
        
        