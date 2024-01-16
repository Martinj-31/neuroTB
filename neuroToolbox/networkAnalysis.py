import os, sys, configparser

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt

import neuroToolbox.utils as utils

class Analysis:
    """
    Class for analysis converted SNN model.
    """

    def __init__(self, input_model_name, config):
        """
        Initialize the networkAnalysis instance.

        Args:
            x_norm (Numpy.ndarray): Input dataset for analysis.
            input_model_name (String): Input model name for SNN conversion.
            config (configparser.ConfigParser): Configuration settings for compiling.
        """
        self.config = config

        self.input_model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"{input_model_name}.h5"))
        self.parsed_model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"parsed_{input_model_name}.h5"))
        self.input_model_name = input_model_name

        self.v_th = self.config.getint('conversion', 'threshold')
        self.t_ref = self.config.getint('conversion', 'refractory') / 1000

        self.snn_filepath = os.path.join(self.config['paths']['models'], self.config['names']['snn_model'])
        os.makedirs(self.config['paths']['path_wd'] + '/snn_model_firing_rates')
        os.makedirs(self.config['paths']['path_wd'] + '/map_corr')
        os.makedirs(self.config['paths']['path_wd'] + '/fr_corr')

        with open(self.snn_filepath + '_Converted_neurons.pkl', 'rb') as f:
            self.neurons = pickle.load(f)
        with open(self.snn_filepath + '_Converted_synapses.pkl', 'rb') as f:
            self.synapses = pickle.load(f)

        self.data_size = 0
    

    def run(self, data_size):
        print(f"Preparing for running converted snn.")

        x_test_file = np.load(os.path.join(self.config["paths"]["dataset_path"], 'x_test.npz'))
        x_test = x_test_file['arr_0']
        x_test = x_test[::data_size]
        y_test_file = np.load(os.path.join(self.config["paths"]["dataset_path"], 'y_test.npz'))
        y_test = y_test_file['arr_0']
        y_test = y_test[::data_size]

        self.data_size = len(x_test)
        
        self.input_firing_rates = {}
        self.output_firing_rates = {}
        for i in range(len(x_test)):
            self.input_firing_rates[f"input {i+1}"] = {}
            self.output_firing_rates[f"input {i+1}"] = {}
        for i in range(len(x_test)):
            for layer in self.synapses.keys():
                self.input_firing_rates[f"input {i+1}"][layer] = []
                self.output_firing_rates[f"input {i+1}"][layer] = []

        print(f"Input data length : {len(x_test)}")
        print(f"...\n")

        print(f"Loading synaptic weights ...\n")
        weights = utils.weightDecompile(self.synapses)

        fr_dist = {}
        w_dict = {}
        w_cal = {}
        for layer in self.synapses.keys():
            fr_dist[layer] = []
            w_dict[layer] = []
            w_cal[layer] = []
        
        score = 0
        syn_operation = 0
        for input_idx in range(len(x_test)):
            firing_rate = x_test[input_idx].flatten()
            for layer, synapse in self.synapses.items():
                # Calculate synaptic operations
                for neu_idx in range(len(firing_rate)):
                    fan_out = len(np.where(weights[layer][neu_idx][:] > 0)[0])
                    syn_operation += firing_rate[neu_idx] * fan_out
                firing_rate = np.dot(firing_rate, weights[layer])
                firing_rate = self.add_bias(firing_rate, layer, synapse)
                neg_idx = np.where(firing_rate < 0)[0]
                firing_rate[neg_idx] = 0
                self.input_firing_rates[f"input {input_idx+1}"][layer] = np.concatenate((self.input_firing_rates[f"input {input_idx+1}"][layer], firing_rate))
                firing_rate = np.floor(firing_rate / (firing_rate*self.t_ref + self.v_th))
                fr_dist[layer] = np.concatenate((fr_dist[layer], firing_rate))
                self.output_firing_rates[f"input {input_idx+1}"][layer] = np.concatenate((self.output_firing_rates[f"input {input_idx+1}"][layer], firing_rate))
            print(f"Firing rate from output layer for #{input_idx+1} input")
            print(f"{firing_rate}\n")

            if np.argmax(y_test[input_idx]) == np.argmax(firing_rate):
                score += 1
            else: pass

        print(f"______________________________________")
        print(f"Accuracy : {(score/len(x_test))*100} %")
        print(f"Synaptic operation : {syn_operation}")
        print(f"______________________________________\n")
        print(f"End running\n\n")

        # for l in fr_dist.keys():
        #     max_fr = np.max(fr_dist[l])
        #     x_value = np.array([])
        #     for f in range(int(max_fr)):
        #         w_computed = f * w_dict[l]
        #         neg_idx = np.where(w_computed < 0)[0]
        #         w_computed[neg_idx] = 0
        #         x_value = np.concatenate((np.ones(len(w_computed))*f, x_value))
        #         w_cal[l] = np.concatenate((list(w_computed), w_cal[l])) 

        #     plt.figure(figsize=(8, 6))
        #     plt.plot(x_value, w_cal[l], 'b.')
        #     plt.title(f"Firing rate * weight of {l} layer", fontsize=30)
        #     plt.xlabel(f"Firing rate from 0 to maximum", fontsize=27)
        #     plt.ylabel(f"Firing rates * weight", fontsize=27)
        #     plt.yticks(fontsize=20)
        #     plt.grid(True)
        #     plt.savefig(self.config['paths']['path_wd'] + '/fr_distribution' + f"/{l}")
        #     plt.show()

    
    def compareAct(self, input_model_name):

        print(f"##### Comparing activations between input model and parsed model. #####")

        input_model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"{input_model_name}.h5"))
        parsed_model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"parsed_{input_model_name}.h5"))
        
        input_model_activation_dir = os.path.join(self.config['paths']['path_wd'], 'input_model_activations')
        corr_dir = os.path.join(self.config['paths']['path_wd'], 'acts_corr')
        
        os.makedirs(corr_dir, exist_ok=True)
        input_idx = 0
        for input_layer in input_model.layers:

            print(f"Comparing {input_layer.name} layer...")

            if 'input' in input_layer.name:
                input_idx += 1
                continue
            else:
                input_act_file = np.load(os.path.join(input_model_activation_dir, f"input_model_activation_{input_layer.name}.npz"))
                input_act = input_act_file['arr_0']
            for parsed_layer in parsed_model.layers:
                if 'input' in parsed_layer.name:
                    continue
                else:
                    if input_layer.output_shape != parsed_layer.output_shape:
                        continue
                    else:
                        if 'batch' in input_layer.name:
                            if (parsed_layer.name == input_model.layers[input_idx-1].name):
                                print(f'Current parsed layer name : {parsed_layer.name}')
                                loaded_activation = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-2].name}.npz"))['arr_0']
                            else : continue
                        else:
                            if input_layer.name == parsed_layer.name :
                                print(f'Current parsed layer name : {parsed_layer.name}')
                                loaded_activation = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-1].name}.npz"))['arr_0']
                            else : continue

                        parsed_act = keras.models.Model(inputs=parsed_layer.input, outputs=parsed_layer.output).predict(loaded_activation)
                        
                        plt.figure(figsize=(10, 10))
                        plt.scatter(input_act, parsed_act, color='b', marker='o', s=10)
                        plt.title(f"Input vs. Parsed activation correlation", fontsize=30)
                        plt.xlabel(f'input_model : "{input_layer.name}" Activation', fontsize=27)
                        plt.ylabel(f'parsed_model : "{parsed_layer.name}" Activation', fontsize=27)
                        plt.xticks(fontsize=20)
                        plt.yticks(fontsize=20)
                        plt.grid(True)
                        plt.savefig(corr_dir + f"/{input_layer.name}")
                        plt.show()
            input_idx += 1
            print('')


    def evalNetwork(self, model_name='input'):
        if 'input' == model_name:
            model = self.input_model
        elif 'parsed' == model_name:
            model = self.parsed_model
        else: pass
        activation_dir = os.path.join(self.config['paths']['path_wd'], f"{model_name}_model_activations")

        weights = utils.weightDecompile(self.synapses)

        input_idx = 0
        for input_layer in model.layers:
            if 'input' in input_layer.name:
                input_idx += 1
                continue
            elif 'flatten' in input_layer.name:
                input_idx += 1
                continue
            else: pass

            for snn_layer_idx, snn_layer in enumerate(self.synapses.items()):
                if np.prod(input_layer.output_shape[1:]) != list(self.neurons.values())[snn_layer_idx+1]:
                    continue
                else:
                    if 'batch' in input_layer.name:
                        if snn_layer[0] == model.layers[input_idx-1].name:
                            snn_layer_name = model.layers[input_idx-2].name
                        else: continue
                    else:
                        if input_layer.name == snn_layer[0]:
                            snn_layer_name = model.layers[input_idx-1].name
                            if 'flatten' in snn_layer_name:
                                snn_layer_name = model.layers[input_idx-2].name
                            else: pass
                        else: continue
                    
                    input_act_file = np.load(os.path.join(activation_dir, f"{model_name}_model_activation_{snn_layer_name}.npz"))
                    input_act = input_act_file['arr_0']

                    if 'input' in snn_layer_name:
                        input_acts = utils.Input_Image2D(input_act)
                    elif 'conv' in snn_layer_name:
                        input_acts = utils.Input_Conv2D(input_act)
                    elif 'batch' in snn_layer_name:
                        input_acts = utils.Input_Conv2D(input_act)
                    elif 'pooling' in snn_layer_name:
                        input_acts = utils.Input_Pooling(input_act)
                    elif 'dense' in snn_layer_name:
                        input_acts = utils.Input_Dense(input_act)
                    else: pass

                    snn_fr = []
                    for idx in range(len(input_acts)):
                        firing_rate = input_acts[idx].flatten()
                        firing_rate = np.dot(firing_rate, weights[snn_layer[0]])
                        firing_rate = self.add_bias(firing_rate, snn_layer[0], snn_layer[1])
                        neg_idx = np.where(firing_rate < 0)[0]
                        firing_rate[neg_idx] = 0
                        firing_rate = np.floor(firing_rate / (firing_rate*self.t_ref + self.v_th))
                        snn_fr = np.concatenate((snn_fr, firing_rate))

                    loaded_act_file = np.load(os.path.join(activation_dir, f"{model_name}_model_activation_{input_layer.name}.npz"))
                    loaded_act = loaded_act_file['arr_0']

                    if 'conv' in snn_layer[0]:
                        acts = utils.Flattener_Conv2D(loaded_act)
                    elif 'pooling' in snn_layer[0]:
                        acts = utils.Flattener_Pooling(loaded_act)
                    elif 'dense' in snn_layer[0]:
                        acts = utils.Flattener_Dense(loaded_act)
                    else: pass

                    plt.figure(figsize=(10, 10))
                    plt.scatter(acts, snn_fr, color='r', marker='o', s=10)
                    plt.title(f"DNN activation vs. SNN firing rates", fontsize=30)
                    plt.xlabel(f"Activations in {input_layer.name}", fontsize=27)
                    plt.ylabel(f"Firing rates in {snn_layer[0]}", fontsize=27)
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)
                    plt.grid(True)
                    plt.savefig(self.config['paths']['path_wd'] + '/fr_corr' + f"/{snn_layer[0]}")
                    plt.show()
            input_idx += 1


    def IOcurve(self):
        input_firing_rates = {}
        output_firing_rates = {}
        for layer in self.synapses.keys():
            input_firing_rates[layer] = []
            output_firing_rates[layer] = []
        for input_idx in range(self.data_size):
            for layer in self.synapses.keys():
                input_firing_rates[layer] = np.concatenate((input_firing_rates[layer], self.input_firing_rates[f"input {input_idx+1}"][layer]))
                output_firing_rates[layer] = np.concatenate((output_firing_rates[layer], self.output_firing_rates[f"input {input_idx+1}"][layer]))
        
        for layer in self.synapses.keys():
            plt.plot(input_firing_rates[layer], output_firing_rates[layer], 'b.')
            plt.title(f"IO curve {layer} layer")
            plt.show()


    def IOcurve_2(self):
        v_th = 1
        t_ref = 0.005
        lower_bound = 1/t_ref*0.1
        upper_bound = 1/t_ref*0.9
        min = lower_bound / (v_th - lower_bound*t_ref)
        max = upper_bound / (v_th - upper_bound*t_ref)
        print(min, max)

        num_neuron = 200
        input_firing_rate = np.arange(num_neuron)

        for _ in range(5):
            weights = np.identity(num_neuron)
            firing_rate = np.dot(input_firing_rate, weights)
            plt.plot(input_firing_rate, firing_rate, 'r.')
            current_min = np.min(firing_rate)
            current_max = np.max(firing_rate)
            scaled_firing_rate = np.array([((x - current_min) / (current_max - current_min)) * (max - min) + min for x in firing_rate])
            scaled_firing_rate_shifted = scaled_firing_rate - min

            normalization_factor = np.max(scaled_firing_rate_shifted) / np.max(firing_rate)
            print(normalization_factor)
            new_weights = weights * normalization_factor

            output_firing_rate = np.dot(input_firing_rate, new_weights)
            output_firing_rate = output_firing_rate / (output_firing_rate*t_ref + v_th)

            input_firing_rate = firing_rate

            plt.plot(firing_rate, output_firing_rate, 'b.')
            plt.grid(True)
            plt.show()


    def spikes(self):
        
        return self.firing_rates
    

    def add_bias(self, firing_rate, layer, synapse):
        if 'conv' in layer:
            s = 0
            for oc_idx, oc in enumerate(synapse[4]):
                firing_rate[s:oc] = (firing_rate[s:oc] + synapse[3][oc_idx]) // self.v_th
                s = oc + 1
        elif 'dense' in layer:
            firing_rate = (firing_rate + synapse[3]) // self.v_th
        else:
            firing_rate = firing_rate // self.v_th

        return firing_rate
    
    