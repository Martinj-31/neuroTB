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
        
        self.v_th = config.getfloat('spiking_neuron', 'threshold')
        self.t_ref = config.getint('spiking_neuron', 'refractory') / 1000
            
        bias_flag = config["options"]["bias"]
        if bias_flag == 'False':
            self.bias_flag = False
        elif bias_flag == 'True':
            self.bias_flag = True
        else: print(f"ERROR !!")

        self.snn_filepath = os.path.join(self.config['paths']['models'], self.config['names']['snn_model'])
        os.makedirs(self.config['paths']['path_wd'] + '/snn_model_firing_rates')
        os.makedirs(self.config['paths']['path_wd'] + '/map_corr')
        os.makedirs(self.config['paths']['path_wd'] + '/fr_corr')

        with open(self.snn_filepath + '_Converted_neurons.pkl', 'rb') as f:
            self.neurons = pickle.load(f)
        with open(self.snn_filepath + '_Converted_synapses.pkl', 'rb') as f:
            self.synapses = pickle.load(f)
    

    def run(self, data_size):
        print(f"Preparing for running converted snn.")

        x_test_file = np.load(os.path.join(self.config["paths"]["dataset_path"], 'x_test.npz'))
        x_test = x_test_file['arr_0']
        self.x_test = x_test[::data_size]
        y_test_file = np.load(os.path.join(self.config["paths"]["dataset_path"], 'y_test.npz'))
        y_test = y_test_file['arr_0']
        y_test = y_test[::data_size]

        print(f"Input data length : {len(self.x_test)}")
        print(f"...\n")

        print(f"Loading synaptic weights ...\n")
        
        weights = {}
        for key in self.synapses.keys():
            weights[key] = self.synapses[key][2]

        score = 0
        self.syn_operation = 0
        for input_idx in range(len(self.x_test)):
            firing_rate = self.x_test[input_idx].flatten()
            first_layer_flag = True
            for layer, synapse in self.synapses.items():
                # Calculate synaptic operations
                for neu_idx in range(len(firing_rate)):
                    fan_out = len(np.where(weights[layer][neu_idx][:] > 0)[0])
                    self.syn_operation += firing_rate[neu_idx] * fan_out
                if first_layer_flag:
                    first_layer_flag = False
                    firing_rate = utils.neuron_model(firing_rate, weights[layer], self.v_th, 0, layer, synapse, self.bias_flag, False)
                    log_firing_rate = utils.data_transfer(firing_rate, 'log', False)
                    firing_rate = np.floor(log_firing_rate / (log_firing_rate*self.t_ref + 1))
                else:
                    firing_rate = utils.data_transfer(firing_rate, 'linear', False)
                    firing_rate = utils.neuron_model(firing_rate, weights[layer], self.v_th, 0, layer, synapse, self.bias_flag, False)
                    log_firing_rate = utils.data_transfer(firing_rate, 'log', False)
                    firing_rate = np.floor(log_firing_rate / (log_firing_rate*self.t_ref + 1))
            print(f"Firing rate from output layer for #{input_idx+1} input")
            print(f"{firing_rate}\n")

            if np.argmax(y_test[input_idx]) == np.argmax(firing_rate):
                score += 1
            else: pass

        self.accuracy = (score/len(self.x_test))*100
        print(f"______________________________________")
        print(f"Accuracy : {self.accuracy} %")
        print(f"Synaptic operation : {self.syn_operation}")
        print(f"______________________________________\n")
        print(f"End running\n\n")


    def plot_compare(self):
        activation_dir = os.path.join(self.config['paths']['path_wd'], f"parsed_model_activations")
        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']

        weights = {}
        for key in self.synapses.keys():
            weights[key] = self.synapses[key][2]

        firing_rate = x_norm
        first_layer_flag = True
        for layer, synapse in self.synapses.items():
            act_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{layer}.npz"))
            acts = act_file['arr_0']
            
            activations = utils.Input_Activation(acts, layer)

            if first_layer_flag:
                first_layer_flag = False
                fr = []
                for idx in range(len(firing_rate)):
                    spikes = firing_rate[idx].flatten()
                    spikes = utils.neuron_model(spikes, weights[layer], self.v_th, 0, layer, synapse, self.bias_flag, False)
                    fr.append(spikes)
                firing_rate = np.array(fr)
                log_firing_rate = utils.data_transfer(firing_rate, 'log', False)
                firing_rate = log_firing_rate / (log_firing_rate*self.t_ref + 1)
            else:
                fr = []
                firing_rate = utils.data_transfer(firing_rate, 'linear', False)
                for idx in range(len(firing_rate)):
                    spikes = firing_rate[idx].flatten()
                    spikes = utils.neuron_model(spikes, weights[layer], self.v_th, 0, layer, synapse, self.bias_flag, False)
                    fr.append(spikes)
                firing_rate = np.array(fr)
                log_firing_rate = utils.data_transfer(firing_rate, 'log', False)
                firing_rate = log_firing_rate / (log_firing_rate*self.t_ref + 1)

            plt.figure(figsize=(10, 10))
            plt.plot(activations, firing_rate, 'o', markersize=2, color='red', linestyle='None')
            # plt.title(f"DNN activation vs. SNN firing rates", fontsize=30)
            plt.title(f"DNN activation vs. Expected firing rates", fontsize=30)
            plt.xlabel(f"Activations in {layer}", fontsize=27)
            # plt.ylabel(f"Firing rates in {layer}", fontsize=27)
            plt.ylabel(f"Expected firing rates in {layer}", fontsize=27)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(True)
            plt.yscale('symlog')
            plt.ylim([1,200])
            plt.savefig(self.config['paths']['path_wd'] + '/fr_corr' + f"/{layer}", transparent=True)
            plt.show()
    
    
    def IOcurve(self, axis_scale='linear'):
        weights = utils.weightDecompile(self.synapses)

        plot_input_spikes = {}
        plot_output_spikes = {}
        for layer in self.synapses.keys():
            plot_input_spikes[layer] = []
            plot_output_spikes[layer] = []

        for input_idx in range(len(self.x_test)):
            input_spike = self.x_test[input_idx].flatten()
            input_spike = np.reshape(input_spike, (len(input_spike), 1))
            for layer, synapse in self.synapses.items():
                cnt = 0
                input_spikes = np.zeros(len(input_spike)*len(weights[layer][0])*len(input_spike[0]))
                output_spikes = np.zeros(len(input_spike)*len(weights[layer][0])*len(input_spike[0]))
                for i in range(len(input_spike)):
                    for j in range(len(weights[layer][0])):
                        for k in range(len(input_spike[0])):
                            output_spike = input_spike[i][k] * weights[layer][k][j]
                            if output_spike < 0:
                                output_spike = 0
                            else: pass
                            output_spike = np.floor(output_spike / (output_spike*self.t_ref + self.v_th))
                            input_spikes[cnt] = input_spike[i][k] * weights[layer][k][j]
                            output_spikes[cnt] = output_spike
                            cnt += 1
                idx = np.where(input_spikes >= 0)[0]
                plot_input_spikes[layer] = np.concatenate((plot_input_spikes[layer], input_spikes[idx]))
                plot_output_spikes[layer] = np.concatenate((plot_output_spikes[layer], output_spikes[idx]))

                next_spike = np.dot(input_spike.flatten(), weights[layer])
                neg_idx = np.where(next_spike < 0)[0]
                next_spike[neg_idx] = 0
                next_spike = np.floor(next_spike / (next_spike*self.t_ref + self.v_th))
                next_spike = np.reshape(next_spike, (len(next_spike), 1))
                input_spike = next_spike

        for layer in self.synapses.keys():
            plt.plot(plot_input_spikes[layer], plot_output_spikes[layer], 'b.')
            plt.title(f"IO curve {layer}")
            plt.grid(True)
            if axis_scale == 'symlog':
                plt.xscale('symlog')
                plt.yscale('symlog')
            else: pass
            plt.show()


    def spikes(self):
        
        return self.firing_rates
            
    
    def act_compare(self):

        print(f"##### Comparing activations between input model and parsed model. #####")

        input_model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"{self.input_model_name}.h5"))
        parsed_model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"parsed_{self.input_model_name}.h5"))
        
        input_model_activation_dir = os.path.join(self.config['paths']['path_wd'], 'input_model_activations')
        corr_dir = os.path.join(self.config['paths']['path_wd'], 'acts_corr')
        
        def plot(input_act, parsed_act, input_layer, parsed_layer):
            plt.figure(figsize=(10, 10))
            plt.scatter(input_act, parsed_act, color='b', marker='o', s=10)
            plt.title(f"Input vs. Parsed activation correlation", fontsize=30)
            plt.xlabel(f'input_model : "{input_layer.name}" Activation', fontsize=27)
            plt.ylabel(f'parsed_model : "{parsed_layer.name}" Activation', fontsize=27)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(True)
            plt.savefig(corr_dir + f"/{input_layer.name} - {parsed_layer.name}")
            plt.show()

        print(f'input_model_name : {self.input_model_name}')
        os.makedirs(corr_dir, exist_ok=True)
        if 'ResNet' in self.input_model_name:
            input_idx = 0
            add_idx = 0 
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
                            if 'add' in input_layer.name:
                                if (add_idx == 0):
                                    if ('batch' in input_model.layers[input_idx-2].name) and ('concatenate' == parsed_layer.name): # convolution block (Input phase)
                                        print(f'Current parsed layer name : {parsed_layer.name}') 
                                        loaded_activation_A = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-1].name}.npz"))['arr_0']
                                        loaded_activation_B = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-2].name}.npz"))['arr_0']
                                    elif ('conv' in input_model.layers[input_idx-2].name) and ('concatenate' == parsed_layer.name): # identity block (Input phase)
                                        print(f'Current parsed layer name : {parsed_layer.name}') 
                                        loaded_activation_A = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-1].name}.npz"))['arr_0']
                                        loaded_activation_B = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-5].name}.npz"))['arr_0']
                                    else: 
                                        continue

                                    idx = parsed_model.layers.index(parsed_layer)

                                    parsed_act = keras.models.Model(inputs=parsed_layer.input, outputs=parsed_layer.output).predict([loaded_activation_A , loaded_activation_B])
                                    parsed_act = keras.models.Model(inputs=parsed_model.layers[idx+1].input, outputs=parsed_model.layers[idx+1].output).predict([parsed_act])
                                    plot(input_act, parsed_act, input_layer, parsed_model.layers[idx+1])
                                    add_idx +=1
                                    break
                                elif (add_idx != 0):
                                    if ('batch' in input_model.layers[input_idx-2].name) and ('concatenate' in parsed_layer.name) and (input_layer.name[-2:] == parsed_layer.name[-2:]): # convolution block 
                                        print(f'Current parsed layer name : {parsed_layer.name}') 
                                        loaded_activation_A = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-1].name}.npz"))['arr_0']
                                        loaded_activation_B = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-2].name}.npz"))['arr_0']
                                    elif ('conv' in input_model.layers[input_idx-2].name) and ('concatenate' in parsed_layer.name) and (input_layer.name[-2:] == parsed_layer.name[-2:]): # identity block 
                                        print(f'Current parsed layer name : {parsed_layer.name}') 
                                        loaded_activation_A = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-1].name}.npz"))['arr_0']
                                        if 'ResNet50' in self.input_model_name:
                                            loaded_activation_B = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-7].name}.npz"))['arr_0']
                                        else:
                                            loaded_activation_B = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-5].name}.npz"))['arr_0']
                                    else: continue

                                    idx = parsed_model.layers.index(parsed_layer)

                                    parsed_act = keras.models.Model(inputs=parsed_layer.input, outputs=parsed_layer.output).predict([loaded_activation_A , loaded_activation_B])
                                    parsed_act = keras.models.Model(inputs=parsed_model.layers[idx+1].input, outputs=parsed_model.layers[idx+1].output).predict([parsed_act])
                                    plot(input_act, parsed_act, input_layer, parsed_model.layers[idx+1])
                                    add_idx +=1
                                    break
                            elif 'global_average_pooling2d' in input_layer.name:
                                if 'global_average_pooling2d_avg' in parsed_layer.name:
                                    print(f'Current parsed layer name : {parsed_layer.name}')
                                    print(f'input_model.layers[input_idx-1].name : {input_model.layers[input_idx-1].name}')
                                    loaded_activation = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-1].name}.npz"))['arr_0']
                                    
                                    idx = parsed_model.layers.index(parsed_layer)

                                    parsed_act = keras.models.Model(inputs=parsed_layer.input, outputs=parsed_layer.output).predict(loaded_activation)
                                    parsed_act = keras.models.Model(inputs=parsed_model.layers[idx+1].input, outputs=parsed_model.layers[idx+1].output).predict([parsed_act])
                                    plot(input_act, parsed_act, input_layer, parsed_model.layers[idx+1])
                                    break
                                else: continue
                        else:
                            if 'batch' in input_layer.name:
                                if ('batch' in input_model.layers[input_idx+1].name) and (input_model.layers[input_idx-2].name == parsed_layer.name) and (input_layer.name[-2:] == parsed_layer.name[-2:]):
                                    print(f'Current parsed layer name : {parsed_layer.name}')
                                    loaded_activation = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-3].name}.npz"))['arr_0']
                                    pass
                                elif ('batch' in input_model.layers[input_idx-1].name) and (input_model.layers[input_idx-2].name == parsed_layer.name) and (input_layer.name[-2:] == parsed_layer.name[-2:]):
                                    if 'ResNet50' in self.input_model_name:
                                        print(f'Current parsed layer name : {parsed_layer.name}')
                                        loaded_activation = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-8].name}.npz"))['arr_0']
                                    else:   
                                        print(f'Current parsed layer name : {parsed_layer.name}')
                                        loaded_activation = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-6].name}.npz"))['arr_0']
                                    pass
                                elif (parsed_layer.name == input_model.layers[input_idx-1].name) and (input_layer.name[-2:] == parsed_layer.name[-2:]):
                                    print(f'Current parsed layer name : {parsed_layer.name}')
                                    loaded_activation = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-2].name}.npz"))['arr_0']
                                    pass
                                elif (parsed_layer.name == input_model.layers[input_idx-1].name) and ('input' in input_model.layers[input_idx-2].name):
                                    print(f'Current parsed layer name : {parsed_layer.name}')
                                    loaded_activation = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-2].name}.npz"))['arr_0']
                                    pass
                                else: continue
                            elif (input_layer.name == parsed_layer.name) and ('conv' in input_layer.name) and ('conv' in input_model.layers[input_idx-1].name):
                                if 'ResNet50' in self.input_model_name:
                                    print(f'Current parsed layer name : {parsed_layer.name}')
                                    loaded_activation = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-6].name}.npz"))['arr_0']
                                else:
                                    print(f'Current parsed layer name : {parsed_layer.name}')
                                    loaded_activation = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-4].name}.npz"))['arr_0']
                            else:
                                if input_layer.name == parsed_layer.name :
                                    print(f'Current parsed layer name : {parsed_layer.name}')
                                    loaded_activation = np.load(os.path.join(self.config['paths']['path_wd'], 'input_model_activations', f"input_model_activation_{input_model.layers[input_idx-1].name}.npz"))['arr_0']
                                else : continue

                            parsed_act = keras.models.Model(inputs=parsed_layer.input, outputs=parsed_layer.output).predict(loaded_activation)
                            plot(input_act, parsed_act, input_layer, parsed_layer)
                input_idx += 1
                print(' ')

        else: 
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
    
    
    def input_log_domain_trans_plot(self):
        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']
        
        ori_x = x_norm
        max_value = np.max(abs(ori_x))
        log_x = utils.data_transfer(ori_x, 'log', max_value)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(ori_x, log_x, color='r', marker='o', s=10)
        plt.title(f"Log domain transfer correlation", fontsize=30)
        plt.xlabel(f"Before transfer", fontsize=27)
        plt.ylabel(f"After transfer", fontsize=27)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.yscale('symlog')
        plt.ylim([1, 500])
        plt.show()
                
    
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
        logfile.writelines(f"Input domain : {self.config['options']['trans_domain']} \n")
        
        logfile.writelines(f"\n")
        
        logfile.writelines(f"/// Neuron setup \n")
        logfile.writelines(f"Neuron model : {self.config['conversion']['neuron']} neuron \n")
        if 'IF' == self.config["conversion"]["neuron"]:
            logfile.writelines(f"Percentile : {self.config['options']['percentile']} % \n")
        elif 'LIF' == self.config["conversion"]["neuron"]:
            logfile.writelines(f"Refractory period : {self.config['spiking_neuron']['refractory']} ms \n")
            logfile.writelines(f"Threshold : {self.config['spiking_neuron']['threshold']} \n")
        
        logfile.writelines(f"\n")
        
        logfile.writelines(f"/// RESULT \n")
        logfile.writelines(f"Input Model Accuracy : {float(self.config['result']['input_model_acc'])*100:.2f} %\n")
        logfile.writelines(f"Parsed Model Accuracy : {float(self.config['result']['parsed_model_acc'])*100:.2f} %\n\n")
        logfile.writelines(f"Accuracy for {self.config['names']['dataset']} {self.config['test']['data_size']} : {self.accuracy} % \n")
        logfile.writelines(f"Synaptic operation : {self.syn_operation} \n")
        logfile.writelines(f"\n")
        logfile.writelines(f"\n")
        logfile.writelines(f"///////////////////////////////////////////////// \n")
        
        logfile.close()
        
        