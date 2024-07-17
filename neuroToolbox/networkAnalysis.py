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
        
        self.fp_precision = config["conversion"]["fp_precision"]
        self.timesteps = config['conversion']['timesteps']
        self.timesteps = np.log(self.timesteps)
        
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
        x_test = np.floor(x_test[:data_size])
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
                    self.syn_operation += firing_rate[neu_idx] * fan_out
                if '_identity' in layer or 'add' in layer:
                    if layer == 'conv2d_identity':
                        firing_rate = utils.neuron_model(firing_rate, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.fp_precision, self.bias_flag, self.timesteps)
                    if 'add' in layer:
                        firing_rate = firing_rate + shortcut
                    shortcut = firing_rate
                elif '_conv' in layer:
                    shortcut = utils.neuron_model(shortcut, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.fp_precision, self.bias_flag, self.timesteps)
                else:
                    firing_rate = utils.neuron_model(firing_rate, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.fp_precision, self.bias_flag, self.timesteps)

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
                        spikes = utils.neuron_model(spikes, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.fp_precision, self.bias_flag, self.timesteps)
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
                    spikes = utils.neuron_model(spikes, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.fp_precision, self.bias_flag, self.timesteps)
                    fr.append(spikes)
                shortcut = np.array(fr)
            else: 
                for idx in range(len(firing_rate)):
                    spikes = firing_rate[idx].flatten()
                    spikes = utils.neuron_model(spikes, weights[layer], self.v_th[layer], self.t_ref, layer, synapse, self.fp_precision, self.bias_flag, self.timesteps)
                    fr.append(spikes)
                firing_rate = np.array(fr)

            plt.figure(figsize=(10, 10))
            plt.plot(activations, firing_rate, 'o', markersize=2, color='red', linestyle='None')
            plt.title(f"DNN activation vs. Expected firing rates", fontsize=30)
            plt.xlabel(f"Activations in {layer}", fontsize=27)
            plt.ylabel(f"Expected firing rates in {layer}", fontsize=27)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(True)
            # plt.yscale('symlog')
            # plt.ylim([1,200])
            plt.savefig(self.config['paths']['path_wd'] + '/fr_corr' + f"/{layer}", transparent=False)
            plt.show()
    
    
    def set_threshold(self, threshold):
        
        self.v_th = threshold
    
    
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
        
        