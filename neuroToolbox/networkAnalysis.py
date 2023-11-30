import os, sys, configparser

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt

class Analysis:

    def __init__(self, x_norm, input_model_name, threshold, config):
        self.config = config

        self.x_norm = x_norm
        self.input_model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"{input_model_name}.h5"))
        self.parsed_model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"parsed_{input_model_name}.h5"))
        self.input_model_name = input_model_name
        self.threshold = threshold

        self.snn_filepath = os.path.join(self.config['paths']['models'], self.config['names']['snn_model'])
        os.makedirs(self.config['paths']['path_wd'] + '/snn_model_firing_rates')
        os.makedirs(self.config['paths']['path_wd'] + '/map_corr')
        os.makedirs(self.config['paths']['path_wd'] + '/fr_corr')

        with open(self.snn_filepath + '_Converted_neurons.pkl', 'rb') as f:
            self.neurons = pickle.load(f)
        with open(self.snn_filepath + '_Converted_synapses.pkl', 'rb') as f:
            self.synapses = pickle.load(f)
            

    def evalMapping(self, name='input'):        
        if 'input' == name:
            activation_dir = os.path.join(self.config['paths']['path_wd'], 'input_model_activations')
            model = self.input_model
        elif 'parsed' == name:
            activation_dir = os.path.join(self.config['paths']['path_wd'], 'parsed_model_activations')
            model = self.parsed_model
        
        synCnt = 0
        w_list = []
        for synapse in self.synapses.values():
            src = np.array(synapse[0]) - synCnt
            synCnt += 1024
            tar = np.array(synapse[1]) - synCnt
            w = np.array(synapse[2])
            source = len(np.unique(src))
            target = len(np.unique(tar))
            weights = np.zeros(source * target).reshape(source, target)
            
            for i in range(len(w)):
                weights[src[i]][tar[i]] = w[i]
            w_list.append(weights)
        
        input_idx = 0
        w_idx = 0
        for layer in model.layers:

            print(f"Analysis for {layer.name} ...")

            if 'input' in layer.name:
                input_idx += 1
                continue
            elif 'flatten' in layer.name:
                input_idx += 1
                continue
            else:
                input_act_file = np.load(os.path.join(activation_dir, f"input_model_activation_{layer.name}.npz"))
                input_act = input_act_file['arr_0']
            for neuron_idx, neuron in enumerate(self.neurons.keys()):
                if 'input' in neuron:
                    continue
                else:
                    if np.prod(layer.output_shape[1:]) != list(self.neurons.values())[neuron_idx]:
                        continue
                    else:
                        if 'batch' in layer.name:
                            if neuron == model.layers[input_idx-1].name:
                                neuron_name = model.layers[input_idx-2].name
                            else: continue
                        else:
                            if layer.name == neuron:
                                neuron_name = model.layers[input_idx-1].name
                                if 'flatten' in neuron_name:
                                    neuron_name = model.layers[input_idx-2].name
                                else: pass
                            else: continue
                        loaded_activation_file = np.load(os.path.join(activation_dir, f"input_model_activation_{neuron_name}.npz"))
                        loaded_activation = loaded_activation_file['arr_0']
                        loaded_acts = []
                        if 'input' in neuron_name:
                            for ic in range(loaded_activation.shape[0]):
                                temp = []
                                for oc in range(loaded_activation.shape[-1]):
                                    temp = np.concatenate((temp, loaded_activation[ic, :, :, oc].flatten()))
                                loaded_acts.append(temp)
                        elif 'conv' in neuron_name:
                            for ic in range(loaded_activation.shape[0]):
                                temp = []
                                for oc in range(loaded_activation.shape[-1]):
                                    temp = np.concatenate((temp, loaded_activation[ic, :, :, oc].flatten()))
                                loaded_acts.append(temp)
                        elif 'pooling' in neuron_name:
                            for ic in range(loaded_activation.shape[0]):
                                temp = []
                                for oc in range(loaded_activation.shape[-1]):
                                    for i in range(loaded_activation.shape[1]):
                                        for j in range(loaded_activation.shape[2]):
                                            temp = np.concatenate((temp, loaded_activation[ic, i, j, oc].flatten()))
                                loaded_acts.append(temp)
                        elif 'batch' in neuron_name:
                            for ic in range(loaded_activation.shape[0]):
                                temp = []
                                for oc in range(loaded_activation.shape[-1]):
                                    temp = np.concatenate((temp, loaded_activation[ic, :, :, oc].flatten()))
                                loaded_acts.append(temp)
                        elif 'dense' in neuron_name:
                            for ic in range(loaded_activation.shape[0]):
                                temp = []
                                for oc in range(loaded_activation.shape[-1]):
                                    temp = np.concatenate((temp, loaded_activation[ic, oc].flatten()))
                                loaded_acts.append(temp)
                        else: pass

                        acts = []
                        if 'conv' in neuron:
                            for ic in range(input_act.shape[0]):
                                for oc in range(input_act.shape[-1]):
                                    acts = np.concatenate((acts, input_act[ic, :, :, oc].flatten()))
                        elif 'pooling' in neuron:
                            for ic in range(input_act.shape[0]):
                                for oc in range(input_act.shape[-1]):
                                    for i in range(input_act.shape[1]):
                                        for j in range(input_act.shape[2]):
                                            acts = np.concatenate((acts, input_act[ic, i, j, oc].flatten()))
                        elif 'dense' in neuron:
                            for ic in range(input_act.shape[0]):
                                for oc in range(input_act.shape[-1]):
                                    acts = np.concatenate((acts, input_act[ic, oc].flatten()))
                        else: pass

                        snn_fr = []
                        if 'batch' in layer.name:
                            w_idx -= 1
                        else: pass
                        for idx in range(len(loaded_acts)):
                            firing_rate = loaded_acts[idx].flatten()
                            firing_rate = np.dot(firing_rate, w_list[w_idx])
                            neg_idx = np.where(firing_rate < 0)[0]
                            firing_rate[neg_idx] = 0
                            snn_fr = np.concatenate((snn_fr, firing_rate))
                        
                        plt.figure(figsize=(10, 10))
                        plt.scatter(acts, snn_fr, color='b', marker='o', s=10)
                        plt.title(f"DNN vs. SNN activation correlation", fontsize=30)
                        plt.xlabel(f"Activations in {layer.name}", fontsize=27)
                        plt.ylabel(f"Firing rate in {neuron}", fontsize=27)
                        plt.xticks(fontsize=20)
                        plt.yticks(fontsize=20)
                        plt.grid(True)
                        plt.savefig(self.config['paths']['path_wd'] + '/map_corr' + f"/{neuron}")
                        plt.show()
                    w_idx += 1
            input_idx += 1
            print('')
            

    def evalNetwork(self):
        activation_dir = os.path.join(self.config['paths']['path_wd'], 'input_model_activations')

        input_idx = 0
        synCnt = 0
        for input_layer in self.input_model.layers:
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
                        synCnt -= 1024
                        if snn_layer[0] == self.input_model.layers[input_idx-1].name:
                            snn_layer_name = self.input_model.layers[input_idx-2].name
                        else: continue
                    else:
                        if input_layer.name == snn_layer[0]:
                            snn_layer_name = self.input_model.layers[input_idx-1].name
                            if 'flatten' in snn_layer_name:
                                snn_layer_name = self.input_model.layers[input_idx-2].name
                            else: pass
                        else: continue
                    
                    input_act_file = np.load(os.path.join(activation_dir, f"input_model_activation_{snn_layer_name}.npz"))
                    input_act = input_act_file['arr_0']
                    input_acts = []
                    if 'input' in snn_layer_name:
                        for ic in range(input_act.shape[0]):
                            temp = []
                            for oc in range(input_act.shape[-1]):
                                temp = np.concatenate((temp, input_act[ic, :, :, oc].flatten()))
                            input_acts.append(temp)
                    elif 'conv' in snn_layer_name:
                        for ic in range(input_act.shape[0]):
                            temp = []
                            for oc in range(input_act[-1]):
                                temp = np.concatenate((temp, input_act[ic, :, :, oc].flatten()))
                            input_acts.append(temp)
                    elif 'batch' in snn_layer_name:
                        for ic in range(input_act.shape[0]):
                            temp = []
                            for oc in range(input_act.shape[-1]):
                                temp = np.concatenate((temp, input_act[ic, :, :, oc].flatten()))
                            input_acts.append(temp)
                    elif 'pooling' in snn_layer_name:
                        for ic in range(input_act.shape[0]):
                            temp = []
                            for oc in range(input_act.shape[-1]):
                                for i in range(input_act.shape[1]):
                                    for j in range(input_act.shape[2]):
                                        temp = np.concatenate((temp, input_act[ic, i, j, oc].flatten()))
                            input_acts.append(temp)
                    elif 'dense' in snn_layer_name:
                        for ic in range(input_act.shape[0]):
                            temp = []
                            for oc in range(input_act.shape[-1]):
                                temp = np.concatenate((temp, input_act[ic, oc].flatten()))
                            input_acts.append(temp)
                    else: pass

                    src = np.array(snn_layer[1][0]) - synCnt
                    synCnt += 1024
                    tar = np.array(snn_layer[1][1]) - synCnt
                    w = np.array(snn_layer[1][2])
                    source = len(np.unique(src))
                    target = len(np.unique(tar))
                    weights = np.zeros(source * target).reshape(source, target)
                    for i in range(len(w)):
                        weights[src[i]][tar[i]] = w[i]
                    snn_fr = []
                    for idx in range(len(input_acts)):
                        firing_rate = input_acts[idx].flatten()
                        if 'conv' in snn_layer[0]:
                            firing_rate = np.dot(firing_rate, weights)
                            s = 0
                            for oc in range(len(snn_layer[1][3])):
                                firing_rate[s:s+oc] = firing_rate[s:s+oc] // self.threshold[snn_layer[0]][oc]
                                s += oc
                            neg_idx = np.where(firing_rate < 0)[0]
                            firing_rate[neg_idx] = 0
                        else:
                            firing_rate = np.dot(firing_rate, weights)
                            firing_rate = firing_rate / self.threshold[snn_layer[0]]
                            neg_idx = np.where(firing_rate < 0)[0]
                            firing_rate[neg_idx] = 0
                        snn_fr = np.concatenate((snn_fr, firing_rate))

                    loaded_act_file = np.load(os.path.join(activation_dir, f"input_model_activation_{input_layer.name}.npz"))
                    loaded_act = loaded_act_file['arr_0']
                    acts = []
                    if 'conv' in snn_layer[0]:
                        for ic in range(loaded_act.shape[0]):
                            for oc in range(loaded_act.shape[-1]):
                                acts = np.concatenate((acts, loaded_act[ic, :, :, oc].flatten()))
                    elif 'pooling' in snn_layer[0]:
                        for ic in range(loaded_act.shape[0]):
                            for oc in range(loaded_act.shape[-1]):
                                for i in range(loaded_act.shape[1]):
                                    for j in range(loaded_act.shape[2]):
                                        acts = np.concatenate((acts, loaded_act[ic, i, j, oc].flatten()))
                    elif 'dense' in snn_layer[0]:
                        for ic in range(loaded_act.shape[0]):
                            for oc in range(loaded_act.shape[-1]):
                                acts = np.concatenate((acts, loaded_act[ic, oc].flatten()))
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
            print('')