import os, sys, configparser

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt

class Analysis:

    def __init__(self, x_norm, input_model_name, config):
        self.config = config

        self.x_norm = x_norm
        self.input_model = keras.models.load_model(os.path.join(self.config["paths"]["path_wd"], f"{input_model_name}.h5"))
        self.parsed_model = keras.models.load_model(os.path.join(self.config["paths"]["path_wd"], f"parsed_{input_model_name}.h5"))

        self.syn_operation = 0


    def evalMapping(self, input_model_name, name='input'):
        filepath = os.path.join(self.config['paths']['converted_model'], self.config['paths']['filename_snn'])
        os.makedirs(self.config['paths']['path_wd'] + '/fr_corr')
        
        with open(filepath + '_Converted_neurons.pkl', 'rb') as f:
            neurons = pickle.load(f)
        with open(filepath + '_Converted_synapses.pkl', 'rb') as f:
            synapses = pickle.load(f)
            
        if 'input' == name:
            activation_dir = os.path.join(self.config['paths']['path_wd'], 'input_model_activations')
            model = keras.models.load_model(os.path.join(self.config["paths"]["path_wd"], f"{input_model_name}.h5"))
        elif 'parsed' == name:
            activation_dir = os.path.join(self.config['paths']['path_wd'], 'parsed_model_activations')
            model = keras.models.load_model(os.path.join(self.config["paths"]["path_wd"], f"parsed_{input_model_name}.h5")) 
        
        synCnt = 0
        w_list = []
        for synapse in synapses.values():
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
            for neuron_idx, neuron in enumerate(neurons.keys()):
                if 'input' in neuron:
                    continue
                else:
                    if np.prod(layer.output_shape[1:]) != list(neurons.values())[neuron_idx]:
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
                        print(f"Input activation : {neuron_name}")
                        print(f"Current SNN layer name : {neuron}")
                        idx = 0
                        snn_fr = []
                        if 'batch' in layer.name:
                            w_idx -= 1
                        else: pass
                        for idx in range(len(loaded_acts)):
                            firing_rate = loaded_acts[idx].flatten()
                            firing_rate = np.dot(firing_rate, w_list[w_idx])
                            neg_idx = np.where(firing_rate < 0)[0]
                            firing_rate[neg_idx] = 0
                            self.syn_operation += np.sum(firing_rate)
                            snn_fr = np.concatenate((snn_fr, firing_rate))
                        
                        plt.figure(figsize=(15, 15))
                        plt.scatter(acts, snn_fr, color='b', marker='o', s=10)
                        plt.title(f"DNN vs. SNN activation correlation", fontsize=30)
                        plt.xlabel(f"Activations in {layer.name}", fontsize=27)
                        plt.ylabel(f"Firing rate in {neuron}", fontsize=27)
                        plt.xticks(fontsize=20)
                        plt.yticks(fontsize=20)
                        plt.grid(True)
                        plt.savefig(self.config['paths']['path_wd'] + '/fr_corr' + f"/{neuron}")
                        plt.show()
                    w_idx += 1
            input_idx += 1
            print('')
        print(f"Synapse operation : {self.syn_operation}")
            