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
        os.makedirs(self.config['paths']['path_wd'] + '/plot2')
        
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
            source = np.array(synapse[0]) - synCnt
            synCnt += 1024
            target = np.array(synapse[1]) - synCnt
            weights = np.array(synapse[2])
            src = len(np.unique(source))
            tar = len(np.unique(target))
            w = np.zeros(src * tar).reshape(src, tar)
            
            for j in range(len(weights)):
                w[source[j]][target[j]] = weights[j]
            w_list.append(w)
        
        input_idx = 0
        w_idx = 0
        for layer in model.layers:
            if 'input' == layer.name:
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
                            print(f"Current input model layer : {layer.name}")
                            print(f"Current SNN layer : {neuron}")
                            neuron_name = model.layers[input_idx-2].name
                            loaded_activation_file = np.load(os.path.join(activation_dir, f"input_model_activation_{model.layers[input_idx-2].name}.npz"))
                            print(f"Input layer for SNN : {neuron_name}")
                        else:
                            print(f"Current input model layer : {layer.name}")
                            print(f"Current SNN layer : {neuron}")
                            neuron_name = model.layers[input_idx-1].name
                            if 'flatten' in neuron_name:
                                neuron_name = model.layers[input_idx-2].name
                                loaded_activation_file = np.load(os.path.join(activation_dir, f"input_model_activation_{model.layers[input_idx-2].name}.npz"))
                            else:
                                loaded_activation_file = np.load(os.path.join(activation_dir, f"input_model_activation_{model.layers[input_idx-1].name}.npz"))
                            print(f"Input layer for SNN : {neuron_name}")
                        loaded_activation = loaded_activation_file['arr_0']
                        print(loaded_activation.shape)
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
                                    temp = np.concatenate((temp, loaded_activation[ic, :, :, oc].flatten()))
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
                                    acts = np.concatenate((acts, input_act[ic, :, :, oc].flatten()))
                        elif 'dense' in neuron:
                            for ic in range(input_act.shape[0]):
                                for oc in range(input_act.shape[-1]):
                                    acts = np.concatenate((acts, input_act[ic, oc].flatten()))

                        idx = 0
                        snn_fr = []
                        if 'batch' in layer.name:
                            w_idx -= 1
                        else: pass
                        print(f"Weight index ; {w_idx}")
                        for idx in range(len(loaded_acts)):
                            firing_rate = loaded_acts[idx].flatten()
                            print(f"{firing_rate.shape} X {w_list[w_idx].shape}")
                            firing_rate = np.dot(firing_rate, w_list[w_idx])
                            neg_idx = np.where(firing_rate < 0)[0]
                            firing_rate[neg_idx] = 0
                            snn_fr = np.concatenate((snn_fr, firing_rate))
                        print(len(acts), len(snn_fr))
                        plt.xlabel(f"Activations in {layer.name}", size=20)
                        plt.ylabel(f"Firing rate in {neuron}", size=20)
                        plt.xticks(size=15)
                        plt.yticks(size=15)
                        plt.scatter(acts, snn_fr, color='b', marker='o', s=10)
                        plt.savefig(self.config['paths']['path_wd'] + '/plot2' + f"/{neuron}")
                        plt.show()
                        print('')
                    w_idx += 1
            input_idx += 1
            