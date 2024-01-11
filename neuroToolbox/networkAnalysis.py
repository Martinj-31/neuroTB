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

    def __init__(self, x_norm, input_model_name, config):
        """
        Initialize the networkAnalysis instance.

        Args:
            x_norm (Numpy.ndarray): Input dataset for analysis.
            input_model_name (String): Input model name for SNN conversion.
            config (configparser.ConfigParser): Configuration settings for compiling.
        """
        self.config = config

        self.x_norm = x_norm
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
    

    def run(self, image, label):
        print(f"Preparing for running converted snn.")
        x_test = image
        y_test = label
        syn_operation = 0

        print(f"Input data length : {len(x_test)}")
        print(f"...\n")

        print(f"Loading synaptic weights ...\n")
        fr_dist = {}
        w_dict = {}
        w_cal = {}
        for layer in self.synapses.keys():
            fr_dist[layer] = []
            w_dict[layer] = []
            w_cal[layer] = []
        
        score = 0
        for input_idx in range(len(x_test)):
            synCnt = 0
            firing_rate = x_test[input_idx].flatten()
            for layer, neuron in self.synapses.items():
                src = np.array(neuron[0]) - synCnt
                synCnt += 1024
                tar = np.array(neuron[1]) - synCnt
                w = np.array(neuron[2])
                source = len(np.unique(src))
                target = len(np.unique(tar))
                weights = np.zeros(source * target).reshape(source, target)
                w_dict[layer] = w
                for i in range(len(w)):
                    weights[src[i]][tar[i]] = w[i]

                for neu_idx in range(len(firing_rate)):
                    fan_out = len(np.where(weights[neu_idx][:] > 0)[0])
                    syn_operation += firing_rate[neu_idx] * fan_out

                firing_rate = np.dot(firing_rate, weights)
                if 'conv' in layer:
                    s = 0
                    for oc_idx, oc in enumerate(neuron[4]):
                        firing_rate[s:oc] = (firing_rate[s:oc] / self.v_th) + neuron[3][oc_idx]
                        firing_rate[s:oc] = np.floor(firing_rate[s:oc] / (firing_rate[s:oc]*self.t_ref + self.v_th))
                        s = oc
                else:
                    firing_rate = firing_rate // self.v_th
                    firing_rate = np.floor(firing_rate / (firing_rate*self.t_ref + self.v_th))
                neg_idx = np.where(firing_rate < 0)[0]
                firing_rate[neg_idx] = 0
                fr_dist[layer] = np.concatenate((fr_dist[layer], firing_rate))
            print(f"Firing rate from output layer for #{input_idx+1} input")
            print(firing_rate)
            print('')

            if np.argmax(y_test[input_idx]) == np.argmax(firing_rate):
                score += 1
            else: pass

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

        print(f"______________________________________")
        print(f"Accuracy : {(score/len(x_test))*100} %")
        print(f"Synaptic operation : {syn_operation}")
        print(f"______________________________________\n")
        print(f"End running\n\n")

    
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
                        firing_rate = np.dot(firing_rate, weights)
                        if 'conv' in snn_layer[0]:
                            s = 0
                            for oc_idx, oc in enumerate(snn_layer[1][4]):
                                firing_rate[s:oc] = (firing_rate[s:oc] / self.v_th) + snn_layer[1][3][oc_idx]
                                firing_rate[s:oc] = np.floor(firing_rate[s:oc] / (firing_rate[s:oc]*self.t_ref + self.v_th))
                                s = oc
                        else:
                            firing_rate = firing_rate // self.v_th
                            firing_rate = np.floor(firing_rate / (firing_rate*self.t_ref + self.v_th))
                        neg_idx = np.where(firing_rate < 0)[0]
                        firing_rate[neg_idx] = 0
                        snn_fr = np.concatenate((snn_fr, firing_rate))

                    loaded_act_file = np.load(os.path.join(activation_dir, f"input_model_activation_{input_layer.name}.npz"))
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
