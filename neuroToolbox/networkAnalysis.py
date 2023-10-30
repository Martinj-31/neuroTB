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

    def conversionPlot(self):
        activation_dir = os.path.join(self.config['paths']['path_wd'], 'activations')
        os.makedirs(self.config['paths']['path_wd'] + '/plot')
        filepath = self.config.get('paths', 'converted_model')
        filename = self.config.get('paths', 'filename_snn')
        
        with open(filepath + filename + '_Converted_neurons.pkl', 'rb') as f:
            neurons = pickle.load(f)
        with open(filepath + filename + '_Converted_synapses.pkl', 'rb') as f:
            synapses = pickle.load(f)
        with open(filepath + 'threshold.pkl', 'rb') as f:
            threshold = pickle.load(f)

        w_list = []
        synCnt = 0
        for i, layer in enumerate(neurons.keys()):
            if i == 0:
                continue

            print(f"Analysis for {layer}...")

            if 'pooling' in layer:
                pass
            else:
                activations_file = np.load(os.path.join(activation_dir, f'activation_{layer}.npz'))
                activations = activations_file['arr_0']

            source = np.array(list(synapses.values())[i-1][0]) - synCnt
            synCnt += 1024
            target = np.array(list(synapses.values())[i-1][1]) - synCnt
            weights = np.array(list(synapses.values())[i-1][2])
            src = len(np.unique(source))
            tar = len(np.unique(target))
            w = np.zeros(src * tar).reshape(src, tar)

            for j in range(len(weights)):
                w[source[j]][target[j]] = weights[j]
            w_list.append(w)

            acts_list = []
            fr_list = []
            for input_idx in range(len(self.x_norm)):
                firing_rate = self.x_norm[input_idx].flatten()
                for i, w in enumerate(w_list):
                    # firing_rate = np.dot(firing_rate, w) / list(threshold.values())[i]
                    firing_rate = np.dot(firing_rate, w)
                    neg_idx = np.where(firing_rate < 0)[0]
                    firing_rate[neg_idx] = 0
                fr_list = np.concatenate((fr_list, firing_rate))
            # fr_max = np.max(fr_list)
            # fr_list = fr_list / fr_max

            if 'pooling' in layer:
                continue
            elif 'conv' in layer:
                for ic in range(activations.shape[0]):
                    for oc in range(activations.shape[-1]):
                        acts_list = np.concatenate((acts_list, activations[ic, :, :, oc].flatten()))
            elif 'dense' in layer:
                for ic in range(activations.shape[0]):
                    for oc in range(activations.shape[-1]):
                        acts_list = np.concatenate((acts_list, activations[ic, oc].flatten()))
            # acts_max = np.max(acts_list)
            # acts_list = acts_list / acts_max
            
            correlation = np.corrcoef(acts_list, fr_list)[0, 1]
            print(f"Correlation of this layer : {correlation}")
            print(f"Maximum weight of {layer} : {np.max(w)}")
            plt.xlabel(f"Activations in {layer}", size=20)
            plt.ylabel(f"Firing rate in {layer}", size=20)
            plt.xticks(size=15)
            plt.yticks(size=15)
            plt.scatter(acts_list, fr_list, label=f"Correlation: {correlation:.2f}")
            plt.savefig(self.config['paths']['path_wd'] + '/plot' + f"/{layer}")
            plt.show()
            print('')

    def parseCorrPlot(self):
        os.makedirs(self.config["paths"]["path_wd"] + '/batch_corr')

        for layer_1 in self.input_model.layers:
            output_act_input = keras.models.Model(inputs=self.input_model.input, outputs=layer_1.output).predict(self.x_norm)
            for layer_2 in self.parsed_model.layers:
                output_act_parsed = keras.models.Model(inputs=self.parsed_model.input, outputs=layer_2.output).predict(self.x_norm)

                if len(output_act_input.flatten()) == len(output_act_parsed.flatten()):
                    correlation = np.corrcoef(output_act_input.flatten(), output_act_parsed.flatten())[0, 1]

                    plt.figure(figsize=(8, 6))
                    plt.scatter(output_act_input, output_act_parsed, color='b', marker='o', s=10, label=f'Correlation: {correlation:.2f}')
                    plt.xlabel(f'input_model : "{layer_1.name}" layer Activation')
                    plt.ylabel(f'parsed_model : "{layer_2.name}" layer Activation')
                    plt.title('Correlation Plot')
                    
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(self.config["paths"]["path_wd"] + '/batch_corr' + f"/{layer_2.name} of {layer_1.name}")
                    # plt.show()



