import sys, os, pickle
sys.path.append(os.getcwd())

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

import neuroToolbox.neuPLUSNetwork as net

class evaluation:
    def __init__(self, parsed_model, config):
        self.config = config
        self.parsed_model = parsed_model

    def genLayer(self):
        print(f"")
        print(f"SNN model will be stored layer by layer for evaluation.")
        print(f"")
        filepath = self.config.get('paths', 'evaluation_layers')
        filename = self.config.get('paths', 'filename_snn')
        os.makedirs(filepath)

        temp = {}
        eval = net.networkGen(self.parsed_model, self.config)
        for layer in self.parsed_model.layers[1:]:
            layer_type = layer.__class__.__name__
            if layer_type == 'Dense':
                temp[layer_type] = eval.Synapse_dense(layer, evaluation=True)
                with open(filepath + filename + '_' + layer.name + '_eval.pkl', 'wb') as f:
                    pickle.dump(temp, f)
            elif 'Conv' in layer_type:
                temp[layer_type] = eval.Synapse_convolution(layer, evaluation=True)    
                with open(filepath + filename + '_' + layer.name + '_eval.pkl', 'wb') as f:
                    pickle.dump(temp, f)
            else:
                continue
            print(f"Generate {layer_type} for evaluation.")
        
        print(f"")
        print(f"All work is done.")
        print(f"Take to your hardware.")

    def evaluate(self, layer):
        # Activates from DNN.
        activates = np.load(self.filepath + '/activations/' + layer.name + '.npz')

        # Spike rate from SNN.
        spike_rate = np.load(self.filepath + '/spike_train/' + layer.name + 'npz')

        x=np.arange(10)
        y=x
        plt.plot(activates, spike_rate, 'b.')
        plt.plot(x, y, 'r', alpha=0.5)
        plt.show()
