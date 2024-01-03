import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import neuroToolbox.utils as utils

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


class Convert:
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        self.threshold = {}


    def convertWeights(self):
        activation_dir = os.path.join(self.config['paths']['path_wd'], 'parsed_model_activations')

        print("\n\n######## Weight conversion ########\n\n")

        for layer in self.model.layers:
            if len(layer.weights) == 0:
                continue
            
            inbound = utils.get_inbound_layers_with_params(layer)
            prev_layer = inbound[0]

            activations_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{prev_layer.name}.npz"))
            activations = activations_file['arr_0']

            perc = self.config.getfloat('conversion', 'percentile')
            v_th = self.config.getint('conversion', 'threshold')
            t_ref = self.config.getint('conversion', 'refractory') / 1000
            ratio = self.config.getfloat('conversion', 'ratio')

            weight, bias = layer.get_weights()

            if 'conv' in layer.name:
                for oc in range(weight.shape[-1]):
                    acts_range = utils.get_percentile_activation(activations, perc) / ratio + bias[oc]
                    print(f"99.9th percentile activations of channel {oc+1} from {prev_layer.name} : {acts_range}")
                    weight[:, :, :, oc] = weight[:, :, :, oc] * (v_th / (1 - acts_range * t_ref))
            else:
                acts_range = utils.get_percentile_activation(activations, perc) / ratio
                print(f"99.9th percentile activations : {acts_range}")
                weight = weight * (v_th / (1 - acts_range * t_ref))
            
            layer.set_weights([weight, bias])
            print('')
    

    def normalize_parameters(self):
        
        activation_dir = os.path.join(self.config['paths']['path_wd'], 'activations')
        os.makedirs(activation_dir, exist_ok=True)
        
        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']  # Access the data stored in the .npz file
        
        print("\n\n######## Weight conversion ########\n\n")
        
        # Declare and initialize variables
        batch_size = self.config.getint('conversion', 'batch_size')

        # Norm factors initialization
        norm_facs = {self.model.layers[0].name: 1.0}

        i = 0          
        # Layer rotation of the parsed_model
        for layer in self.model.layers:
            # Skip if there is no weight in the layer
            if len(layer.weights) == 0:
                continue
            
            activations = self.get_activations_layer(self.model, layer, 
                                                     x_norm, batch_size, activation_dir)

            print("Maximum activation: {:.5f}.".format(np.max(activations)))
            nonzero_activations = activations[np.nonzero(activations)]
            del activations
            perc = self.config.getfloat('conversion', 'percentile')
            
            cliped_max_activation = self.get_percentile_activation(nonzero_activations, perc)
            norm_facs[layer.name] = cliped_max_activation
            print("Cliped maximum activation: {:.5f}.\n".format(norm_facs[layer.name]))
            i += 1
            
          
        # Apply scale factor to normalize parameters for parsed_model layer
        for layer in self.model.layers:
            
            if len(layer.weights) == 0:
                continue
            
            # Adjust weight part 
            ann_weights = list(layer.get_weights())[0]
            if layer.activation.__name__ == 'softmax':
                norm_fac = 1.0
                print("\n Using norm_factor: {:.2f}.".format(norm_fac))
            else:
                norm_fac = norm_facs[layer.name]
            
            # Check the previous layer of that layer through _inbound_nodes
            inbound = utils.get_inbound_layers_with_params(layer)

            # Weight normalization
            if len(inbound) == 0: #Input layer
                ann_weights_norm = \
                    ann_weights * norm_facs[self.model.layers[0].name] / norm_fac
                print("\n +++++ input norm_facs +++++ \n ", norm_facs[self.model.layers[0].name])
                print("  ---------------")
                print(" ", norm_fac)
           
            elif len(inbound) == 1:                   
                ann_weights_norm = \
                    ann_weights * norm_facs[inbound[0].name] / norm_fac
                print("\n +++++ norm_facs +++++\n ", norm_facs[inbound[0].name])
                print("  ---------------")
                print(" ", norm_fac)              
            
            else:
                ann_weights_norm = ann_weights
            
            # --- NeuPLUS weight ---
            # snn_weights_norm = np.array(ann_weights_norm)
            # max_weight_value = np.max(np.abs(snn_weights_norm))
            # snn_weights_norm = snn_weights_norm / max_weight_value * self.config.getfloat('initial', 'w_mag')

            layer.set_weights([ann_weights_norm])

          
    def get_activations_layer(self, layer_in, layer_out, x, batch_size=None, path=None, normalize=True):
        
        # Set to 10 if batch_size is not specified
        if batch_size is None:
            batch_size = 10
        
        # If input sample x and batch_size are divided and the remainder is nonzero
        if len(x) % batch_size != 0:
            # Delete the remainder divided by input sample list x
            x = x[: -(len(x) % batch_size)]
        
        print("Calculating activations of layer {}.".format(layer_out.name))
        # Calculate the activation of the corresponding layer neuron by putting 
        # an input sample in the predict function
        activations = tf.keras.models.Model(inputs=layer_in.input, 
                                            outputs=layer_out.output).predict(x, batch_size)
        
        # Save activations as an npz file
        if normalize:
            print("Writing activations to disk.")
            if path is not None:
                np.savez_compressed(os.path.join(path, f'activation_{layer_out.name}.npz'), activations)
        
        
        return np.array(activations)    
