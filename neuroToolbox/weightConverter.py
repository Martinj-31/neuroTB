import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


class Normalize:
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        self.threshold = {}

    def normalize_parameter(self):
        
        activation_dir = os.path.join(self.config['paths']['path_wd'], 'activations')
        os.makedirs(activation_dir, exist_ok=True)
        
        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']  # Access the data stored in the .npz file
        
        print("\n\n######## Weight conversion ########\n\n")
        
        # Declare and initialize variables
        batch_size = self.config.getint('initial', 'batch_size')

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
            perc = self.config.getfloat('initial', 'percentile')
            
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
            inbound = self.get_inbound_layers_with_params(layer)

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
    

    # Activation return corresponding to n-th percentile
    def get_percentile_activation(self, activations, percentile):

        return np.percentile(activations, percentile) if activations.size else 1

    
    def get_inbound_layers_with_params(self, layer):
        
        inbound = layer
        prev_layer = None
        # Repeat when a layer with weight exists
        while True:
            inbound = self.get_inbound_layers(inbound)
            
            if len(inbound) == 1 and not isinstance(inbound[0], 
                                                    tf.keras.layers.BatchNormalization):
                inbound = inbound[0]
                if self.has_weights(inbound):
                    return [inbound]
                
            # If there is no layer information
            # In the case of input layer, the previous layer does not exist, 
            # so it is empty list return
            else:
                result = []
                for inb in inbound:

                    if isinstance(inb, tf.keras.layers.BatchNormalization):
                        prev_layer = self.get_inbound_layers_with_params(inb)[0]
                            
                    if self.has_weights(inb):
                        result.append(inb)
                        
                    else:
                        result += self.get_inbound_layers_with_params(inb)
                        
                if prev_layer is not None:
                    return [prev_layer]
                
                else:
                    return result
    
    def get_inbound_layers(self, layer):
        
        # Check the previous layer of that layer through _inbound_nodes
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        
        if not isinstance(inbound_layers, (list, tuple)):
            inbound_layers = [inbound_layers]
            
        return inbound_layers
        
    
    def has_weights(self, layer):
        
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            return False
        
        else:    
            return len(layer.weights)
        
        
    def balThreshold(self, shift_params):
        activation_dir = os.path.join(self.config['paths']['path_wd'], 'parsed_model_activations')

        shift_idx = 0
        for i, layer in enumerate(self.model.layers):
            if 'conv' in layer.name:
                prev_layer = self.model.layers[i-1]
                input_file = np.load(os.path.join(activation_dir, f"parsed_model_activation_{prev_layer.name}.npz"))
                input_act = input_file['arr_0']
                
                activations = tf.keras.models.Model(inputs=layer.input, outputs=layer.output).predict(input_act)
                vth_list = []
                for oc in range(activations.shape[-1]):
                    vth = np.max(activations[:, :, :, oc]) / (np.max(activations[:, :, :, oc]) + shift_params[shift_idx][oc])
                    vth_list.append(vth)
                self.threshold[layer.name] = vth_list
                shift_idx += 1
            else:
                self.threshold[layer.name] = 1

        return self.threshold