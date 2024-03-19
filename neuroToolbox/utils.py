import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


def get_inbound_layers_with_params(layer):
    """
        Retrieve inbound layers with weights for a given layer.
        
        Parameters:
        - layer (tf.keras.layers.Layer): The target layer for which to find inbound layers.
        
        Returns:
        - list: A list of layers that are predecessors of the target layer and have weights.
        
        This method recursively searches for inbound layers of the target layer that have weights (excluding BatchNormalization layers) and returns a list of such layers.
        """
    inbound = layer
    prev_layer = None
    # Repeat when a layer with weight exists
    while True:
        inbound = get_inbound_layers(inbound)

        if isinstance(inbound[0], tf.keras.layers.InputLayer):
            inbound = inbound[0]
            return[inbound]
        
        elif len(inbound) == 1 and not isinstance(inbound[0], 
                                                tf.keras.layers.BatchNormalization):
            inbound = inbound[0]
            if has_weights(inbound):
                return [inbound]
            else:
                if isinstance(inbound, tf.keras.layers.AveragePooling2D):
                    return [inbound]
                else:
                    pass
            
        # If there is no layer information
        # In the case of input layer, the previous layer does not exist, 
        # so it is empty list return
        else:
            result = []
            for inb in inbound:

                if isinstance(inb, tf.keras.layers.BatchNormalization):
                    prev_layer = get_inbound_layers_with_params(inb)[0]
                        
                if has_weights(inb):
                    result.append(inb)
                    
                else:
                    result += get_inbound_layers_with_params(inb)
                    
            if prev_layer is not None:
                return [prev_layer]
            
            else:
                return result
            

def get_inbound_layers(layer):
    """
        Get inbound layers of a given layer.
    
        Parameters:
        - layer (tf.keras.layers.Layer): The target layer for which to retrieve inbound layers.
    
        Returns:
        - list: A list of inbound layers connected to the target layer.
    
        This method retrieves the inbound layers connected to the target layer and returns them as a list.
        """
    
    # Check the previous layer of that layer through _inbound_nodes
    inbound_layers = layer._inbound_nodes[0].inbound_layers
    
    if not isinstance(inbound_layers, (list, tuple)):
        inbound_layers = [inbound_layers]
        
    return inbound_layers


def has_weights(layer):
    """
        Check if a layer has trainable weights.
    
        Parameters:
        - layer (tf.keras.layers.Layer): The layer to check for trainable weights.
    
        Returns:
        - bool: True if the layer has trainable weights, False otherwise.
    
        This method checks whether a layer has trainable weights and returns True if it has weights, excluding BatchNormalization layers.
        """
    
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        return False
    
    else:    
        return len(layer.weights)
    

def get_percentile_activation(activations, percentile):

    return np.percentile(activations, percentile) if activations.size else 1


def weightDecompile(synapses):
    weight_list = {}
    synCnt = 0
    for layer, synapse in synapses.items():
        src = np.array(synapse[0]) - synCnt
        synCnt += 1024
        tar = np.array(synapse[1]) - synCnt
        w = np.array(synapse[2])
        source = len(np.unique(src))
        target = len(np.unique(tar))
        weights = np.zeros(source * target).reshape(source, target)
        
        for i in range(len(w)):
            weights[src[i]][tar[i]] = w[i]
        
        weight_list[layer] = weights

    return weight_list


def Activation_Flattener(loaded_activations, layer_name):
    if 'conv' in layer_name:
        acts = []
        for ic in range(loaded_activations.shape[0]):
            for oc in range(loaded_activations.shape[-1]):
                acts = np.concatenate((acts, loaded_activations[ic, :, :, oc].flatten()))
    elif 'pooling' in layer_name:
        acts = []
        for ic in range(loaded_activations.shape[0]):
            for oc in range(loaded_activations.shape[-1]):
                for i in range(loaded_activations.shape[1]):
                    for j in range(loaded_activations.shape[2]):
                        acts = np.concatenate((acts, loaded_activations[ic, i, j, oc].flatten()))
    elif 'dense' in layer_name:
        acts = []
        for ic in range(loaded_activations.shape[0]):
            for oc in range(loaded_activations.shape[-1]):
                acts = np.concatenate((acts, loaded_activations[ic, oc].flatten()))
    else: pass
    
    return acts


def Input_Activation(input_activations, layer_name):
    if 'conv' in layer_name:
        input_acts = []
        for ic in range(input_activations.shape[0]):
            temp = []
            for oc in range(input_activations.shape[-1]):
                temp = np.concatenate((temp, input_activations[ic, :, :, oc].flatten()))
            input_acts.append(temp)
    elif 'pooling' in layer_name:
        input_acts = []
        for ic in range(input_activations.shape[0]):
            temp = []
            for oc in range(input_activations.shape[-1]):
                for i in range(input_activations.shape[1]):
                    for j in range(input_activations.shape[2]):
                        temp = np.concatenate((temp, input_activations[ic, i, j, oc].flatten()))
            input_acts.append(temp)
    elif 'dense' in layer_name:
        input_acts = []
        for ic in range(input_activations.shape[0]):
            temp = []
            for oc in range(input_activations.shape[-1]):
                temp = np.concatenate((temp, input_activations[ic, oc].flatten()))
            input_acts.append(temp)
    elif 'input' in layer_name:
        input_acts = []
        for ic in range(input_activations.shape[0]):
            temp = []
            for oc in range(input_activations.shape[-1]):
                temp = np.concatenate((temp, input_activations[ic, :, :, oc].flatten()))
            input_acts.append(temp)
    else: pass
    
    return np.array(input_acts)


def neuron_model(spikes, weights, threshold, refractory, layer_name, synapse, bias_flag):
    spikes = np.dot(spikes, weights)
    if bias_flag:
        if 'conv' in layer_name:
            s = 0
            for oc_idx, oc in enumerate(synapse[4]):
                spikes[s:oc] = (spikes[s:oc] + synapse[3][oc_idx])
                s = oc + 1
        elif 'dense' in layer_name:
            spikes = (spikes + synapse[3])
        else:
            spikes = spikes
    else: pass
    
    neg_idx = np.where(spikes < 0)[0]
    spikes[neg_idx] = 0
    spikes = np.floor((spikes / threshold) / ((spikes / threshold)*refractory + 1))
    
    return spikes


def log_transfer(input_data, input_trans):
    if input_trans == 'log':
        input_data = input_data / np.max(input_data)
        input_data = np.floor((np.exp(input_data) - 1) / np.max(np.exp(input_data) - 1) * 255)
    else: input_data = input_data
    
    return input_data


def spikeGen(input_spikes, neurons, duration, delta_t):
    
    return