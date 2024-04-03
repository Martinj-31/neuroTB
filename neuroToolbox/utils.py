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


def weightFormat(weights, format='FP32'):
    if 'FP32' == format:
        weights = weights
    elif 'FP8' == format:
        weights = toFloat34(weights)
    elif 'INT8' == format:
        weights = toInt8(weights)
    else: pass
    
    return weights


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


def neuron_model(spikes, weights, threshold, refractory, layer_name, synapse, precision, bias_flag, clip=True):
    spikes = get_weighted_sum(spikes, weights, precision)
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
    
    if 'pooling' in layer_name:
        spikes = spikes / threshold
    else: spikes = (spikes / threshold) / ((spikes / threshold)*refractory + 1)
    
    if clip:
        spikes = np.floor(spikes)
    else: pass
    
    return spikes


def get_weighted_sum(spikes, w, precision='FP32'):
    if precision == 'FP8':
        spikes = np.tile(spikes, (w.shape[1], 1))
        spikes = np.multiply(spikes, w.T)
        spikes = toFloat34(spikes)
        spikes = np.sum(spikes, axis=1)
    elif precision == 'FP32':
        spikes = np.dot(spikes, w)
    elif precision == 'INT8':
        spikes = np.tile(spikes, (w.shape[1], 1))
        spikes = np.multiply(spikes, w.T)
        spikes = toInt8(spikes)
        spikes = np.sum(spikes, axis=1)
    else: pass
    
    return spikes


def data_transfer(input_data, trans_domain, clip=True):
    if trans_domain == 'log':
        transferred_data = np.zeros_like(input_data)
        max_data = np.max(abs(input_data))
        factor = max_data / np.log(max_data + 1)
        pos_idx = np.where(input_data >= 0)
        neg_idx = np.where(input_data < 0)
        if clip:
            transferred_data[pos_idx] = np.floor(np.exp(input_data[pos_idx] / factor)) - 1
            transferred_data[neg_idx] = np.floor(np.exp((-1)*input_data[neg_idx] / factor) * (-1)) + 1
        else:
            transferred_data[pos_idx] = np.exp(input_data[pos_idx] / factor) - 1
            transferred_data[neg_idx] = np.exp((-1)*input_data[neg_idx] / factor) * (-1) + 1
    elif trans_domain == 'linear':
        transferred_data = np.zeros_like(input_data)
        max_data = np.max(abs(input_data))
        factor = max_data / np.log(max_data + 1)
        pos_idx = np.where(input_data >= 0)
        neg_idx = np.where(input_data < 0)
        if clip:
            transferred_data[pos_idx] = np.log(input_data[pos_idx] + 1) * factor
            transferred_data[neg_idx] = (-1) * np.log((-1) * (input_data[neg_idx] - 1)) * factor
        else:
            transferred_data[pos_idx] = np.log(input_data[pos_idx] + 1) * factor
            transferred_data[neg_idx] = (-1) * np.log((-1) * (input_data[neg_idx] - 1)) * factor
    else: transferred_data = input_data
    
    return transferred_data


def toFloat34(before_value):
    reference = np.array([  0.      ,   1.0625   ,   1.125   ,   1.1875   ,   1.25    ,   1.3125   ,
                            1.375   ,   1.4375   ,   1.5     ,   1.5625   ,   1.625   ,   1.6875   ,
                            1.75    ,   1.8125   ,   1.875   ,   1.9375   ,   2.      ,   2.125    ,
                            2.25    ,   2.375    ,   2.5     ,   2.625    ,   2.75    ,   2.875    ,
                            3.      ,   3.125    ,   3.25    ,   3.375    ,   3.5     ,   3.625    ,
                            3.75    ,   3.875    ,   4.      ,   4.25     ,   4.5     ,   4.75     ,
                            5.      ,   5.25     ,   5.5     ,   5.75     ,   6.      ,   6.25     ,
                            6.5     ,   6.75     ,   7.      ,   7.25     ,   7.5     ,   7.75     ,
                            8.      ,   8.5      ,   9.      ,   9.5      ,   10.     ,   10.5     ,
                            11.     ,   11.5     ,   12.     ,   12.5     ,   13.     ,   13.5     ,
                            14.     ,   14.5     ,   15.     ,   15.5     ,   16.     ,   17.      ,
                            18.     ,   19.      ,   20.     ,   21.      ,   22.     ,   23.      ,
                            24.     ,   25.      ,   26.     ,   27.      ,   28.     ,   29.      ,
                            30.     ,   31.      ,   32.     ,   34.      ,   36.     ,   38.      ,
                            40.     ,   42.      ,   44.     ,   46.      ,   48.     ,   50.      ,
                            52.     ,   54.      ,   56.     ,   58.      ,   60.     ,   62.      ,
                            64.     ,   68.      ,   72.     ,   76.      ,   80.     ,   84.      ,
                            88.     ,   92.      ,   96.     ,   100.     ,   104.    ,   108.     ,
                            112.    ,   116.     ,   120.    ,   124.     ,   128.    ,   136.     ,
                            144.    ,   152.     ,   160.    ,   168.     ,   176.    ,   184.     ,
                            192.    ,   200.     ,   208.    ,   216.     ,   224.    ,   232.     ,
                            240.    ,   248.                                                          ])

    before_value_array = np.atleast_1d(before_value).astype(float)
    sign_array = np.sign(before_value_array)
    abs_before_value = np.abs(before_value_array)
    
    indices = np.searchsorted(reference, abs_before_value, side="left")
    
    indices[indices == len(reference)] = len(reference) - 1
    lower_bound_diff = abs_before_value - reference[indices - 1]
    upper_bound_diff = reference[indices] - abs_before_value

    indices[lower_bound_diff <= upper_bound_diff] -= 1

    result = sign_array * reference[indices]
    
    return result if isinstance(before_value, np.ndarray) else result[0]


def toInt8(before_value):
    
    reference = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 
        70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 
        120, 121, 122, 123, 124, 125, 126, 127
    ])
    
    before_value_array = np.atleast_1d(before_value).astype(float)
    sign_array = np.sign(before_value_array)
    abs_before_value = np.abs(before_value_array)
    
    indices = np.searchsorted(reference, abs_before_value, side="left")
    
    indices[indices == len(reference)] = len(reference) - 1
    lower_bound_diff = abs_before_value - reference[indices - 1]
    upper_bound_diff = reference[indices] - abs_before_value

    indices[lower_bound_diff <= upper_bound_diff] -= 1

    result = sign_array * reference[indices]
    
    return result if isinstance(before_value, np.ndarray) else result[0]