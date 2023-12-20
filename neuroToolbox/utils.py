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
    

def Flattener_Dense(loaded_act):
    acts = []
    for ic in range(loaded_act.shape[0]):
        for oc in range(loaded_act.shape[-1]):
            acts = np.concatenate((acts, loaded_act[ic, oc].flatten()))

    return acts


def Flattener_Pooling(loaded_act):
    acts = []
    for ic in range(loaded_act.shape[0]):
        for oc in range(loaded_act.shape[-1]):
            for i in range(loaded_act.shape[1]):
                for j in range(loaded_act.shape[2]):
                    acts = np.concatenate((acts, loaded_act[ic, i, j, oc].flatten()))

    return acts


def Flattener_Conv2D(loaded_act):
    acts = []
    for ic in range(loaded_act.shape[0]):
        for oc in range(loaded_act.shape[-1]):
            acts = np.concatenate((acts, loaded_act[ic, :, :, oc].flatten()))

    return acts


def Input_Dense(input_act):
    input_acts = []
    for ic in range(input_act.shape[0]):
        temp = []
        for oc in range(input_act.shape[-1]):
            temp = np.concatenate((temp, input_act[ic, oc].flatten()))
        input_acts.append(temp)

    return input_acts


def Input_Pooling(input_act):
    input_acts = []
    for ic in range(input_act.shape[0]):
        temp = []
        for oc in range(input_act.shape[-1]):
            for i in range(input_act.shape[1]):
                for j in range(input_act.shape[2]):
                    temp = np.concatenate((temp, input_act[ic, i, j, oc].flatten()))
        input_acts.append(temp)

    return input_acts


def Input_Conv2D(input_act):
    input_acts = []
    for ic in range(input_act.shape[0]):
        temp = []
        for oc in range(input_act.shape[-1]):
            temp = np.concatenate((temp, input_act[ic, :, :, oc].flatten()))
        input_acts.append(temp)
    
    return input_acts


def Input_Image2D(input_act):
    input_acts = []
    for ic in range(input_act.shape[0]):
        temp = []
        for oc in range(input_act.shape[-1]):
            temp = np.concatenate((temp, input_act[ic, :, :, oc].flatten()))
        input_acts.append(temp)

    return input_acts