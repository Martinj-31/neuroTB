import os
import sys

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class Parser:
    """
    Class for parsing an ANN model into a format suitable for ANN to SNN conversion.
    """

    def __init__(self, input_model, config):
        """
        Initialize the Parser instance.
        
        Parameters:
        - input_model (tf.keras.Model): The input model to be parsed.
        - config (configparser.ConfigParser): Configuration settings for parsing.
        
        This class is responsible for parsing a given TensorFlow/Keras model according to specified rules in the configuration file.
        """
        self.input_model = input_model
        self.config = config
        self._parsed_layer_list = []
        
        self.add_layer_mapping = {}

    def parse(self):
        """
        Parse the input model according to the specified rules for ANN to SNN conversion.
    
        Returns:
        - tf.keras.Model: The parsed model with modified layers.
    
        This method parses the input model by applying specific rules for ANN to SNN conversion. The rules include:
    
        1. Absorbing BatchNormalization layers: BatchNormalization layer parameters are absorbed into the previous layer's weights.
        2. Handling MaxPooling2D layers: If MaxPooling2D layers are present, it raises a ValueError since SNNs typically use AveragePooling2D.
        3. Transforming Add layers: Add layers are replaced with a Concatenate layer followed by a Conv2D layer (Note: This transformation is incomplete).
        4. Replacing GlobalAveragePooling2D layers: GlobalAveragePooling2D layers are replaced with an AveragePooling2D layer followed by a Flatten layer.
        5. Ensuring the presence of a Flatten layer: If a Flatten layer is missing in the input model, it raises an error as it's essential for building the parsed model.
        6. Skipping unnecessary layers: Layers not required for SNN modeling, such as Dropout and Activation layers, are skipped.
    
        The method returns the parsed model with the specified modifications.
        """
        
        layers = self.input_model.layers
        convertible_layers = eval(self.config.get('restrictions', 'convertible_layers'))
        flatten_added = False
        afterParse_layer_list = []
        layer_id_map = {}

        
        print("\n\n####### parsing input model #######\n\n")
        
        for i, layer in enumerate(layers):
            
            layer_type = layer.__class__.__name__
            print("\n current parsing layer... layer type : ", layer_type)

            # Check for bias in the layer before parse
            if hasattr(layer, 'use_bias') and layer.use_bias:
                raise ValueError("Layer {} has bias enabled. Please set use_bias=False for all layers.".format(layer_type))


            if  layer_type == 'BatchNormalization': 
                
                # Find the previous layer
                inbound = self.get_inbound_layers_parameters(layer)
                prev_layer = inbound[0]
                print("prev_layer type : ", prev_layer.__class__.__name__)                

                # Get BN parameter
                BN_parameters = list(self._get_BN_parameters(layer))
                
                # print("This is BN params : ", BN_parameters)
                gamma, mean, var, var_eps_sqrt_inv, axis = BN_parameters

                # Absorb the BatchNormalization parameters into the previous layer's weights and biases
                weight = prev_layer.get_weights()[0] # Only Weight, No bias
                print("get weight...")

                new_weight = self._absorb_bn_parameters(weight, gamma, mean, var_eps_sqrt_inv, axis)

                # Set the new weight and bias to the previous layer
                print("Set Weight with Absorbing BN params")                
                prev_layer.set_weights([new_weight])

                # Remove the current layer (BatchNormalization layer) from the afterParse_layers
                print("remove BatchNormalization Layer in layerlist")
                
                continue
            
            elif layer_type == 'MaxPooling2D':
                raise ValueError("MaxPooling2D layer detected. Please replace all MaxPooling2D layers with AveragePooling2D layers and retrain your model.")
                  
            
            elif layer_type == 'Add':
                print("Replace Add layer to concatenate + Conv2D layer")
                # Retrieve the input tensors for the Add layer
                add_input_tensors = layer.input
            
                # Create a concatenate layer
                concat_layer = tf.keras.layers.Concatenate()
                afterParse_layer_list.append((concat_layer, add_input_tensors))
            
                # Create a Conv2D layer
                conv2d_layer = tf.keras.layers.Conv2D(filters=add_input_tensors[0].shape[-1], kernel_size=(1, 1), strides=(1,1), padding='same', activation='relu')
                afterParse_layer_list.append(conv2d_layer)
                continue
            
            elif layer_type == 'GlobalAveragePooling2D':
                # Replace GlobalAveragePooling2D layer with AveragePooling2D plus Flatten layer
                
                # Get the spatial dimensions of the input tensor
                spatial_dims = layer.input_shape[1:-1]  # Exclude the batch and channel dimensions
            
                # Create an AveragePooling2D layer with the same spatial dimensions as the input tensor
                avg_pool_layer = tf.keras.layers.AveragePooling2D(name=layer.name + "_avg",pool_size=spatial_dims)
                afterParse_layer_list.append(avg_pool_layer)
                flatten_layer = tf.keras.layers.Flatten(name=layer.name + "_flatten")
                afterParse_layer_list.append(flatten_layer)
                
                flatten_added = True
                print("Replaced GlobalAveragePooling2D layer with AveragePooling2D and Flatten layer.")
                
                continue
            
            # elif flatten_added == False:
            #     raise ValueError("input model doesn't have flatten layer. please check again")

               
            elif layer_type not in convertible_layers:
                print("Skipping layer {}.".format(layer_type))
                
                continue

           
            afterParse_layer_list.append(layer)
        

        parsed_model = self.build_parsed_model(afterParse_layer_list)
        keras.models.save_model(parsed_model, os.path.join(self.config["paths"]["path_wd"], "parsed_" + self.config["paths"]["filename_ann"] + '.h5'))

        return parsed_model
    
    def build_parsed_model(self, layer_list):
        """
        Build and compile the parsed model.

        Parameters:
        - layer_list (list): List of layers for the parsed model.

        Returns:
        - tf.keras.Model: The compiled parsed model.

        This method constructs the parsed model by connecting the layers provided in the layer_list (afterParse_layer_list)
        """
       
        print("\n###### build parsed model ######\n")
        # input_shape = layer_list[0].input_shape[0][1:]
        # batch_size = self.config.getint('initial', 'batch_size')
        # batch_shape = (batch_size,) + input_shape
        # new_input_layer = keras.layers.Input(batch_shape=batch_shape, name=layer_list[0].name)
        # x = new_input_layer

        x = layer_list[0].input
    
        for layer in layer_list[1:]:
            x = layer(x)

        # model = keras.models.Model(inputs=new_input_layer, outputs=x, name="parsed_model")
        model = keras.models.Model(inputs=layer_list[0].input, outputs=x, name="parsed_model")
        
        model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
        
        return model

      
    def _get_BN_parameters(self, layer):
        """
        Extract the parameters of a BatchNormalization layer.        
        
        Parameters
        ----------
        layer : keras.layers.BatchNormalization
            The BatchNormalization layer to extract parameters from.
        
        Returns
        -------
        tuple
            A tuple containing gamma (scale parameter), mean (moving mean), 
            var (moving variance), var_eps_sqrt_inv (inverse of the square root of the variance + epsilon) and axis
        """
        
        print("get BN parameters...")        

        axis = layer.axis
        if isinstance(axis, (list, tuple)):
            assert len(axis) == 1, "Multiple BatchNorm axes not understood."
            axis = axis[0]

        mean = keras.backend.get_value(layer.moving_mean)


        var = keras.backend.get_value(layer.moving_variance)


        var_eps_sqrt_inv = 1 / np.sqrt(var + layer.epsilon)


        gamma = keras.backend.get_value(layer.gamma)


        # beta = keras.backend.get_value(layer.beta) 

    
        return  gamma, mean, var, var_eps_sqrt_inv, axis

    def _absorb_bn_parameters(self, weight, gamma, mean, var_eps_sqrt_inv, axis):
        """
        Absorb BatchNormalization parameters into previous layer's weights.
        
        Parameters:
        - weight (np.ndarray): The previous layer's weight matrix.
        - gamma (np.ndarray): The gamma parameter of BatchNormalization.
        - mean (np.ndarray): The mean parameter of BatchNormalization.
        - var_eps_sqrt_inv (np.ndarray): The inverse square root of the variance epsilon parameter of BatchNormalization.
        - axis (int): The axis along which to perform the absorption.
        
        Returns:
        - np.ndarray: The modified layer weight matrix after absorbing BatchNormalization parameters.
        
        This method absorbs BatchNormalization parameters (gamma, mean, var_eps_sqrt_inv) into the previous layer's weight matrix.
        It adjusts the weights to incorporate the effect of BatchNormalization, resulting in a modified weight matrix.
        """

        axis = weight.ndim + axis if axis < 0 else axis


        if weight.ndim == 4:  # Conv2D

            channel_axis = 3
            layer2kernel_axes_map = [None, 0, 1, channel_axis]
            axis = layer2kernel_axes_map[axis]
        
        broadcast_shape = [1] * weight.ndim
        broadcast_shape[axis] = weight.shape[axis]

        var_eps_sqrt_inv = np.reshape(var_eps_sqrt_inv, broadcast_shape)
        gamma = np.reshape(gamma, broadcast_shape)
        #beta = np.reshape(beta, broadcast_shape) #beta is shift parameter
        mean = np.reshape(mean, broadcast_shape)
        # new_bias = np.ravel(beta + (bias - mean) * gamma * var_eps_sqrt_inv) # we don't use bias
        new_weight = weight * gamma * var_eps_sqrt_inv
        
        return new_weight
    
    def get_inbound_layers_parameters(self, layer):
        """
        Retrieve inbound layers with weights for a given layer.
        
        Parameters:
        - layer (tf.keras.layers.Layer): The target layer for which to find inbound layers.
        
        Returns:
        - list: A list of layers that are predecessors of the target layer and have weights.
        
        This method recursively searches for inbound layers of the target layer that have weights (excluding BatchNormalization layers) and returns a list of such layers.
        """

        inbound = layer
        while True:
            inbound = self.get_inbound_layers(inbound)
            if len(inbound) == 1:
                inbound = inbound[0]
                if self.has_weights(inbound):
                    return [inbound]
            else:
                result = []
                for inb in inbound:
                    if self.has_weights(inb):
                        result.append(inb)
                    else:
                        result += self.get_inbound_layers_parameters(inb)
        return result
    
    def get_inbound_layers(self, layer):
        """
        Get inbound layers of a given layer.
    
        Parameters:
        - layer (tf.keras.layers.Layer): The target layer for which to retrieve inbound layers.
    
        Returns:
        - list: A list of inbound layers connected to the target layer.
    
        This method retrieves the inbound layers connected to the target layer and returns them as a list.
        """
        
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        
        if not isinstance(inbound_layers, (list, tuple)):
            inbound_layers = [inbound_layers]
            
        return inbound_layers
        
    
    def has_weights(self, layer):
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

    def parseAnalysis(self, model_1, model_2, x_test, y_test):
        """
        Evaluate and compare two models on a given test dataset.

        Parameters:
        - model_1 (tf.keras.Model): The model before parsed 
        - model_2 (tf.keras.Model): The model after parsed
        - x_test (np.ndarray): The test input data.
        - y_test (np.ndarray): The test target data.

        Returns:
        - tuple: A tuple containing two evaluation scores, one for each model.

        This function evaluates two models, `model_1` and `model_2`, on the provided test dataset (`x_test` and `y_test`). It computes evaluation metrics for each model and returns them as a tuple, allowing for comparison between the two models.
        """
        
        score1 = model_1.evaluate(x_test, y_test, verbose=0)
        score2 = model_2.evaluate(x_test, y_test, verbose=0)

        return score1, score2


