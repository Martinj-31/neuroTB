import tensorflow as tf
from tensorflow import keras
import numpy as np

class Parser:
    def __init__(self, input_model, config):
        self.input_model = input_model
        self.config = config
        self.afterParse_layer_list = []
        
    def parse(self):
        
        """
        Parse a Keras model and return a model that is suitable for conversion to an SNN.
        
        This function goes through the layers of the input model and performs the following operations:
        - Absorbs BN parameters from BatchNormalization layers into the previous layer's weights and biases, and removes the BatchNormalization layer.
        - Replaces GlobalAveragePooling2D layers with an AveragePooling2D layer and a Flatten layer.
        - Inserts a Flatten layer before the first Dense layer if no Flatten layer has been added yet.
        - Skips layers not defined as convertible in the config file.
        - Appends all other layers to the parsed model.
    
        Returns
        -------
        parsed_model : keras.Model
            The parsed Keras model. This model has the same architecture as the input model, except that:
            - BatchNormalization layers have been removed.
            - GlobalAveragePooling2D layers have been replaced by an AveragePooling2D layer and a Flatten layer.
            - A Flatten layer has been inserted before the first Dense layer if no Flatten layer was in the original model.
        """
        
        layers = self.input_model.layers
        convertible_layers = eval(self.config.get('restrictions', 'convertible_layers'))
        flatten_added = False 
        print("\n\n####### parsing input model #######\n\n")

        for i, layer in enumerate(layers):
            
            layer_type = layer.__class__.__name__
            print("\n current... layer type : ", layer_type)
            if isinstance(layer, tf.keras.layers.BatchNormalization): #
                
                # Get BN parameter
                BN_parameters = list(self._get_BN_parameters(layer))
                gamma, beta, mean, var, var_eps_sqrt_inv = BN_parameters
                
                # Get the previous layer
                prev_layer = layers[i - 1]
                
                # Absorb the BatchNormalization parameters into the previous layer's weights and biases
                weight, bias = self._get_weight_bias(prev_layer)
                
                new_weight, new_bias = self._absorb_bn_parameters(weight, bias, gamma, beta, mean, var_eps_sqrt_inv)

                # Set the new weight and bias to the previous layer
                self._set_weight_bias(prev_layer, new_weight, new_bias)
                
                # Remove the current layer (which is a BatchNormalization layer) from the afterParse_layers
                print("remove BatchNormalization Layer in layerlist")
                continue
            
            elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                # Replace GlobalAveragePooling2D layer with AveragePooling2D plus Flatten layer
                
                # Get the spatial dimensions of the input tensor
                spatial_dims = layer.input_shape[1:-1]  # Exclude the batch and channel dimensions
            
                # Create an AveragePooling2D layer with the same spatial dimensions as the input tensor
                avg_pool_layer = tf.keras.layers.AveragePooling2D(name=layer.name + "_avg",pool_size=spatial_dims)
                self.afterParse_layer_list.append(avg_pool_layer)
                flatten_layer = tf.keras.layers.Flatten(name=layer.name + "_flatten")
                self.afterParse_layer_list.append(flatten_layer)
                flatten_added = True
                print("Replaced GlobalAveragePooling2D layer with AveragePooling2D and Flatten layer.")
                
                continue
            
            elif isinstance(layer, tf.keras.layers.Flatten):
                # If a Flatten layer is encountered, set the flag to True
                flatten_added = True
                print("Encountered Flatten layer.")
                
            elif isinstance(layer, tf.keras.layers.Dense) and not flatten_added:
                # If a Dense layer is encountered and no Flatten layer has been encountered yet,
                # insert a Flatten layer and set the flag to True
                print("flatten added : ", flatten_added)
                flatten_layer = tf.keras.layers.Flatten()
                self.afterParse_layer_list.append(flatten_layer)
                flatten_added = True
                print("Added Flatten layer before Dense layer.")
                
            elif layer_type not in convertible_layers:
                print("Skipping layer {}.".format(layer_type))
                continue
           
            self.afterParse_layer_list.append(layer)
        
    
        parsed_model = self.build_parsed_model()
        
        return parsed_model
    
    def build_parsed_model(self):
    
        """
       Construct the parsed Keras model based on the `afterParse_layer_list`.
       
       This function iterates over the list of layers stored in `afterParse_layer_list` 
       that were modified in the parsing process and constructs a Keras model from them. 
       The first layer of the model is an Input layer with the same shape as the input of the original model.
       The remaining layers are the layers in the `afterParse_layer_list`, connected sequentially.
    
       Returns
       -------
       parsed_model : keras.Model
           The parsed Keras model. This model is ready to be converted to a Spiking Neural Network (SNN).
       """
       
        print("\n###### build parsed model ######\n")
        
        x = self.afterParse_layer_list[0].input
    
        for layer in self.afterParse_layer_list[1:]:
            x = layer(x)
    
        parsed_model = tf.keras.models.Model(inputs=self.afterParse_layer_list[0].input, outputs=x, name="parsed_model")
        return parsed_model

      
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
            A tuple containing gamma (scale parameter), beta (offset parameter), mean (moving mean), 
            var (moving variance), and var_eps_sqrt_inv (inverse of the square root of the variance + epsilon).
        """
        
        mean = keras.backend.get_value(layer.moving_mean)
        var = keras.backend.get_value(layer.moving_variance)
        var_eps_sqrt_inv = 1 / np.sqrt(var + layer.epsilon)
        gamma = keras.backend.get_value(layer.gamma)
        beta = keras.backend.get_value(layer.beta)
    
        return  gamma, beta, mean, var, var_eps_sqrt_inv

    
    def _get_weight_bias(self, layer):
        
        """
        Get the weights and biases of a layer.
    
        Parameters
        ----------
        layer : keras.layers.Layer
            The layer to extract weights and biases from.
    
        Returns
        -------
        list
            A list where the first element is the weight array and the second element is the bias array.
        """
        
        # Get the weight and bias of the layer
        return layer.get_weights()
    
    def _set_weight_bias(self, layer, weight, bias):
        
        """
        Set the weights and biases of a layer.
        
        Parameters
        ----------
        layer : keras.layers.Layer
            The layer to set weights and biases.
        weight : np.array
            The new weight array to set.
        bias : np.array
            The new bias array to set.
        """
        
        # Set the new weight and bias to the layer
        layer.set_weights([weight, bias])
        
    def _absorb_bn_parameters(self, weight, bias, mean, var_eps_sqrt_inv, gamma, beta):
        
        """
        Absorb the BN parameters of a BatchNormalization layer into the weights and biases of the previous layer.
    
        Parameters
        ----------
        weight : np.array
            The weight array of the previous layer.
        bias : np.array
            The bias array of the previous layer.
        mean : np.array
            The moving mean from the BatchNormalization layer.
        var_eps_sqrt_inv : np.array
            The inverse of the square root of the variance plus a small constant for numerical stability.
        gamma : np.array
            The scale parameter from the BatchNormalization layer.
        beta : np.array
            The offset parameter from the BatchNormalization layer.
    
        Returns
        -------
        tuple
            A tuple containing the 'new weight' and 'new bias' arrays after absorption of the BatchNormalization parameters.
        """
    
        # Calculation by Numpy :  Area where BN parameters abosrb
        
        new_weight = weight * gamma * var_eps_sqrt_inv
        new_bias = beta + (bias - mean) * gamma * var_eps_sqrt_inv
        
        # Calculation by loop
        
        weight_bn_loop = np.zeros_like(weight)
        bias_bn_loop = np.zeros_like(bias)
        
        for i in range(weight.shape[0]):
            for j in range(weight.shape[1]):
                for k in range(weight.shape[2]):
                    for l in range(weight.shape[3]):
                        weight_bn_loop[i, j, k, l] = weight[i, j, k, l] * gamma[l] * var_eps_sqrt_inv[l]
                        bias_bn_loop[l] = beta[l] + (bias[l] - mean[l]) * gamma[l] * var_eps_sqrt_inv[l]
    
        
        # evaluation
        weight_eval_arr = new_weight - weight_bn_loop
        bias_eval_arr = new_bias - bias_bn_loop
    
        if np.all(weight_eval_arr == 0) and np.all(bias_eval_arr == 0) :
            print("BN parameter is properly absorbed into previous layer.")
        else:
            raise NotImplementedError("BN parameter absorption is not properly implemented.")

        
        return new_weight, new_bias
    
    '''
    def print_layer_connections(self):
        # Iterate over the layers in the model
        for i in range(len(self.input_model.layers)-1):
            # Get current layer and next layer
            current_layer = self.input_model.layers[i]
            next_layer = self.input_model.layers[i+1]
            
            # Print the connection
            print(f"{current_layer.name} is connected to {next_layer.name}")
    '''