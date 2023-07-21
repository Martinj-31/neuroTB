# parse.py
import tensorflow as tf
import tensorflow.keras.backend as k
import numpy as np

class Parser:
    def __init__(self, input_model, config):
        self.input_model = input_model
        self.config = config
        self.beforeParse_layers = []
        self.afterParse_layers = []
        
    def parse(self):
        layers = self.input_model.layers
        
        print("\n\n####### parsing input model #######\n\n")

        for i, layer in enumerate(layers):
            
            self.beforeParse_layers.append(layer)
            self.afterParse_layers.append(layer)
            
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                
                # Get BN parameter #
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
                self.afterParse_layers.pop()
                print("remove BatchNormalization Layer in layerlist")
                
            if isinstance(layer, tf.keras.layers.Dropout):
                
                self.afterParse_layers.pop()
                print("remove Dropout Layer in layerlist")
        
        for i, layer in enumerate(self.afterParse_layers):
            print(f"Layer {i} ({layer.name}):")
            print(f"  Input shape: {layer.input_shape}")
            print(f"  Output shape: {layer.output_shape}")
                        

                
        print("\n\n beforeParse layer name list : ", [layer.name for layer in self.beforeParse_layers])    
        print("\n\n afterParse layer name list : ", [layer.name for layer in self.afterParse_layers])
        
        
        parsed_model = self.build_parsed_model()
        
        return parsed_model
    
    def build_parsed_model(self):
        input_layer = tf.keras.layers.Input(shape=self.afterParse_layers[0].input_shape[0][1:])
        x = input_layer
    
        for layer in self.afterParse_layers[1:]:
            x = layer(x)
    
        parsed_model = tf.keras.models.Model(inputs=input_layer, outputs=x, name="parsed_model")
        return parsed_model

      
    def _get_BN_parameters(self, layer):
        mean = k.get_value(layer.moving_mean)
        var = k.get_value(layer.moving_variance)
        var_eps_sqrt_inv = 1 / np.sqrt(var + layer.epsilon)
        gamma = k.get_value(layer.gamma)
        beta = k.get_value(layer.beta)
    
        return  gamma, beta, mean, var, var_eps_sqrt_inv

    
    def _get_weight_bias(self, layer):
        # Get the weight and bias of the layer
        return layer.get_weights()
    
    def _set_weight_bias(self, layer, weight, bias):
        # Set the new weight and bias to the layer
        layer.set_weights([weight, bias])
        
    def _absorb_bn_parameters(self, weight, bias, mean, var_eps_sqrt_inv, gamma, beta):
    
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
            print("BN parameter is properly absorbed.")
        else:
            raise NotImplementedError("BN parameter absorption is not properly implemented.")
        ##########################################################################
     
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