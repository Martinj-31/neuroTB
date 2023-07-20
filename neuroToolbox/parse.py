# parse.py
import tensorflow as tf
import tensorflow.keras.backend as k
import numpy as np

class Parser:
    def __init__(self, input_model, config):
        self.input_model = input_model
        self.config = config
        
    def parse(self):
        layers = self.input_model.layers
        
        print("#### parsing input model ####")

        for i, layer in enumerate(layers):
            # Check if the layer is a BatchNormalization layer
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                
                # Get BN parameter #
                BN_parameters = list(self._get_BN_parameters(layer))
                gamma, beta, mean, var, var_eps_sqrt_inv = BN_parameters
                
                print("gamma : ", gamma)
                print("beta : ", beta)
                print("mean : ", mean)
                print("varience : ", var)
                
                # Get the previous layer
                prev_layer = layers[i - 1]
                
                # Absorb the BatchNormalization parameters into the previous layer's weights and biases
                weight, bias = self._get_weight_bias(prev_layer)
                
                print("weight : ", weight)
                print("bias : ", bias)
                new_weight, new_bias = self._absorb_bn_parameters(weight, bias, gamma, beta, mean, var_eps_sqrt_inv)

                # Set the new weight and bias to the previous layer
                self._set_weight_bias(prev_layer, new_weight, new_bias)
      
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
    
        ###### Calculation by Numpy :  Area where BN parameters abosrb ######
        
        new_weight = weight * gamma * var_eps_sqrt_inv
        new_bias = beta + (bias - mean) * gamma * var_eps_sqrt_inv
        
        ############## Calculation by loop ##############
        
        weight_bn_loop = np.zeros_like(weight)
        bias_bn_loop = np.zeros_like(bias)
        
        for i in range(weight.shape[0]):
            for j in range(weight.shape[1]):
                for k in range(weight.shape[2]):
                    for l in range(weight.shape[3]):
                        weight_bn_loop[i, j, k, l] = weight[i, j, k, l] * gamma[l] * var_eps_sqrt_inv[l]
                        bias_bn_loop[l] = beta[l] + (bias[l] - mean[l]) * gamma[l] * var_eps_sqrt_inv[l]
    
        
        ############################### evaluation ###############################
        weight_eval_arr = new_weight - weight_bn_loop
        bias_eval_arr = new_bias - bias_bn_loop
       
        print("\n\n weight_eval_arr : \n\n", weight_eval_arr)
        print("\n\n bias_eval_arr : \n\n", bias_eval_arr)
    
        if np.all(weight_eval_arr == 0) and np.all(bias_eval_arr == 0) :
            print("BN parameter is properly absorbed.")
        else:
            raise NotImplementedError("BN parameter absorption is not properly implemented.")
        ##########################################################################
     
        return new_weight, new_bias
