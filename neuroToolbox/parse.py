import os
import sys

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import tensorflow as tf
from tensorflow import keras
import numpy as np

class Parser:
    def __init__(self, input_model, config):
        self.input_model = input_model
        self.config = config
        self.add_layer_mapping = {}

    def parse(self):
        
        layers = self.input_model.layers
        convertible_layers = eval(self.config.get('restrictions', 'convertible_layers'))
        flatten_added = False
        afterParse_layer_list = []

        
        print("\n\n####### parsing input model #######\n\n")

        for i, layer in enumerate(layers):
            
            layer_type = layer.__class__.__name__
            print("\n current parsing layer... layer type : ", layer_type)

            # Check for bias in the layer before parse
            if hasattr(layer, 'use_bias') and layer.use_bias:
                raise ValueError("Layer {} has bias enabled. Please set use_bias=False for all layers.".format(layer_type))


            if isinstance(layer, tf.keras.layers.BatchNormalization): 
                
                # Get BN parameter
                BN_parameters = list(self._get_BN_parameters(layer))
                gamma, mean, var, var_eps_sqrt_inv, axis = BN_parameters

                # Find the nearest Conv2D layer by iterating backward
                prev_layer = layers[i - 1]
                while not isinstance(prev_layer, tf.keras.layers.Conv2D) and i > 0:
                    i -= 1
                    prev_layer = layers[i - 1]

                if not isinstance(prev_layer, tf.keras.layers.Conv2D):
                    print("Skipping BatchNormalization layer because no previous Conv2D layer was found.")
                    continue
                
                # Absorb the BatchNormalization parameters into the previous layer's weights and biases
                weight = prev_layer.get_weights()[0] # Only Weight, No bias
                print("get weight...")

                new_weight = self._absorb_bn_parameters(weight, gamma, mean, var_eps_sqrt_inv, axis)

                # Set the new weight and bias to the previous layer
                print("Set Weight with Absorbing BN params")                
                prev_layer.set_weights([new_weight])
                
                # Remove the current layer (which is a BatchNormalization layer) from the afterParse_layers
                print("remove BatchNormalization Layer in layerlist")
                
                continue
            
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                raise ValueError("MaxPooling2D layer detected. Please replace all MaxPooling2D layers with AveragePooling2D layers and retrain your model.")
                  
            
            elif isinstance(layer, tf.keras.layers.Add):
                print("Replace Add layer to concatenate + Conv2D layer")
                # Retrieve the input tensors for the Add layer
                add_input_tensors = layer.input
            
                # Create a concatenate layer but don't call it yet
                concat_layer = tf.keras.layers.Concatenate()
                afterParse_layer_list.append((concat_layer, add_input_tensors))
            
                # Create a Conv2D layer but don't call it yet
                conv2d_layer = tf.keras.layers.Conv2D(filters=add_input_tensors[0].shape[-1], kernel_size=(1, 1), strides=(1,1), padding='same', activation='relu')
                afterParse_layer_list.append(conv2d_layer)
                continue
            
            elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
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
            
            
            elif isinstance(layer, tf.keras.layers.Flatten):
                # If a Flatten layer is encountered, set the flag to True
                flatten_added = True
                print("Encountered Flatten layer.")
                continue         

               
            elif layer_type not in convertible_layers:
                print("Skipping layer {}.".format(layer_type))
                
                continue

           
            afterParse_layer_list.append(layer)
        

        parsed_model = self.build_parsed_model(afterParse_layer_list)

        return parsed_model
    
    def build_parsed_model(self, layer_list):
       
        print("\n###### build parsed model ######\n")
        print("afterParse layer list : ", layer_list)
        x = layer_list[0].input
    
        for layer in layer_list[1:]:
            x = layer(x)
        
        model = tf.keras.models.Model(inputs=layer_list[0].input, outputs=x, name="parsed_model")
      
        return model

      
    def _get_BN_parameters(self, layer):
        
        print("get BN parameters...")

        axis = layer.axis
        if isinstance(axis, (list, tuple)):
            assert len(axis) == 1, "Multiple BatchNorm axes not understood."
            axis = axis[0]

        
        mean = keras.backend.get_value(layer.moving_mean)
        #print("get.. mean : \n", mean)

        var = keras.backend.get_value(layer.moving_variance)
        #print("get.. var : \n", var)

        var_eps_sqrt_inv = 1 / np.sqrt(var + layer.epsilon)
        #print("get.. var_eps_sqrt_inv : \n", var_eps_sqrt_inv)

        gamma = keras.backend.get_value(layer.gamma)
        #print("get.. gamma : \n", gamma)

        #beta = keras.backend.get_value(layer.beta)
        #print("Beta : ", beta)
    
        return  gamma, mean, var, var_eps_sqrt_inv, axis

    def _absorb_bn_parameters(self, weight, gamma, mean, var_eps_sqrt_inv, axis):

        axis = weight.ndim + axis if axis < 0 else axis


        if weight.ndim == 4:  # Conv2D

            channel_axis = 3
            layer2kernel_axes_map = [None, 0, 1, channel_axis]
            axis = layer2kernel_axes_map[axis]
        
        broadcast_shape = [1] * weight.ndim
        broadcast_shape[axis] = weight.shape[axis]

        #print("before reshape... var_eps_sqrt_inv : \n", var_eps_sqrt_inv)
        var_eps_sqrt_inv = np.reshape(var_eps_sqrt_inv, broadcast_shape)
        #print("After reshape... var_eps_sqrt_inv : \n", var_eps_sqrt_inv)

        #print("before reshape... gamma : \n", gamma)
        gamma = np.reshape(gamma, broadcast_shape)
        #print("After reshape... gamma : \n", gamma)

        #beta = np.reshape(beta, broadcast_shape)
        #print("beta : ", beta)

        #print("before reshape... mean : \n", mean)
        mean = np.reshape(mean, broadcast_shape)
        #print("After reshape... mean : \n", mean)

        # new_bias = np.ravel(beta + (bias - mean) * gamma * var_eps_sqrt_inv)

        #print("before absorb... weight : \n", weight)
        new_weight = weight * gamma * var_eps_sqrt_inv
        #print("After absorb... weight (new_weight): \n", new_weight)
        
        # Calculation by loop
        '''
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
            print("BN parameter absorption is not properly implemented.")

        '''
        return new_weight

def evaluate(model, config):

    # Print the output of the BatchNormalization layer
    conv2D_absorbBN_output_model = keras.Model(inputs=model.input, outputs=model.layers[0].output)  # Assuming BatchNormalization is the second layer

    x_train = None

    x_train_file = np.load(os.path.join(config['paths']['path_wd'], 'x_train.npz'))
    x_train = x_train_file['arr_0']  # Access the data stored in the .npz file

    sample_input = x_train[:1]
    conv2D_absorbBN_output = conv2D_absorbBN_output_model.predict(sample_input)
    print("Output of the Conv2D (Absorb BN params) layer:")
    print(conv2D_absorbBN_output)

    x_test_file = np.load(os.path.join(config["paths"]["path_wd"], 'x_test.npz'))
    x_test = x_test_file['arr_0']
    y_test_file = np.load(os.path.join(config["paths"]["path_wd"], 'y_test.npz'))
    y_test = y_test_file['arr_0']

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])
    score = model.evaluate(x_test, y_test, verbose=0)

    return score
