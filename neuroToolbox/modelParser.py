import os, sys
import numpy as np
import tensorflow as tf

from tensorflow import keras

import neuroToolbox.utils as utils

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


class Parser:
    """
    Class for parsing an ANN model into a format suitable for ANN to SNN conversion.
    """

    def __init__(self, input_model, config, plot=False):
        """
        Initialize the Parser instance.

        Parameters:
        - input_model (tf.keras.Model): The input model to be parsed.
        - config (configparser.ConfigParser): Configuration settings for parsing.
        
        This class is responsible for parsing a given TensorFlow/Keras model according to specified rules in the configuration file.
        """
        self.input_model = input_model
        self.config = config
        self.plot = plot

        self._parsed_layer_list = []
        self.add_layer_mapping = {}
        self.conv_weights = []

        self.input_model_name = config["names"]["input_model"]

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


        print("\n\n####### parsing input model #######\n")
        
        if self.plot == True:
            os.makedirs(self.config["paths"]["path_wd"] + '/model_graph/', exist_ok=True)
            keras.utils.plot_model(self.input_model, os.path.join(self.config["paths"]["path_wd"] + '/model_graph' + '/input_model.png'), show_shapes=True)
        else: pass

        for i, layer in enumerate(layers):
            
            layer_type = layer.__class__.__name__
            print("\n current parsing layer... layer type : ", layer_type)

            # Check for bias in the layer before parse
            # if hasattr(layer, 'use_bias') and layer.use_bias:
            #     raise ValueError("Layer {} has bias enabled. Please set use_bias=False for all layers.".format(layer_type))

            if  layer_type == 'BatchNormalization': 
                
                # Find the previous layer
                inbound = utils.get_inbound_layers_with_params(layer)
                prev_layer = inbound[0]
                print("prev_layer type : ", prev_layer.__class__.__name__)                

                # Get BN parameter
                gamma, mean, var_eps_sqrt_inv, beta, axis = list(self._get_BN_parameters(layer))

                # Absorb the BatchNormalization parameters into the previous layer's weights and biases
                weight, bias = prev_layer.get_weights()
                print("get weight...")

                new_weight, new_bias = self._absorb_bn_parameters(weight, bias, gamma, mean, var_eps_sqrt_inv, beta, axis)

                # Set the new weight and bias to the previous layer
                print("Set Weight with Absorbing BN params")                
                prev_layer.set_weights([new_weight, new_bias])

                # Remove the current layer (BatchNormalization layer) from the afterParse_layers
                print("remove BatchNormalization Layer in layerlist")
                
                continue
            
            elif layer_type == 'MaxPooling2D':
                raise ValueError("MaxPooling2D layer detected. Please replace all MaxPooling2D layers with AveragePooling2D layers and retrain your model.")
                  
            
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
        keras.models.save_model(parsed_model, os.path.join(self.config["paths"]["models"], self.config['names']['parsed_model'] + '.h5'))

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
        
        shortcut = None
        
        for i, layer in enumerate(layer_list[1:]):
            prev_layer = layer_list[i]

            if i == 0:
                x = layer(x)
                shortcut = x
                continue
            
            if isinstance(layer, keras.layers.Add):
                if shortcut is not None:
                    x = keras.layers.Add()([x, shortcut])
                    shortcut = x
            elif isinstance(layer, keras.layers.Conv2D):
                if layer.strides == (2, 2) and 'conv' in prev_layer.name and prev_layer.strides == (1, 1):
                    shortcut = layer(shortcut)
                else: x = layer(x)
            else: x = layer(x)
        
        model = keras.models.Model(inputs=layer_list[0].input, outputs=x, name="parsed_model")
        model.compile(loss='categorical_crossentropy', 
                      optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                      metrics=['accuracy'])  

        if self.plot == True:
            keras.utils.plot_model(model, self.config['paths']['path_wd'] + '/model_graph' + '/parsed_model.png', show_shapes=True)
        else:
            pass

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

        gamma = keras.backend.get_value(layer.gamma)
        mean = keras.backend.get_value(layer.moving_mean)
        var = keras.backend.get_value(layer.moving_variance)
        var_eps_sqrt_inv = 1 / np.sqrt(var + layer.epsilon)
        beta = keras.backend.get_value(layer.beta)
            
        return  gamma, mean, var_eps_sqrt_inv, beta, axis


    def _absorb_bn_parameters(self, weight, bias, gamma, mean, var_eps_sqrt_inv, beta, axis):
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
        mean = np.reshape(mean, broadcast_shape)
        beta = np.reshape(beta, broadcast_shape) #beta is shift parameter
        new_bias = np.ravel(beta + (bias - mean) * gamma * var_eps_sqrt_inv)
        new_weight = weight * gamma * var_eps_sqrt_inv
        
        return new_weight, new_bias


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

 
    def get_model_activation(self, name='input'):
        x_norm = None
        x_norm_file = np.load(os.path.join(self.config['paths']['dataset_path'], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']
        
        model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"{self.input_model_name}.h5"))
        
        input_model_activation_dir = os.path.join(self.config['paths']['path_wd'], 'input_model_activations')
        os.makedirs(input_model_activation_dir, exist_ok=True)
        parsed_model_activation_dir = os.path.join(self.config['paths']['path_wd'], 'parsed_model_activations')
        os.makedirs(parsed_model_activation_dir, exist_ok=True)
        
        for layer in model.layers:
            model_activation = tf.keras.models.Model(inputs=model.input, outputs=layer.output).predict(x_norm)
            np.savez_compressed(os.path.join(input_model_activation_dir, f"input_model_activation_{layer.name}.npz"), model_activation)
            if 'lambda' in layer.name:
                inbound = utils.get_inbound_layers_with_params(layer)
                pre_layer = inbound[0]
                np.savez_compressed(os.path.join(parsed_model_activation_dir, f"parsed_model_activation_{pre_layer.name}.npz"), model_activation)
            elif 'conv' in layer.name or 'dense' in layer.name:
                if hasattr(layer, 'activation'):
                    if layer.activation.__name__ == 'softmax':
                        np.savez_compressed(os.path.join(parsed_model_activation_dir, f"parsed_model_activation_{layer.name}.npz"), model_activation)
            else: np.savez_compressed(os.path.join(parsed_model_activation_dir, f"parsed_model_activation_{layer.name}.npz"), model_activation)
                
            
    def get_model_MAC(self, data_size):
        MAC = 0
        
        bias_flag = self.config["options"]["bias"]
        if bias_flag == 'False':
            bias_flag = False
        elif bias_flag == 'True':
            bias_flag = True
        else: print(f"ERROR !!")
        
        model = keras.models.load_model(os.path.join(self.config["paths"]["models"], f"{self.input_model_name}.h5"))
        for layer in model.layers:
            
            if 'conv' in layer.name:
                if bias_flag:
                    w, bias = layer.get_weights()
                else: w = layer.get_weights()[0]
            
                ii = 1 if keras.backend.image_data_format() == 'channels_first' else 0
                
                input_channels = w.shape[2]
                output_channels = w.shape[3]
                height_fm = layer.input_shape[1 + ii]
                width_fm = layer.input_shape[2 + ii]
                height_kn, width_kn = layer.kernel_size
                stride_y, stride_x = layer.strides
                
                if 'valid' == layer.padding:
                    padding_y = 0
                    padding_x = 0
                    numCols = int((width_fm - width_kn)/stride_x + 1)
                    numRows = int((height_fm - height_kn)/stride_y + 1)
                elif 'same' == layer.padding:
                    padding_y = (height_kn - 1) // 2
                    padding_x = (width_kn - 1) // 2
                    numCols = int((width_fm - width_kn + 2*padding_x)/stride_x + 1)
                    numRows = int((height_fm - height_kn + 2*padding_y)/stride_y + 1)

                mac = height_kn * width_kn * input_channels * output_channels * numCols * numRows
                MAC += mac
            
            elif 'dense' in layer.name:
                if bias_flag:
                    w, bias = layer.get_weights()
                else: w = layer.get_weights()[0]
                
                length_src = w.shape[0]
                length_tar = w.shape[1]
                
                mac = length_src * length_tar
                MAC += mac
            
            else: pass
            
        MAC *= data_size
        
        self.config["result"]["input_model_mac"] = str(MAC)
        
        