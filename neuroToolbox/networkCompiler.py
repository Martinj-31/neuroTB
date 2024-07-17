import sys, os, warnings, pickle, math, time
import numpy as np

from tensorflow import keras


sys.path.append(os.getcwd())

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


class networkCompiler:
    """
    Class for compiling an ANN model into a SNN model.
    """

    def __init__(self, parsed_model, config):
        """
        Initialize the networkCompile instance.

        Args:
            parsed_model (tf.keras.Model): The parsed model from Parser.
            config (configparser.ConfigParser): Configuration settings for compiling.
        """
        self.config = config
        self.parsed_model = parsed_model
        self.num_classes = int(self.parsed_model.layers[-1].output_shape[-1])
        self.nCount = 1024
        self.synCnt = 0
        self.core_cnt = 0
        self.flatten_shapes = []
        
        bias_flag = config["options"]["bias"]
        if bias_flag == 'False':
            self.bias_flag = False
        elif bias_flag == 'True':
            self.bias_flag = True
        else: print(f"ERROR !!")

        self.neurons = {}
        self.synapses = {}

        os.makedirs(self.config['paths']['path_wd'] + '/fr_distribution')


    def setup_layers(self, input_shape):
        """
        Set a layer from ANN model to SNN synapse connections.

        Args:
            input_shape (_type_): _description_
        """

        print("\n\n####### Compiling an ANN model into a SNN model #######\n")
        print(f"Data format is '{keras.backend.image_data_format()}'\n")
        convertible_layers = eval(self.config.get('restrictions', 'convertible_layers'))

        self.neuron_input_layer(input_shape)
        layers = []
        for layer in self.parsed_model.layers[1:]:
            layers.append(layer)
            layer_type = layer.__class__.__name__
            
            if layer_type not in convertible_layers and layer_type != 'Add':
                continue

            print(f"\n Building layer for {layer.__class__.__name__} ({layer.name})")

            if layer_type == 'Flatten':
                print(f"Flatten layer are not converted to SNN. (Skipped)")
                self.flatten_shapes.append(layers[-2])
                continue
            
            self.neuron_layer(layer)
            if layer_type == 'Dense':
                self.Synapse_dense(layer)
            # There are two types of convolution layer. 1D or 2D
            elif 'Conv' in layer_type:
                self.Synapse_convolution(layer)
            elif layer_type == 'AveragePooling2D':
                self.Synapse_pooling(layer)
            
            elif layer_type == 'Add':
                for node in layer._inbound_nodes:
                    incoming_layers = node.inbound_layers
                    if isinstance(incoming_layers, list):
                        connections = [incoming.name for incoming in incoming_layers]
                    else: connections = [incoming_layers.name]
                lambda_layer = self.get_layer_by_name(connections[1])
                
                for node in lambda_layer._inbound_nodes:
                    incoming_layer = node.inbound_layers
                    if isinstance(incoming_layer, list):
                        target = [incoming.name for incoming in incoming_layer]
                    else: target = [incoming_layer.name]
                
                target_layer = self.get_layer_by_name(target[0])
                
                if target_layer:
                    target_layer_type = target_layer.__class__.__name__
                    new_layer_name = target_layer.name
                    if target_layer.name == 'conv2d':
                        new_layer_name += '_identity'
                    elif target_layer_type == 'Add':
                        new_layer_name += '_identity'
                    elif 'Conv' in target_layer_type:
                        new_layer_name += '_conv'
                    print(f"-----(shortcut detect)-----")
                    target_layer._name = new_layer_name
                self.synapses[layer.name] = []

        print(f"\n>>> Setup layers complete.\n")


    def get_layer_by_name(self, layer_name):
        for layer in self.parsed_model.layers[1:]:
            if layer.name == layer_name:
                return layer
        
        return None
    
    
    # Input will be made in neuralSim library.
    def neuron_input_layer(self, input_shape):
        """
        Generate input layer neurons.

        Args:
            input_shape (_type_): _description_
        """
        self.neurons['input_layer'] = np.prod(input_shape[1:])


    def neuron_layer(self, layer):
        """
        Generate neurons for each layer.

        Args:
            layer (_type_): _description_
        """
        if len(layer._inbound_nodes) > 1:
            output_shape = layer.get_output_shape_at(1)
            self.neurons[layer.name] = np.prod(output_shape[1:])
        else:
            self.neurons[layer.name] = np.prod(layer.output_shape[1:])


    def Synapse_dense(self, layer):
        """_summary_
        This method is for generating synapse connection from CNN layer to SNN layer with neuron index.

        Args:
            layer (tf.keras.Model): Keras CNN model with weight information.

        Parameters:

        """
        print(f"Connecting layer...")

        if self.bias_flag:
            w, bias = layer.get_weights()
        else: w = layer.get_weights()[0]

        length_src = w.shape[0]
        length_tar = w.shape[1]

        source = np.zeros(length_src*length_tar)
        target = np.zeros(length_src*length_tar)
        weights = np.zeros(length_src*length_tar)

        cnt = 0
        if len(self.flatten_shapes) == 1:
            shape = self.flatten_shapes.pop().output_shape[1:]

            print(f"*** Flatten was detected. ***")

            y_in, x_in, f_in = shape

            for m in range(length_src):
                f = m % f_in
                y = m // (f_in * x_in)
                x = (m // f_in) % x_in
                new_m = f * x_in * y_in + x_in * y + x
                for n in range(length_tar):
                    source[cnt] = new_m
                    target[cnt] = n
                    weights[cnt] = w[m, n]
                    cnt += 1
        else:
            for m in range(length_src):
                for n in range(length_tar):
                    source[cnt] = m
                    target[cnt] = n
                    weights[cnt] = w[m, n]
                    cnt += 1
        
        source = source.astype(int) + self.synCnt
        self.synCnt += self.nCount
        target = target.astype(int) + self.synCnt
        num_src = np.prod(layer.input_shape[1:])
        num_tar = np.prod(layer.output_shape[1:])
        
        if self.bias_flag:
            self.synapses[layer.name] = [source, target, [num_src, num_tar], weights, bias]
        else: self.synapses[layer.name] = [source, target, [num_src, num_tar], weights]


    def Synapse_convolution(self, layer):
        """_summary_
        This method is for generating synapse connection from CNN layer to SNN layer with neuron index.

        Args:
            layer (tf.keras.Model): Keras CNN model with weight information.

        Parameters:
            weights (): Data shape is [filter_height, filter_width, input_channels, output_channels].
            height_fm (int): Height of feature map
            width_fm (int): Width of feature map
            height_kn, width_kn (int): Width and height of kernel
            stride_y, stride_x (int): strides
            numCols (): Number of columns in output filters (horizontal moves)
            numRows (): Number of rows in output filters (vertical moves)
            padding_y, paddding_x (int): Zero-padding rows, Zero-padding columns. It is (filter_size-1)/2

        Raises:
            NotImplementedError: _description_
        """

        print(f"Connecting layer...")

        if self.bias_flag:
            w, bias = layer.get_weights()
        else: w = layer.get_weights()[0]

        # 'channel_first' : [batch_size, channels, height, width]
        # 'channel_last' : [batch_size, height, width, channels]
        ii = 1 if keras.backend.image_data_format() == 'channels_first' else 0
        
        if len(layer._inbound_nodes) > 1:
            input_shape = layer.get_input_shape_at(1)
        else:
            input_shape = layer.input_shape

        input_channels = w.shape[2]
        output_channels = w.shape[3]
        height_fm = input_shape[1 + ii]
        width_fm = input_shape[2 + ii]
        height_kn, width_kn = layer.kernel_size
        stride_y, stride_x = layer.strides
        pad_cali = False
        
        fm = np.arange(width_fm*height_fm).reshape((width_fm, height_fm))
        if 'valid' == layer.padding:
            padding_y = 0
            padding_x = 0
            numCols = int((width_fm - width_kn)/stride_x + 1)
            numRows = int((height_fm - height_kn)/stride_y + 1)
            FM = fm
            for i in range(1, input_channels):
                FM = np.concatenate((FM, fm+(width_fm*height_fm*i)), axis=0)
        elif 'same' == layer.padding:
            padding_y = (height_kn - 1) // 2
            padding_x = (width_kn - 1) // 2
            numCols = int((width_fm - width_kn + 2*padding_x)/stride_x + 1)
            numRows = int((height_fm - height_kn + 2*padding_y)/stride_y + 1)
            fm_pad = np.pad(fm, ((padding_y, padding_y), (padding_x, padding_x)), mode='constant', constant_values=-1)
            FM = fm_pad
            for i in range(1, input_channels):
                FM = np.concatenate((FM, np.pad(fm+(width_fm*height_fm*i), ((padding_y, padding_y), (padding_x, padding_x)), mode='constant', constant_values=-1)), axis=0)
            if (width_fm - width_kn + 2*padding_x)/stride_x + 1 != numCols:
                pad_cali = True
                pad_cali_val = int(np.ceil((width_fm - width_kn + 2*padding_x)/stride_x + 1 - numCols))
                
        height_fm = int(FM.shape[0]/input_channels)
        width_fm = FM.shape[1]
        FM = FM.flatten() # Make feature map flatten to indexing

        source = np.zeros(numCols*numRows*(height_kn*width_kn)*input_channels*output_channels)
        target = np.zeros(numCols*numRows*(height_kn*width_kn)*input_channels*output_channels)
        weights = np.zeros(numCols*numRows*(height_kn*width_kn)*input_channels*output_channels)

        idx = 0
        target_cnt = 0
        output_channels_idx = []
        for fout in range(output_channels):
            row_idx = 0
            for i in range(numCols*numRows):
                if 0 == i%numCols and i != 0:
                    if pad_cali:
                        row_idx += width_fm*(stride_y-1) + width_kn - stride_x + pad_cali_val
                    else:
                        row_idx += width_fm*(stride_y-1) + width_kn - stride_x
                for fin in range(input_channels):
                    for j in range(height_kn):
                        source[idx:idx+width_kn] = FM[row_idx+fin*(height_fm*width_fm)+i+(j*width_fm):row_idx+fin*(height_fm*width_fm)+i+(j*width_fm)+width_kn]
                        target[idx:idx+width_kn] = np.zeros(len(source[idx:idx+width_kn])) + target_cnt
                        weights[idx:idx+width_kn] = w[j, 0:width_kn, fin, fout]
                        idx += width_kn
                row_idx += (stride_x-1)
                target_cnt += 1
            output_channels_idx.append(target_cnt)
                
        if 'same' == layer.padding:
            padding_idx = np.where(source == -1)[0]
            source = np.delete(source, padding_idx)
            target = np.delete(target, padding_idx)
            weights = np.delete(weights, padding_idx)

        source = source.astype(int) + self.synCnt
        self.synCnt += self.nCount
        target = target.astype(int) + self.synCnt
        num_src = np.prod(layer.input_shape[1:])
        num_tar = np.prod(layer.output_shape[1:])
        
        if self.bias_flag:
            self.synapses[layer.name] = [source, target, [num_src, num_tar], weights, bias, output_channels_idx]
        else: self.synapses[layer.name] = [source, target, [num_src, num_tar], weights, output_channels_idx]


    def Synapse_pooling(self, layer):
        """
        This method is for generating synapse connection from CNN layer to SNN layer with neuron index.
        
        Args:
            layer (tf.keras.Model): Keras CNN model with weight information.

        Parameters:
            width_fm (): Width of feature map
            height_fm (): Height of feature map
            numFm (): Number of feature maps
            width_pl (): Width of pool
            height_pl (): Height of pool
            sx, sy ():
        """
        if layer.__class__.__name__ == 'MaxPooling2D':
            warnings.warn("Layer type 'MaxPooling2D' is not supported.", RuntimeWarning)
        print(f"Connecting layer...")

        ii = 1 if keras.backend.image_data_format() == 'channels_first' else 0

        width_fm = layer.input_shape[2 + ii]
        height_fm = layer.input_shape[1 + ii]
        numFm = layer.input_shape[3 - 2 * ii]
        width_pl = layer.pool_size[1]
        height_pl = layer.pool_size[0]
        sx = layer.strides[1]
        sy = layer.strides[0]

        fm = np.arange(width_fm*height_fm).reshape((width_fm, height_fm))
        FM = fm
        for i in range(1, numFm):
            FM = np.concatenate((FM, fm+(width_fm*height_fm*i)), axis=0)
        FM = FM.flatten()

        weight = 1 / (width_pl * height_pl)

        source = np.zeros(width_fm*height_fm*numFm)
        target = np.zeros(width_fm*height_fm*numFm)
        weights = np.ones(width_fm*height_fm*numFm)*weight

        idx = 0
        row_idx = 0
        for i in range(int((width_fm/sx)*(height_fm/sy))*numFm):
            if 0 == i%(width_fm/sx) and i != 0:
                row_idx += width_fm*(height_pl-1)
            for j in range(height_pl):
                source[idx:idx+sx] = FM[row_idx+i*sx+(j*width_fm):row_idx+i*sx+(j*width_fm)+sx]
                target[idx:idx+sx] = np.zeros(len(source[idx:idx+sx])) + i
                idx += sx

        source = source.astype(int) + self.synCnt
        self.synCnt += self.nCount
        target = target.astype(int) + self.synCnt
        num_src = np.prod(layer.input_shape[1:])
        num_tar = np.prod(layer.output_shape[1:])

        self.synapses[layer.name] = [source, target, [num_src, num_tar], weights]


    def changekey_neurons(self):
        updated_neurons = {}
        neurons_keys = list(self.neurons.keys())
        i = 0
        
        print(f"Change key of neurons...")
        print(f"_________________________________________________________________")
        for layer in self.parsed_model.layers:
            new_key = layer.name
            
            if '_flatten' in new_key or 'lambda' in new_key:
                continue
            if i < len(neurons_keys):
                old_key = neurons_keys[i]
                if '_identity' in new_key or '_conv' in new_key:
                    updated_neurons[new_key] = self.neurons.pop(old_key)
                    print(f"Update {old_key} to {new_key}")
                else:
                    updated_neurons[old_key] = self.neurons[old_key]
                i += 1
        
        print(f"_________________________________________________________________\n")
        
        self.neurons = updated_neurons
        return self.neurons
    
    def changekey_synapses(self):
        updated_synapses = {}
        synapses_keys = list(self.synapses.keys())
        i = 0
        
        print(f"Change key of synapses...")
        print(f"_________________________________________________________________")
        for layer in self.parsed_model.layers:
            new_key = layer.name
            
            if '_flatten' in new_key or 'input' in new_key or 'lambda' in new_key:
                continue
            if i < len(synapses_keys):
                old_key = synapses_keys[i]
                if '_identity' in new_key or '_conv' in new_key:
                    updated_synapses[new_key] = self.synapses.pop(old_key)
                    print(f"Update {old_key} to {new_key}")
                else:
                    updated_synapses[old_key] = self.synapses[old_key]
                i += 1
        
        print(f"_________________________________________________________________\n")
        
        self.synapses = updated_synapses
        return self.synapses
    

    def build(self):
        """
        Build converted SNN model.

        Returns:
            _type_: _description_
        """

        print(f"Build and Store compiled SNN model...")

        filepath = self.config['paths']['models']
        filename = self.config['names']['snn_model']
        with open(filepath + filename + '_Converted_neurons.pkl', 'wb') as f:
            pickle.dump(self.neurons, f)
        with open(filepath + filename + '_Converted_synapses.pkl', 'wb') as f:
            pickle.dump(self.synapses, f)

        print(f"\n>>> Compiling DONE.\n\n")
        
        return [self.neurons, self.synapses]


    def layers(self):
        """
        Return layers.

        Returns:
            _type_: _description_
        """
        return self.synapses


    def neuronCoreNum(self):
        """
        Return the number of neuron core to be used on neuplus.

        Returns:
            _type_: _description_
        """
        neuron_num = {}
        for i in range(len(self.neurons)):
            neuron_num[list(self.neurons.keys())[i]] = list(self.neurons.values())[i]
            for j in range(math.ceil(list(self.neurons.values())[i]/1024)):
                self.core_cnt += 1
        return self.core_cnt


    def summarySNN(self):
        """
        Summarize converted SNN structure.
        """
        self.neurons = self.changekey_neurons()
        self.synapses = self.changekey_synapses()
        total_neuron = 0
        print(f"_________________________________________________________________")
        print(f"{'Model: '}{self.config['names']['snn_model']}")
        print(f"=================================================================")
        print(f"{'Network':<40}{'Parameters #':<40}")
        print(f"=================================================================")
        for i in range(len(self.neurons)):
            print(f"Neurons for")
            print(f"{list(self.neurons.keys())[i]:<40}{list(self.neurons.values())[i]:<40}")
            print(f"_________________________________________________________________")
            total_neuron += list(self.neurons.values())[i]
        print(f"=================================================================")
        print(f"Total neurons : {total_neuron}")
        print(f"Total {self.neuronCoreNum()} cores of neuron are needed.")
        print(f"_________________________________________________________________")
