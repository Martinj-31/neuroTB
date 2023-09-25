import sys, os, warnings, pickle
sys.path.append(os.getcwd())

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from tensorflow import keras
import numpy as np
import math

class networkGen:

    def __init__(self, parsed_model, config):
        self.config = config
        self.parsed_model = parsed_model
        self.num_classes = int(self.parsed_model.layers[-1].output_shape[-1])
        self.nCount = 1024
        self.synCnt = 0
        self.core_cnt = 0
        self.flatten_shapes = []

        self.neurons = {}
        self.synapses = {}

        # mode On/Off

    def setup_layers(self, input_shape):
        self.neuron_input_layer(input_shape)
        layers = []
        for layer in self.parsed_model.layers[1:]:
            layers.append(layer)
            print(f"\nBuilding layer for {layer.__class__.__name__}")
            layer_type = layer.__class__.__name__
            if layer_type == 'Flatten':
                print(f"Flatten layer is not going to be conoverted to SNN. (Skipped)")
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

        print(f"Total {int(self.synCnt/1024)} neuron cores are going to be used.")

    # Input will be made in neuralSim library.
    def neuron_input_layer(self, input_shape):
        self.neurons['input_layer'] = np.prod(input_shape[1:])

    def neuron_layer(self, layer):
        self.neurons[layer.name] = np.prod(layer.output_shape[1:])

    def Synapse_dense(self, layer):
        print(f"Connecting layer...")
        
        w = list(layer.get_weights())[0]

        length_src = w.shape[0]
        length_tar = w.shape[1]

        source = np.zeros(length_src*length_tar)
        target = np.zeros(length_src*length_tar)
        weights = np.zeros(length_src*length_tar)

        cnt = 0
        if len(self.flatten_shapes) == 1:
            shape = self.flatten_shapes.pop().output_shape[1:]
            print(f"Flatten was detected.")
            print(f"Data format is '{keras.backend.image_data_format()}'")
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
        
        self.synapses[layer.name] = [source, target, weights]

    def Synapse_convolution(self, layer):
        """_summary_
        This method is for generating synapse connection from CNN layer to SNN layer with neuron index.

        Args:
            layer (Keras.model): Keras CNN model with weight information.
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

        w = list(layer.get_weights())[0]

        # 'channel_first' : [batch_size, channels, height, width]
        # 'channel_last' : [batch_size, height, width, channels]
        ii = 1 if keras.backend.image_data_format() == 'channels_first' else 0

        input_channels = w.shape[2]
        output_channels = w.shape[3]
        height_fm = layer.input_shape[1 + ii]
        width_fm = layer.input_shape[2 + ii]
        height_kn, width_kn = layer.kernel_size
        stride_y, stride_x = layer.strides

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

        height_fm = int(FM.shape[0]/input_channels)
        width_fm = FM.shape[1]

        FM = FM.flatten() # Make feature map flatten to indexing

        source = np.zeros(numCols*numRows*(height_kn*width_kn)*input_channels*output_channels)
        target = np.zeros(numCols*numRows*(height_kn*width_kn)*input_channels*output_channels)
        weights = np.zeros(numCols*numRows*(height_kn*width_kn)*input_channels*output_channels)

        idx = 0
        fin = 0
        target_cnt = 0
        reset_cnt = 0
        for fout in range(output_channels):
            row_idx = 0
            for i in range(numCols*numRows*input_channels):
                if 0 == i%(numCols*numRows) and i != 0:
                    if 'valid' == layer.padding:
                        row_idx += 2*width_fm + width_kn - stride_x
                    else:
                        row_idx += 2*padding_y*width_fm + width_kn - stride_x
                    fin += 1
                    target_cnt = reset_cnt
                    if fin == input_channels-1:    
                        fin = 0
                # When stride is more than 1,
                elif 0 == i%numCols and i != 0:
                    row_idx += width_fm*(stride_y-1) + width_kn - stride_x
                for j in range(height_kn):
                    source[idx:idx+width_kn] = FM[row_idx+i+(j*width_fm):row_idx+i+(j*width_fm)+width_kn]
                    target[idx:idx+width_kn] = np.zeros(len(source[idx:idx+width_kn])) + target_cnt
                    weights[idx:idx+width_kn] = w[j, 0:width_kn, fin, fout]
                    idx += width_kn
                row_idx += (stride_x-1)
                target_cnt += 1
            reset_cnt = target_cnt
                
        if 'same' == layer.padding:
            padding_idx = np.where(source == -1)[0]
            source = np.delete(source, padding_idx)
            target = np.delete(target, padding_idx)
            weights = np.delete(weights, padding_idx)

        source = source.astype(int) + self.synCnt
        self.synCnt += self.nCount
        target = target.astype(int) + self.synCnt

        self.synapses[layer.name] = [source, target, weights]

    def Synapse_pooling(self, layer):
        """_summary_

        Args:
            layer (Keras.model): Keras CNN model with weight information.
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

        self.synapses[layer.name] = [source, target, weights]

    def build(self):
        filepath = self.config.get('paths', 'converted_model')
        filename = self.config.get('paths', 'filename_snn')
        with open(filepath + filename + '_Converted_neurons.pkl', 'wb') as f:
            pickle.dump(self.neurons, f)
        with open(filepath + filename + '_Converted_synapses.pkl', 'wb') as f:
            pickle.dump(self.synapses, f)

        print(f"Spiking neural network build completed!")

    def layers(self):
        return self.synapses

    def neuronCoreNum(self):
        neuron_num = {}
        for i in range(len(self.neurons)):
            neuron_num[list(self.neurons.keys())[i]] = list(self.neurons.values())[i]
            for j in range(math.ceil(list(self.neurons.values())[i]/1024)):
                self.core_cnt += 1
        return self.core_cnt

    def summarySNN(self):
        print(f"_________________________________________________________________")
        print(f"{'Model: '}{self.config.get('paths', 'filename_snn')}")
        print(f"=================================================================")
        print(f"{'Network':<40}{'Parameters #':<40}")
        print(f"=================================================================")
        for i in range(len(self.neurons)):
            print(f"Neurons for")
            print(f"{list(self.neurons.keys())[i]:<40}{list(self.neurons.values())[i]:<40}")
            print(f"_________________________________________________________________")
        print(f"=================================================================")
        print(f"Total neurons : {0}")
        print(f"Total {self.neuronCoreNum()} of neuron cores are needed.")
        print(f"_________________________________________________________________")
