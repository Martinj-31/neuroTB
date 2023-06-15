#write by Jin
from importlib import import_module
import tensorflow.keras.backend as k
import numpy as np

#DNN2SNN_conversion\snn_toolbox-master\snntoolbox\parsing\model_libs\keras_input_lib.py
class Parser():
    def __init__(self, input_model, config):
        self.input_model = input_model
        self.config = config
        self._layer_list = []
        self._layer_dict = {}
        self.parsed_model = None
        
    def parse_model(self):
        # 변수 및 초기화
        layers = self.get_layer_iterable()
        convertible_layers = eval(self.config.get('restrictions', 'snn_layers'))
        name_map = {}
        idx = 0
        inserted_flatten = False
    
        # 레이어 순회
        for layer in layers:
            layer_type = self.get_type(layer)
    
            if layer_type == 'BatchNormalization':
                # BatchNormalization 레이어의 파라미터를 이전 레이어의 파라미터에 흡수
                
                # 1.Get BatchNorm parameters
                bn_parameters = self.get_bn_parameters(layer)
                bn_parameters, axis = bn_parameters[:-1], bn_parameters[-1]
                
                #2.Get previous layer data
                inbound = self.get_inbound_layers_with_parameters(layer) # BatchNormalization 레이어 앞에 있는 레이어 가져오기
                prev_layer = inbound[0]
                prev_layer_idx = name_map[str(id(prev_layer))] # 앞에 있는 레이어의 인덱스 가져오기
                parameters = list(self._layer_list[prev_layer_idx]['parameters']) # 앞에 있는 레이어의 파라미터 가져오기
                prev_layer_type = self.get_type(prev_layer) # 앞에 있는 레이어의 타입 가져오기
                
                # 3. absorb batchnorm parameters to previous layer
                parameters_to_absorb = self.absorb_bn_parameters(*args)
                self._layer_list[prev_layer_idx]['parameters'] = parameters_to_absorb
    
            if layer_type not in convertible_layers:
                # Dropout과 같이 ANN training에만 사용되는 layer는 제거
                continue
    
            if not inserted_flatten:
                # Conv 레이어와 FC 레이어 사이에 Flatten 레이어 삽입
                inserted_flatten = self.try_insert_flatten(layer, idx, name_map)
                idx += inserted_flatten
     
            # 인바운드 레이어 가져오기
            if inserted_flatten:
                inbound = [self._layer_list[-1]['name']]
                inserted_flatten = False
            else:
                inbound = self.get_inbound_names(layer, name_map)
    
            # 레이어 속성 초기화
            attributes = self.initialize_attributes(layer)
    
            if layer_type == 'Dense':
                # Dense 레이어 파싱
                self.parse_dense(layer, attributes)
    
            if layer_type == 'Sparse':
                # Sparse 레이어 파싱
                self.parse_sparse(layer, attributes)
    
            if layer_type in {'Conv1D', 'Conv2D'}:
                # Convolution 레이어 파싱
                self.parse_convolution(layer, attributes)
    
            # 다른 레이어 파싱 처리...
    
            # 레이어 리스트에 속성 추가
            self._layer_list.append(attributes)
    
            # 레이어 인덱스 매핑
            name_map[str(id(layer))] = idx
            idx += 1
    
        print('')

    def absorb_bn_parameters(weight, bias, mean, var_eps_sqrt_inv, gamma, beta,
                             axis, image_data_format, is_depthwise=False):
        """
        Absorb the parameters of a batch-normalization layer into the previous
        layer.
        """
    
        axis = weight.ndim + axis if axis < 0 else axis
    
        print("Using BatchNorm axis {}.".format(axis))
    
        # Map batch norm axis from layer dimension space to kernel dimension space.
        # Assumes that kernels are shaped like
        # [height, width, num_input_channels, num_output_channels],
        # and layers like [batch_size, channels, height, width] or
        # [batch_size, height, width, channels].
        if weight.ndim == 4:  # Conv2D
    
            channel_axis = 2 if is_depthwise else 3
    
            if image_data_format == 'channels_first':
                layer2kernel_axes_map = [None, channel_axis, 0, 1]
            else:
                layer2kernel_axes_map = [None, 0, 1, channel_axis]
    
            axis = layer2kernel_axes_map[axis]
        elif weight.ndim == 3:  # Conv1D
    
            channel_axis = 2
    
            if image_data_format == 'channels_first':
                layer2kernel_axes_map = [None, channel_axis, 0]
            else:
                layer2kernel_axes_map = [None, 0, channel_axis]
    
            axis = layer2kernel_axes_map[axis]
    
        broadcast_shape = [1] * weight.ndim
        broadcast_shape[axis] = weight.shape[axis]
        var_eps_sqrt_inv = np.reshape(var_eps_sqrt_inv, broadcast_shape)
        gamma = np.reshape(gamma, broadcast_shape)
        beta = np.reshape(beta, broadcast_shape)
        bias = np.reshape(bias, broadcast_shape)
        mean = np.reshape(mean, broadcast_shape)
        bias_bn = np.ravel(beta + (bias - mean) * gamma * var_eps_sqrt_inv)
        weight_bn = weight * gamma * var_eps_sqrt_inv
    
        return weight_bn, bias_bn        


        



        
                
        
    def get_layer_iterable(self):
        return self.input_model.layers

    def get_type(self, layer):
        return get_type(layer)

    def get_bn_params(self, layer):
        mean = k.get_value(layer.moving_mean)
        var = k.get_value(layer.moving_variance)
        gamma = k.get_value(layer.gamma)
        beta = k.get_value(layer.beta)
        axis = layer.axis
        
        var_eps_sqrt_inv = 1 / np.sqrt(var + layer.epsilon)
        
        
        
        if isinstance(axis, (list, tuple)):
            assert len(axis) == 1, "Multiple BatchNorm axes not understood."
            axis = axis[0]

        return [mean, var_eps_sqrt_inv, gamma, beta, axis]

    def get_inbound_layers(self, layer):
        return get_inbound_layers(layer)
    
    def get_inbound_names(self, layer, name_map):
        """Get names of inbound layers.

        Parameters
        ----------

        layer:
            Layer
        name_map: dict
            Maps the name of a layer to the `id` of the layer object.

        Returns
        -------

        : list
            The names of inbound layers.

        """

        inbound = self.get_inbound_layers(layer)
        for ib in range(len(inbound)):
            for _ in range(len(self.layers_to_skip)):
                if self.get_type(inbound[ib]) in self.layers_to_skip:
                    inbound[ib] = self.get_inbound_layers(inbound[ib])[0]
                else:
                    break
        if len(self._layer_list) == 0 or \
                any([self.get_type(inb) == 'InputLayer' for inb in inbound]):
            return [self.input_layer_name]
        else:
            inb_idxs = [name_map[str(id(inb))] for inb in inbound]
            return [self._layer_list[i]['name'] for i in inb_idxs]

    def get_inbound_layers_with_parameters(self, layer): ##원래 abstract##
        """Iterate until inbound layers are found that have parameters.

        Parameters
        ----------

        layer:
            Layer

        Returns
        -------

        : list
            List of inbound layers.
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
                        result += self.get_inbound_layers_with_parameters(inb)
                return result

    @property
    def layers_to_skip(self):
        # noinspection PyArgumentList
        return ['BatchNormalization',
                'Activation',
                'Dropout',
                'ReLU',
                'ActivityRegularization',
                'GaussianNoise']

    def has_weights(self, layer):
        return has_weights(layer)

    def initialize_attributes(self, layer=None):
        attributes = AbstractModelParser.initialize_attributes(self) ###########################<<<<<<<<<<<<<<<<<check
        attributes.update(layer.get_config())
        return attributes

    def get_input_shape(self):
        return \
            fix_input_layer_shape(self.get_layer_iterable()[0].input_shape)[1:]
            
    def get_batch_input_shape(self): ##원래 abstract##         build parsed model 에서 씀
        """Get the input shape of a network, including batch size.

        Returns
        -------

        batch_input_shape: tuple
            Batch input shape.
        """

        input_shape = tuple(self.get_input_shape())
        batch_size = self.config.getint('simulation', 'batch_size')
        return (batch_size,) + input_shape

    def get_name(self, layer, idx, layer_type=None): ##원래 abstract##
        """Create a name for a ``layer``.

        The format is <layer_num><layer_type>_<layer_shape>.

        >>> # Name of first convolution layer with 32 feature maps and
        >>> # dimension 64x64:
        "00Conv2D_32x64x64"
        >>> # Name of final dense layer with 100 units:
        "06Dense_100"

        Parameters
        ----------

        layer:
            Layer.
        idx: int
            Layer index.
        layer_type: Optional[str]
            Type of layer.

        Returns
        -------

        name: str
            Layer name.
        """

        if layer_type is None:
            layer_type = self.get_type(layer)

        output_shape = self.get_output_shape(layer)

        shape_string = ["{}x".format(x) for x in output_shape[1:]]
        shape_string[0] = "_" + shape_string[0]
        shape_string[-1] = shape_string[-1][:-1]
        shape_string = "".join(shape_string)

        num_str = self.format_layer_idx(idx)

        return num_str + layer_type + shape_string

    def format_layer_idx(self, idx):##원래 abstract##
        """Pad the layer index with the appropriate amount of zeros.

        The number of zeros used for padding is determined by the maximum index
        (i.e. the number of layers in the network).

        Parameters
        ----------

        idx: int
            Layer index.

        Returns
        -------

        num_str: str
            Zero-padded layer index.
        """

        max_idx = len(self.input_model.layers)
        return str(idx).zfill(len(str(max_idx)))

    def get_output_shape(self, layer):
        return layer.output_shape

    def try_insert_flatten(self, layer, idx, name_map):##원래abstract##
        output_shape = self.get_output_shape(layer)
        previous_layers = self.get_inbound_layers(layer)
        prev_layer_output_shape = self.get_output_shape(previous_layers[0])
        if len(output_shape) < len(prev_layer_output_shape) and \
                self.get_type(layer) not in {'Flatten', 'Reshape'} and \
                self.get_type(previous_layers[0]) != 'InputLayer':
            assert len(previous_layers) == 1, \
                "Layer to flatten must be unique."
            print("Inserting layer Flatten.")
            num_str = self.format_layer_idx(idx)
            shape_string = str(np.prod(prev_layer_output_shape[1:]))
            self._layer_list.append({
                'name': num_str + 'Flatten_' + shape_string,
                'layer_type': 'Flatten',
                'inbound': self.get_inbound_names(layer, name_map)})
            name_map['Flatten' + str(idx)] = idx
            return True
        else:
            return False

    def parse_sparse(self, layer, attributes):
        return self.parse_dense(layer, attributes)

    def parse_dense(self, layer, attributes):
        attributes['parameters'] = list(layer.get_weights())
        if layer.bias is None:
            attributes['parameters'].insert(
                1, np.zeros(layer.output_shape[1]))
            attributes['parameters'] = tuple(attributes['parameters'])
            attributes['use_bias'] = True

    def parse_sparse_convolution(self, layer, attributes):
        return self.parse_convolution(layer, attributes)

    def parse_convolution(self, layer, attributes):
        attributes['parameters'] = list(layer.get_weights())
        if layer.bias is None:
            attributes['parameters'].insert(1, np.zeros(layer.filters))
            attributes['parameters'] = tuple(attributes['parameters'])
            attributes['use_bias'] = True
        assert layer.data_format == k.image_data_format(), (
            "The input model was setup with image data format '{}', but your "
            "keras config file expects '{}'.".format(layer.data_format,
                                                     k.image_data_format()))

    def parse_sparse_depthwiseconvolution(self, layer, attributes):
        return self.parse_depthwiseconvolution(layer, attributes)

    def parse_depthwiseconvolution(self, layer, attributes):
        attributes['parameters'] = list(layer.get_weights())
        if layer.bias is None:
            a = 1 if layer.data_format == 'channels_first' else -1
            attributes['parameters'].insert(1, np.zeros(
                layer.depth_multiplier * layer.input_shape[a]))
            attributes['parameters'] = tuple(attributes['parameters'])
            attributes['use_bias'] = True

    def parse_transpose_convolution(self, layer, attributes):
        self.parse_convolution(layer, attributes)

    def parse_pooling(self, layer, attributes):
        pass
    
    def absorb_activation(self, layer, attributes):##원래abstact##

        activation_str = self.get_activation(layer)

        outbound = layer
        for _ in range(3):
            outbound = list(self.get_outbound_layers(outbound))
            if len(outbound) != 1:
                break
            else:
                outbound = outbound[0]

                if self.get_type(outbound) == 'Activation':
                    activation_str = self.get_activation(outbound)
                    break

                # Todo: Take into account relu parameters.
                if self.get_type(outbound) == 'ReLU':
                    print("Parsing ReLU parameters not yet implemented.")
                    activation_str = 'relu'
                    break

                try:
                    self.get_activation(outbound)
                    break
                except AttributeError:
                    pass

        activation, activation_str = get_custom_activation(activation_str)

        if activation_str == 'softmax' and \
                self.config.getboolean('conversion', 'softmax_to_relu'):
            activation = 'relu'
            print("Replaced softmax by relu activation function.")
        elif activation_str == 'linear' and self.get_type(layer) == 'Dense' \
                and self.config.getboolean('conversion', 'append_softmax',
                                           fallback=False):
            activation = 'softmax'
            print("Added softmax.")
        else:
            print("Using activation {}.".format(activation_str))

        attributes['activation'] = activation

    def get_activation(self, layer):

        return layer.activation.__name__

    def get_outbound_layers(self, layer):

        return get_outbound_layers(layer)

    def parse_concatenate(self, layer, attributes):
        pass

    @property
    def input_layer_name(self):
        # Check if model has a dedicated input layer. If so, return its name.
        # Otherwise, the first layer might be a conv layer, so we return
        # 'input'.
        first_layer = self.input_model.layers[0]
        if 'Input' in self.get_type(first_layer):
            return first_layer.name
        else:
            return 'input'

    def build_parsed_model(self): ##원래 abstract##
        """Create a Keras model suitable for conversion to SNN.

        This method uses the specifications in `_layer_list` to build a
        Keras model. The resulting model contains all essential information
        about the original network, independently of the model library in which
        the original network was built (e.g. Caffe).

        Returns
        -------

        parsed_model: keras.models.Model
            A Keras model, functionally equivalent to `input_model`.
        """

        img_input = keras.layers.Input(
            batch_shape=self.get_batch_input_shape(),
            name=self.input_layer_name)
        parsed_layers = {self.input_layer_name: img_input}
        print("Building parsed model...\n")
        for layer in self._layer_list:
            # Replace 'parameters' key with Keras key 'weights'
            if 'parameters' in layer:
                layer['weights'] = layer.pop('parameters')

            # Add layer
            layer_type = layer.pop('layer_type')
            if hasattr(keras.layers, layer_type):
                parsed_layer = getattr(keras.layers, layer_type)
            else:
                import keras_rewiring
                parsed_layer = getattr(keras_rewiring.sparse_layer, layer_type)

            inbound = [parsed_layers[inb] for inb in layer.pop('inbound')]
            if len(inbound) == 1:
                inbound = inbound[0]
            check_for_custom_activations(layer)
            parsed_layers[layer['name']] = parsed_layer(**layer)(inbound)

        print("Compiling parsed model...\n")
        self.parsed_model = keras.models.Model(img_input, parsed_layers[
            self._layer_list[-1]['name']])
        # Optimizer and loss do not matter because we only do inference.
        top_k = keras.metrics.TopKCategoricalAccuracy(
            self.config.getint('simulation', 'top_k'))
        self.parsed_model.compile('sgd', 'categorical_crossentropy',
                                  ['accuracy', top_k])
        # Todo: Enable adding custom metric via self.input_model.metrics.
        self.parsed_model.summary()
        return self.parsed_model

    def evaluate(self, batch_size, num_to_test, x_test=None, y_test=None,
                  dataflow=None):
        """Evaluate parsed Keras model.

        Can use either numpy arrays ``x_test, y_test`` containing the test
        samples, or generate them with a dataflow
        (``keras.ImageDataGenerator.flow_from_directory`` object).

        Parameters
        ----------

        batch_size: int
            Batch size

        num_to_test: int
            Number of samples to test

        x_test: Optional[np.ndarray]

        y_test: Optional[np.ndarray]

        dataflow: keras.ImageDataGenerator.flow_from_directory
        """

        assert (x_test is not None and y_test is not None or dataflow is not
                None), "No testsamples provided."

        if x_test is not None:
            score = self.parsed_model.evaluate(x_test, y_test, batch_size,
                                                verbose=0)
        else:
            steps = int(num_to_test / batch_size)
            score = self.parsed_model.evaluate(dataflow, steps=steps)
        print("Top-1 accuracy: {:.2%}".format(score[1]))
        print("Top-5 accuracy: {:.2%}\n".format(score[2]))

        return score

    @property
    def input_layer_name(self):
        # Check if model has a dedicated input layer. If so, return its name.
        # Otherwise, the first layer might be a conv layer, so we return
        # 'input'.
        first_layer = self.input_model.layers[0]
        if 'Input' in self.get_type(first_layer):
            return first_layer.name
        else:
            return 'input'

# 원래 model_parser = model_lib.ModelParser(input_model['model'], config)
model_parser = ModelParser(input_model['model'], config)

parsed_model = model_parser.build_parsed_model()