############ 해당 부분은 simulation을 위해 간이로 작성된 코드입니다 ############
from REF_snntoolbox.bin.utils import update_setup, run_pipeline

def main(filepath = None):
    
    if filepath is not None:
        config = update_setup(filepath)
        run_pipeline(config)
        return

main('C:/Users/user/Desktop/JSB/Repo/snn_toolbox-master/temp/1686899764.7490518')


##############################################################################

from importlib import import_module
from tensorflow import keras
import tensorflow.keras.backend as k
import numpy as np

class ModelParser():
    def __init__(self, input_model, config):
        self.input_model = input_model
        self.config = config

        self._layer_list = []
        self._layer_dict = {}
        self.parsed_model = None
        
    def parse(self):
        # 변수 및 초기화

        layers = self.input_model.layers                 # Make layer iterable
        name_map = {}
        
        # 레이어 순회
        for layer in layers:
            layer_type = type(layer)

            # BatchNormalization 레이어의 파라미터를 이전 레이어의 파라미터에 흡수
            if layer_type == 'BatchNormalization':
                
                ########### 1.Get BatchNorm parameters ###########
                
                parameters_bn = list(self.get_batchnorm_parameters(layer))
                
                ########### 2.Get previous layer data ########### 
                ##REF CODE ##
                
                inbound = self.get_inbound_layers(layer)        # BatchNormalization 레이어 앞에 있는 레이어 가져오기
                prev_layer = inbound[0]
                
                print(prev_layer)
                
                prev_layer_idx = name_map[str(id(prev_layer))]                  # 앞에 있는 레이어의 인덱스 가져오기
                parameters = list(
                    self._layer_list[prev_layer_idx]['parameters'])             # 앞에 있는 레이어의 파라미터 가져오기
                prev_layer_type = self.get_type(prev_layer)                     # 앞에 있는 레이어의 타입 가져오기
                print("Absorbing batch-normalization parameters into " +
                      "parameters of previous {}.".format(prev_layer_type))

                _depthwise_conv_names = ['DepthwiseConv2D',
                                         'SparseDepthwiseConv2D']
                _sparse_names = ['Sparse', 'SparseConv2D',
                                 'SparseDepthwiseConv2D']
                is_depthwise = prev_layer_type in _depthwise_conv_names
                is_sparse = prev_layer_type in _sparse_names

                if is_sparse:
                    args = [parameters[0], parameters[2]] + parameters_bn
                else:
                    args = parameters[:2] + parameters_bn

                kwargs = {
                    'image_data_format': keras.backend.image_data_format(),
                    'is_depthwise': is_depthwise}

                # 3. absorb batchnorm parameters to previous layer #
                params_to_absorb = absorb_bn_parameters(*args, **kwargs)

                if is_sparse:
                    # Need to also save the mask associated with sparse layer.
                    params_to_absorb += (parameters[1],)

                self._layer_list[prev_layer_idx]['parameters'] = \
                    params_to_absorb


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
        
        
    def get_batchnorm_parameters(self, layer):
        mean = k.get_value(layer.moving_mean)
        var = k.get_value(layer.moving_variance)
        var_eps_sqrt_inv = 1 / np.sqrt(var + layer.epsilon)
        gamma = k.get_value(layer.gamma)
        beta = k.get_value(layer.beta)
    
        return [mean, var_eps_sqrt_inv, gamma, beta]
    
    def get_inbound_layer(self, layer):
        
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        
        return inbound_layers
        
def absorb_bn_parameters(weight, bias, mean, var_eps_sqrt_inv, gamma, beta,
                         image_data_format, is_depthwise=False):

    ###### Calculation by Numpy :  Area where BN parameters abosrb ######
    
    weight_bn = weight * gamma * var_eps_sqrt_inv
    bias_bn = beta + (bias - mean) * gamma * var_eps_sqrt_inv
    
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
    weight_eval_arr = weight_bn - weight_bn_loop
    bias_eval_arr = bias_bn - bias_bn_loop
   
    print("\n\n weight_eval_arr : \n\n", weight_eval_arr)
    print("\n\n bias_eval_arr : \n\n", bias_eval_arr)

    if np.all(weight_eval_arr == 0) and np.all(bias_eval_arr == 0) :
        print("BN parameter is properly absorbed.")
    else:
        raise NotImplementedError("BN parameter absorption is not properly implemented.")
    ##########################################################################
 
        return weight_bn, bias_bn


# 원래 model_parser = model_lib.ModelParser(input_model['model'], config)
model_parser = ModelParser(input_model['model'], config)

parsed_model = model_parser.build_parsed_model()