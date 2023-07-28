#This file is running for Normalization
import os
import sys
import configparser
import numpy as np
from tensorflow.keras.models import Model

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

class Normalize:
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def normalize_parameter(self):
        
        print("Normalization")

        x_norm = None

        x_norm_file = np.load(os.path.join(self.config["paths"]["path_wd"], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']  # Access the data stored in the .npz file

        print("x_norm_data: \n", x_norm)
        
        """
        # 변수 선언 및 초기화
        batch_size = config.getint('initial', 'batch_size')
        
        #adjust_weight -> weight를 조정하기 위한 변수 초기화
        adj_weights = OrderedDict({model.layers[0].name: 1.00})

        i = 0
        # parsed_model의 layer 순회
        for layer in model.layer:
            # layer에 weight가 없는 경우 skip
            if len(layer.weights) == 0:
                continue
            
            activations = get_activations_layer(layer, model, x_norm, 
                                             batch_size)
            nonzero_activations = activations[np.nonzero(activations)]
            del activations
            perc = get_percentile(config, i)
            cliped_max_activation = get_percentile_activation(nonzero_activations, perc)
            print("percentile maximum activation: {:.2f}.".format(cliped_max_activation))
            
            cliped_activations = clip_activations(nonzero_activations, cliped_max_activation)
            
            adj_weights[layer.name] = cliped_max_activation
            print("Cliped maximum activation: {:.2f}.".format(adj_weights[layer.name]))
            i += 1
            
            
        # scale factor를 적용하여 parsed_model layer에 대해 parameter normalize
        # normalize를 통해 model 수정
        for layer in model.layer:
            
            if len(layer.weights) == 0:
                continue
            
            # Adjust weight part 
            parameters = layer.get_weight()
            if layer.activation.__name__ == 'softmax':
                adj_weight = 1.0
                print("Using cliped maximum activation: {:.2f}.".format(cliped_max_activation))
            
            else:
                adj_weight = adj_weights[layer.name]
            
            #_inbound_nodes를 통해 해당 layer의 이전 layer확인
            inbound = get_inbound_layers_with_params(layer)
            if len(inbound) == 0: #Input layer
                parameters_norm = [
                    parameters[0] * adj_weights[model.layers[0].name] / adj_weight]
           
            elif len(inbound) == 1:
                parameters_norm = [
                    parameters[0] * adj_weights[inbound[0].name] / adj_weight]
            
            else:
                parameters_norm = [parameters[0]]
            
            
    def get_activations_layer(layer_in, layer_out, x, batch_size=None):
        
        # 따로 batch_size가 정해져 있지 않는 경우 10으로 설정
        if batch_size is None:
            batch_size = 10
        
        # input sample x와 batch_size를 나눈 나머지가 0이 아닌 경우 
        if len(x) % batch_size != 0:
            # input sample list x에서 나눈 나머지만큼 지운다 
            x = x[: -(len(x) % batch_size)]
        
        print("Calculating activations of layer {}.".format(layer.name))
        # predict함수에 input sample을 넣어 해당 layer 뉴런의 activation을 계산
        activations = Model(layer_in, layer_out).predict(x, batch_size)
        
        '''
        # 추가로 activations을 npz파일로 저장할 코드를 따로 빼놓을지 고민...
        print("Writing activations to disk.")
        np.savez_compressed(os.path.join(activ_dir, layer.name), activations)
        '''
        
        return np.array(activations)
     
       
    def get_percentile(config, layer_index=None):
        
        perc = config.getfloat('normalization', 'percentile')
        
        return perc
    
    
    # n-th percentile에 해당하는 activation return
    def get_percentile_activation(activations, percentile):

        return np.percentile(activations, percentile) if activations.size else 1
    
    
    # activation array와 clip된 최대 activation을 입력으로 받아
    # 최대 activation보다 큰 것들은 제거한 activation array를 return
    def clip_activations(activations, max_activation):
        
        cliped_activations = np.clip(activations, a_min=None, 
                                     a_max=max_activation)
        
        return cliped_activations
    
#=======================================================================
# 공통으로 들어가는 함수
    
    def get_inbound_layers_with_params(layer):
        
        inbound = layer
        # weight가 존재하는 layer가 나올 때 반복
        while True:
            inbound = get_inbound_layers(inbound)
            # get_inbound_layers()에서 layer정보를 list에 받는데 list안에 layer정보가 있는 경우
            if len(inbound) == 1:
                inbound = inbound[0]
                if has_weights(inbound):
                    return[inbound]
            
            # layer정보가 없는 경우 -> input layer의 경우 previous layer 존재하지 않아 빈 list return
            else:
                result = []
                
                return result
    
    def get_inbound_layers(layer):
        
        #_inbound_nodes를 통해 해당 layer의 이전 layer확인
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        
        # _inbound_nodes[]에서 return된 layer의 정보는 <>의 형식이므로 layer 정보를 list에 넣어줌
        if not isinstance(inbound_layers, (list, tuple)):
            inbound_layers = [inbound_layers]
            
        return inbound_layers
        
    
    def has_weights(layer):
        
        return len(layer.weights)
    """