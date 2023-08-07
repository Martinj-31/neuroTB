#This file is running for Normalization
import os
import sys
import configparser
import tensorflow as tf
import numpy as np
#from tensorflow import keras
#from collections import OrderedDict
#from tensorflow.keras.models import Model

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

class Normalize:
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def normalize_parameter(self):
        
        config = configparser.ConfigParser()
        config.read(config)
        
        print("\n\n######## Normalization ########\n\n")
        
        x_norm = None

        x_norm_file = np.load(os.path.join(self.config["paths"]["path_wd"], 'x_norm.npz'))
        x_norm = x_norm_file['arr_0']  # Access the data stored in the .npz file

        # 변수 선언 및 초기화
        batch_size = self.config.getint('initial', 'batch_size')
        thr = self.config.getint('initial', 'threshold')
        
        #adjust_weight_factors -> weight를 조정하기 위한 변수 초기화
        norm_facs = {self.model.layers[0].name: 1.00}

        i = 0
        
        # parsed_model의 layer 순회
        for layer in self.model.layers:
            # layer에 weight가 없는 경우 skip
            if len(layer.weights) == 0:
                continue
            
            """
            print("\nThis input layer : \n", self.model.input)
            print("This output layer : \n", layer.output)
            print("\n")
            """
            
            activations = self.get_activations_layer(self.model.input, layer.output, 
                                                     x_norm, batch_size)
            print("Maximum activation: {:.5f}.".format(np.max(activations)))
            nonzero_activations = activations[np.nonzero(activations)]
            del activations
            perc = self.get_percentile(self.config, i)
            
            cliped_max_activation = self.get_percentile_activation(nonzero_activations, perc)
            #print("percentile maximum activation: {:.5f}.".format(cliped_max_activation))
            
            cliped_activations = self.clip_activations(nonzero_activations, 
                                                       cliped_max_activation)
            norm_facs[layer.name] = cliped_max_activation
            print("Cliped maximum activation: {:.5f}.\n".format(norm_facs[layer.name]))
            i += 1
            
            
        # scale factor를 적용하여 parsed_model layer에 대해 parameter normalize
        # normalize를 통해 model 수정
        for layer in self.model.layers:
            
            if len(layer.weights) == 0:
                continue
            
            # Adjust weight part 
            parameters = layer.get_weights()
            ann_weights = parameters[0]
            ann_bias = parameters[1]
            print("layer: \n", layer)
            if layer.activation.__name__ == 'softmax':
                norm_fac = 1.0
                print("\n Using norm_factor: {:.2f}.".format(norm_fac))
            
            else:
                norm_fac = norm_facs[layer.name]
            
            #_inbound_nodes를 통해 해당 layer의 이전 layer확인
            inbound = self.get_inbound_layers_with_params(layer)
            print("\ninbound layer: \n", inbound)
            # Weight normalization
            if len(inbound) == 0: #Input layer
                ann_weights_norm = [
                    ann_weights * norm_facs[self.model.layers[0].name] / norm_fac,
                    ann_bias / norm_fac]
           
            elif len(inbound) == 1:                   
                ann_weights_norm = [
                    ann_weights * norm_facs[inbound[0].name] / norm_fac, 
                    ann_bias / norm_fac]
            
            else:
                ann_weights_norm = [ann_weights, ann_bias]
           
            # threshold 
            snn_weights = ann_weights_norm * thr
            layer.set_weights(snn_weights)
            
    def get_activations_layer(self, layer_in, layer_out, x, batch_size=None):
        
        # 따로 batch_size가 정해져 있지 않는 경우 10으로 설정
        if batch_size is None:
            batch_size = 10
        
        # input sample x와 batch_size를 나눈 나머지가 0이 아닌 경우 
        if len(x) % batch_size != 0:
            # input sample list x에서 나눈 나머지만큼 지운다 
            x = x[: -(len(x) % batch_size)]
        
        print("Calculating activations of layer {}.".format(layer_out.name))
        # predict함수에 input sample을 넣어 해당 layer 뉴런의 activation을 계산
        activations = tf.keras.models.Model(inputs=layer_in, 
                                            outputs=layer_out).predict(x, batch_size)
        
        '''
        # 추가로 activations을 npz파일로 저장
        print("Writing activations to disk.")
        np.savez_compressed(os.path.join(path_wd, layer.name), activations)
        '''
        
        return np.array(activations)
     
       
    def get_percentile(self, config, layer_index=None):
        
        # percentile modification
        config['initial'] = { 'percentile': 99.9 }
        perc = config.getfloat('initial', 'percentile')
        
        print("percentile : {}".format(perc))
    
        return perc
    
    
    # n-th percentile에 해당하는 activation return
    def get_percentile_activation(self, activations, percentile):

        return np.percentile(activations, percentile) if activations.size else 1
    
    
    # activation array와 clip된 최대 activation을 입력으로 받아
    # 최대 activation보다 큰 것들은 제거한 activation array를 return
    def clip_activations(self, activations, max_activation):
        
        cliped_activations = np.clip(activations, a_min=None, 
                                     a_max=max_activation)
        
        return cliped_activations

    
    def get_inbound_layers_with_params(self, layer):
        
        inbound = layer
        prev_layer = None
        # weight가 존재하는 layer가 나올 때 반복
        while True:
            inbound = self.get_inbound_layers(inbound)
            
            # get_inbound_layers()에서 layer정보를 list에 받는데 list안에 layer정보가 있는 경우
            if len(inbound) == 1 and not isinstance(inbound[0], 
                                                    tf.keras.layers.BatchNormalization):
                inbound = inbound[0]
                if self.has_weights(inbound):
                    return [inbound]
                
            # layer정보가 없는 경우 -> input layer의 경우 previous layer 존재하지 않아 빈 list return
            else:
                result = []
                for inb in inbound:

                    if isinstance(inb, tf.keras.layers.BatchNormalization):
                        prev_layer = self.get_inbound_layers_with_params(inb)[0]
                            
                    if self.has_weights(inb):
                        result.append(inb)
                        
                    else:
                        result += self.get_inbound_layers_with_params(inb)
                    
                        
                if prev_layer is not None:
                    return [prev_layer]
                
                else:
                    return result
    
    def get_inbound_layers(self, layer):
        
        #_inbound_nodes를 통해 해당 layer의 이전 layer확인
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        
        # _inbound_nodes[]에서 return된 layer의 정보는 <>의 형식이므로 layer 정보를 list에 넣어줌
        if not isinstance(inbound_layers, (list, tuple)):
            inbound_layers = [inbound_layers]
            
        return inbound_layers
        
    
    def has_weights(self, layer):
        
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            return False
        else:    
            return len(layer.weights)
        
        
