import os
import sys
import configparser
from tensorflow.keras.models import load_model

sys.path.append(os.getcwd())

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import neuroToolbox.parse as parse
# import neuroToolbox.normalization as norm
# import neuroToolbox.neuPLUSNetwork as net

def run_neuroTB(config_filepath):
    ###### 1. Load data ######
    config = configparser.ConfigParser()
    config.read(config_filepath)
    
    # input_model_name에 저장된 모델 이름으로 저장된 모델을 불러와 해당 모델의 summary를 출력
    
    # Read 'input_model' value from config.ini
    input_model_name = config["paths"]["filename_ann"]
    # Load the model using the input_model_name
    input_model = load_model(os.path.join(config["paths"]["path_wd"], f"{input_model_name}.h5")) 
    print("Summary of", input_model_name) # Print the summary of the loaded model
    input_model.summary()
    
    # %% Parse model
    
    parser = parse.Parser(input_model, config)
    parser.print_layer_connections()
    parser.parse()
    
    
    
    # %% Normalization and convert
    
    
    
    
    
    # %% Generate neurons and synapse connections for SNN
    '''
    batch_size = config["initial"]["batch_size"]
    batch_shape = list(model.layers[0].input_shape)
    batch_shape[0] = batch_size
    
    spike_model = net.networkGen(config, model)
    spike_model.setup_layers(batch_shape)
    '''