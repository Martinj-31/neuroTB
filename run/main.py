import os
import sys
import configparser
from tensorflow.keras.models import load_model

sys.path.append(os.getcwd())


# import neuroToolbox.parse as parse
# import neuroToolbox.normalization as norm
# import neuroToolbox.neuPLUSNetwork as net

def run_neuroTB(config_filepath):
    ###### 1. Load data ######
    config = configparser.ConfigParser()
    config.read(config_filepath)
    
    # Read 'input_model' value from config.ini
    input_model_name = config["paths"]["filename_ann"]
    
    # Load the model using the input_model_name
    model = load_model(os.path.join(config["paths"]["path_wd"], f"{input_model_name}.h5"))
    
    # Print the summary of the loaded model
    print("Summary of", input_model_name)
    model.summary()
    
    
    # %% Parse model
    
    
    
    
    
    # %% Normalization and convert
    
    
    
    
    
    # %% Generate neurons and synapse connections for SNN
    '''
    batch_size = config["initial"]["batch_size"]
    batch_shape = list(model.layers[0].input_shape)
    batch_shape[0] = batch_size
    
    spike_model = net.networkGen(config, model)
    spike_model.setup_layers(batch_shape)
    '''