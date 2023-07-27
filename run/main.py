import os
import sys
import configparser
from tensorflow import keras

sys.path.append(os.getcwd())

import neuroToolbox.parse as parse
# import neuroToolbox.normalization as norm
import neuroToolbox.neuPLUSNetwork as net

def run_neuroTB(config_filepath):
    ###### 1. Load data ######
    config = configparser.ConfigParser()
    config.read(config_filepath)
    
    # Load the model stored with the model name stored in 'input_model_name' and print the summary of the model
    
    # Read 'input_model' value from config.ini
    input_model_name = config["paths"]["filename_ann"]
    # Load the model using the input_model_name
    input_model = keras.models.load_model(os.path.join(config["paths"]["path_wd"], f"{input_model_name}.h5")) 
    print("Summary of", input_model_name) # Print the summary of the loaded model
    input_model.summary()
    
    # %% Parse model
    
    parser = parse.Parser(input_model, config)

    parsed_model = parser.parse()
    
    parsed_model.summary()

    # %% Normalization and convert
    
    # %% Generate neurons and synapse connections for SNN
    
    batch_size = config["initial"]["batch_size"]
    batch_shape = list(parsed_model.layers[0].input_shape)
    batch_shape[0] = batch_size
    
    spike_model = net.networkGen(config, parsed_model)
    spike_model.setup_layers(batch_shape)
