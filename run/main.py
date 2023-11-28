import os, sys, configparser
import numpy as np
import time

from tensorflow import keras

import neuroToolbox.modelParser as parse
import neuroToolbox.weightConverter as convert
import neuroToolbox.networkCompiler as net
import neuroToolbox.networkAnalysis as networkAnalysis

# Define the path of the working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

sys.path.append(os.getcwd())


def run_neuroTB(config_filepath):
    ###### 1. Load data ######
    config = configparser.ConfigParser()
    config.read(config_filepath)
    
    # Load test dataset (for evaluation)
    x_test_file = np.load(os.path.join(config["paths"]["dataset_path"], 'x_test.npz'))
    x_test = x_test_file['arr_0']

    print("x_test.shape : ", x_test.shape)
    y_test_file = np.load(os.path.join(config["paths"]["dataset_path"], 'y_test.npz'))
    y_test = y_test_file['arr_0']

    x_norm = None
    x_norm_file = np.load(os.path.join(config['paths']['dataset_path'], 'x_norm.npz'))
    x_norm = x_norm_file['arr_0']

    # Read 'input_model' value from config.ini
    input_model_name = config["names"]["input_model"]
    # Load the model using the input_model_name
    input_model = keras.models.load_model(os.path.join(config["paths"]["models"], f"{input_model_name}.h5")) 

    start = time.time()
    # %% Parse model
    parser = parse.Parser(input_model, config)

    parsed_model = parser.parse()
    shift_params = parser.get_shift_params()
    
    # For comparison
    parser.get_models_activation(input_model_name, name='input')
    parser.get_models_activation(input_model_name, name='parsed')
    parser.compareAct(input_model_name)

    parsed_model.summary()
    
    score1, score2 = parser.parseAnalysis(input_model, parsed_model, x_test, y_test)
    # print("input model Test loss : ", score1[0])
    # print("input model Test accuracy : ", score1[1])
    print("parsed model Test loss : ", score2[0])
    print("parsed model Test accuracy : ", score2[1])
    
    # %% Normalization and convert
    converter = convert.Normalize(parsed_model, config)
    # converter.normalize_parameter()
    threshold = converter.balThreshold(shift_params)
    
    # %% Generate neurons and synapse connections for SNN
    batch_size = config["initial"]["batch_size"]
    batch_shape = list(list(parsed_model.layers[0].input_shape)[0])
    batch_shape[0] = batch_size
    data_size = 1000
    
    spike_model = net.networkGen(parsed_model, threshold, config)

    spike_model.setup_layers(batch_shape)
    spike_model.build()
    spike_model.summarySNN()

    spike_model.run(x_test[::data_size], y_test[::data_size])

    # %% Evaluation each step
    evaluation = networkAnalysis.Analysis(x_norm, input_model_name, config)
    
    evaluation.evalMapping(name='input')
    
    took = time.time() - start
    
    print(f"Total simulation time : {took}")