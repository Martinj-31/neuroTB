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

    y_test_file = np.load(os.path.join(config["paths"]["dataset_path"], 'y_test.npz'))
    y_test = y_test_file['arr_0']

    # Read 'input_model' value from config.ini
    input_model_name = config["names"]["input_model"]
    # Load the model using the input_model_name
    input_model = keras.models.load_model(os.path.join(config["paths"]["models"], f"{input_model_name}.h5"))
    data_size = config.getint('test', 'data_size')

    start = time.time()
    # %% Parse model
    parser = parse.Parser(input_model, config)

    parsed_model = parser.parse()
    parsed_model.summary()
    
    # For comparison
    parser.get_model_activation()
    parser.get_model_MAC(data_size)
    
    parsed_model_score = parser.parseAnalysis(parsed_model, x_test[:1000], y_test[:1000])
    print("parsed model Test loss : ", parsed_model_score[0])
    print("parsed model Test accuracy : ", parsed_model_score[1])
    
    config['result']['parsed_model_acc'] = str(parsed_model_score[1])
    
    
    # %% Generate neurons and synapse connections for SNN
    batch_size = config["conversion"]["batch_size"]
    batch_shape = list(list(parsed_model.layers[0].input_shape)[0])
    batch_shape[0] = batch_size
    
    compiler = net.networkCompiler(parsed_model, config)

    compiler.setup_layers(batch_shape)
    compiler.summarySNN()
    spike_model = compiler.build()


    # %% Normalization and convert
    converter = convert.Converter(spike_model, config)
    
    converter.convertWeights()
    threshold = converter.get_threshold()
    
    
    # %% Evaluation each step
    evaluation = networkAnalysis.Analysis(config)
    
    evaluation.set_threshold(threshold)
    evaluation.run(data_size)
    evaluation.genResultFile()
    evaluation.plot_compare()
    
    took = time.time() - start
    
    print(f"Total simulation time : {took:.2f} seconds.")
    
