import os, sys
sys.path.append(os.getcwd())
import configparser

import neuroToolbox.parse as parse
import neuroToolbox.normalization as norm
import neuroToolbox.neuPLUSNetwork as net


# %% Load data
config = configparser.ConfigParser()



# %% Parse model





# %% Normalization and convert





# %% Generate neurons and synapse connections for SNN
batch_size = config.getint('initial', 'batch_size')
batsh_shape = list(model.layers[0].input_shape)
batsh_shape[0] = batch_size

spike_model = net.networkGen(config, model)
spike_model.setup_layers(batch_shape)