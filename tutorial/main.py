import os
os.chdir("./")
print(os.getcwd())
import configparser
import neuroToolbox.neuPLUSNetwork as net


# %% Load data
config = configparser.ConfigParser()



# %% Parse model





# %% Normalization and convert





# %% Generate neurons and synapse connections for SNN
batsh_shape = list(model.layers[0].input_shape)
batsh_shape[0] = 1

spike_model = net.networkGen(config, model)
spike_model.setup_layers(batch_shape)