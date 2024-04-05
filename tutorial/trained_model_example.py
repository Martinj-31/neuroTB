import os, sys, configparser, ssl
import numpy as np
import keras.backend as K

from tensorflow import keras
from datetime import datetime
from keras.layers import Lambda

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# Disable SSL authentication.
ssl._create_default_https_context = ssl._create_unverified_context

# Make directory for Simulation.
path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp', str(datetime.now().strftime("%m-%d" + "/" + "%H%M%S"))))
os.makedirs(path_wd)
os.makedirs(path_wd + '/dataset/')

print("path wd: ", path_wd)

# Import the run_neuroTB function from run.main
from run.main import run_neuroTB

model_path = '/Users/mingyucheon/work/neuroTB/temp/models'
model_name = 'MNIST_CNN'
bias_flag = False
model = keras.models.load_model(os.path.join(model_path, f"models/{model_name}.h5"))
model.summary()
keras.models.save_model(model, os.path.join(path_wd + '/models/', model_name + '.h5'))

x_test_file = np.load(os.path.join(model_path, f"dataset/x_test.npz"))
x_test = x_test_file['arr_0']
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'x_test'), x_test)
x_train_file = np.load(os.path.join(model_path, f"dataset/x_train.npz"))
x_train = x_train_file['arr_0']
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'x_train'), x_train)
y_test_file = np.load(os.path.join(model_path, f"dataset/y_test.npz"))
y_test = y_test_file['arr_0']
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'y_test'), y_test)
y_train_file = np.load(os.path.join(model_path, f"dataset/y_train.npz"))
y_train = y_train_file['arr_0']
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'y_train'), y_train)
x_norm_file = np.load(os.path.join(model_path, f"dataset/x_norm.npz"))
x_norm = x_norm_file['arr_0']
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'x_norm'), x_norm)

# Evaluate the model
score = model.evaluate(x_test[::int(10000 / 1000)], y_test[::int(10000 / 1000)], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# %% Setup configuration file.
# Save the config file
default_config_path = os.path.abspath(os.path.join(current_dir, "..", "default_config"))

# Load the default config file
default_config = configparser.ConfigParser()
default_config.read(default_config_path)

# Update the config values with new values
default_config['paths']['path_wd'] = path_wd
default_config['paths']['dataset_path'] = path_wd + '/dataset/'
default_config['paths']['models'] = path_wd + '/models/'

default_config['names']['dataset'] = 'MNIST'
default_config['names']['input_model'] = model_name
default_config['names']['parsed_model'] = 'parsed_' + model_name
default_config['names']['snn_model'] = 'SNN_' + model_name

default_config['conversion']['neuron'] = 'IF'
default_config['conversion']['batch_size'] = '1'
default_config['conversion']['firing_range'] = '100'
default_config['conversion']['fp_precision'] = 'FP32'
default_config['conversion']['epoch'] = '100'
default_config['conversion']['normalization'] = 'on'
default_config['conversion']['optimizer'] = 'on'
default_config['conversion']['loss_alpha'] = '0.2'
default_config['conversion']['scaling_step'] = '1'

default_config['spiking_neuron']['refractory'] = '5'
default_config['spiking_neuron']['threshold'] = '128.0'
default_config['spiking_neuron']['w_mag'] = '64.0'

default_config['options']['bias'] = str(bias_flag) 
default_config['options']['trans_domain'] = 'log'

default_config['test']['data_size'] = '1000'

default_config['result']['input_model_acc'] = str(score[1])

# Set the relative path of the config file
config_filepath = os.path.join(path_wd, 'config')

# Write the updated config values to a new file named 'config'
with open(config_filepath, 'w') as configfile:
    default_config.write(configfile)

# Call the run_neuroTB function with the config_filepath
run_neuroTB(config_filepath)
