import os, sys, configparser, ssl
import numpy as np
import keras.backend as K

from datetime import datetime
from tensorflow import keras
from keras.layers import Lambda


# %% Setup environment.
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
from run.main import run_neuroTB
import neuroToolbox.utils as utils


# %% Load MNIST dataset and preprocess
# Add a channel dimension.
axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Data preprocessing and normalization
x_train = x_train.reshape(x_train.shape[0], 28, 28, axis).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, axis).astype('float32')

# One-hot encode target vectors.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Input domain transfer to Log domain
x_train = utils.data_transfer(x_train, 'log', False)
x_test = utils.data_transfer(x_test, 'log')


# %% Define DNN model.
# Define the input layer
input_shape = x_train.shape[1:]
inputs = keras.layers.Input(input_shape)

bias_flag = False

# Convolutional layers
x = keras.layers.Conv2D(4, (3, 3), strides=(1, 1), padding='same', use_bias=bias_flag)(inputs)
x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
if bias_flag:
    x = keras.layers.BatchNormalization(epsilon=1e-5, axis = axis)(x)
else: pass
x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

x = keras.layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same', use_bias=bias_flag)(x)
x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
if bias_flag:
    x = keras.layers.BatchNormalization(epsilon=1e-5, axis = axis)(x)
else: pass
x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(units=100, use_bias=bias_flag)(x)
x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
outputs = keras.layers.Dense(units=10, activation='softmax', use_bias=bias_flag)(x)

# Create the model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))


# %% Save model and parameters.
# Save the model
model_name = 'MNIST_CNN'
model.summary()
keras.models.save_model(model, os.path.join(path_wd + '/models/', model_name + '.h5'))

# Save the dataset
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'x_train'), x_train)
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'y_test'), y_test)
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'y_train'), y_train)
# Extracting datasets for Normalization
x_norm = x_train[::6000]
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
default_config['conversion']['firing_range'] = '20'
default_config['conversion']['fp_precision'] = 'FP32'
default_config['conversion']['epoch'] = '30'
default_config['conversion']['normalization'] = 'on'
default_config['conversion']['optimizer'] = 'on'
default_config['conversion']['loss_alpha'] = '0.999'
default_config['conversion']['scaling_step'] = '1'

default_config['spiking_neuron']['refractory'] = '5'
default_config['spiking_neuron']['threshold'] = '16.0'
default_config['spiking_neuron']['w_mag'] = '64.0'

default_config['options']['bias'] = str(bias_flag)
default_config['options']['trans_domain'] = 'log'

default_config['test']['data_size'] = '10'

default_config['result']['input_model_acc'] = str(score[1])

# Define path for the new config file
config_filepath = os.path.join(path_wd, 'config')

# Write the updated config values to a new file named 'config'
with open(config_filepath, 'w') as configfile:
    default_config.write(configfile)
run_neuroTB(config_filepath)  # Use run_neuroTB instead of run_n
