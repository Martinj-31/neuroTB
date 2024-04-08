import os, sys, configparser, ssl
import numpy as np
import keras.backend as K

from datetime import datetime
from tensorflow import keras
from keras.layers import Lambda

ssl._create_default_https_context = ssl._create_unverified_context

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from run import main
import neuroToolbox.utils as utils
import time

start = time.time()

# Add a channel dimension.
axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp', str(datetime.now().strftime("%m-%d" + "/" + "%H%M%S"))))
os.makedirs(path_wd)

print("path wd: ", path_wd)

########################DEFINE MODEL STRUCTURE#################################

def build_model_structure(input_shape=(32, 32, 3), num_classes=10):
    bias_flag = False
    inputs = keras.layers.Input(shape=input_shape)
    
    # Block 1
    x = keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(inputs)
    x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
    if bias_flag:
        x = keras.layers.BatchNormalization(epsilon=1e-5, axis = axis)(x)
    x = keras.layers.AveragePooling2D((2, 2))(x)

    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
    if bias_flag:
        x = keras.layers.BatchNormalization(epsilon=1e-5, axis = axis)(x)
    x = keras.layers.AveragePooling2D((2, 2))(x)

    # Block 3
    x = keras.layers.Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
    if bias_flag:
        x = keras.layers.BatchNormalization(epsilon=1e-5, axis = axis)(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
    if bias_flag:
        x = keras.layers.BatchNormalization(epsilon=1e-5, axis = axis)(x)
    x = keras.layers.AveragePooling2D((2, 2))(x)

    # Block 4
    x = keras.layers.Conv2D(512, (3, 3), padding='same', use_bias=False)(x)
    x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
    if bias_flag:
        x = keras.layers.BatchNormalization(epsilon=1e-5, axis = axis)(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', use_bias=False)(x)
    x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
    if bias_flag:
        x = keras.layers.BatchNormalization(epsilon=1e-5, axis = axis)(x)
    x = keras.layers.AveragePooling2D((2, 2))(x)

    # Block 5
    x = keras.layers.Conv2D(512, (3, 3), padding='same', use_bias=False)(x)
    x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
    if bias_flag:
        x = keras.layers.BatchNormalization(epsilon=1e-5, axis = axis)(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', use_bias=False)(x)
    x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
    if bias_flag:
        x = keras.layers.BatchNormalization(epsilon=1e-5, axis = axis)(x)
    x = keras.layers.AveragePooling2D((2, 2))(x)

    # Fully connected layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096, use_bias=False)(x)
    x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
    x = keras.layers.Dense(4096, use_bias=False)(x)
    x = Lambda(lambda x: K.clip(x, 0, 1e+7)/(K.clip(x, 0, 1e+7)*0.005+1))(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', use_bias=False)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
###############################################################################

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Data preprocessing and normalization
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 

# Convert labels to one-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Input domain transfer to Log domain
x_train = utils.data_transfer(x_train, 'log', False)
x_test = utils.data_transfer(x_test, 'log')

# Save the preprocessed dataset for later use
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'x_train'), x_train)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
np.savez_compressed(os.path.join(path_wd, 'y_train'), y_train)
# Extracting datasets for Normalization
x_norm = x_train[::6000]
np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_norm)

bias_flag = False

# Build VGG11 model
model = build_model_structure()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

# Train the model
batch_size = 64
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model_name = 'VGG11_CIFAR10'
keras.models.save_model(model, os.path.join(path_wd, model_name + '.h5'))

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

main.run_neuroTB(config_filepath)  # Use run_neuroTB instead of run_n

