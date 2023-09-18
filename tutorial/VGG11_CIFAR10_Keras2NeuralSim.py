import os
import sys

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import numpy as np
import configparser
from datetime import datetime
from tensorflow import keras
from run import main


path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp', str(datetime.now().strftime("%m-%d" + "/" + "%H%M%S"))))
os.makedirs(path_wd)

print("path wd: ", path_wd)

########################DEFINE MODEL STRUCTURE#################################

def build_model_structure(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.layers.Input(shape=input_shape)
    
    # Block 1
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.AveragePooling2D((2, 2))(x)

    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.AveragePooling2D((2, 2))(x)

    # Block 3
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.AveragePooling2D((2, 2))(x)

    # Block 4
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.AveragePooling2D((2, 2))(x)

    # Block 5
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.AveragePooling2D((2, 2))(x)

    # Fully connected layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
###############################################################################

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Data preprocessing and normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to one-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Save the preprocessed dataset for later use
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'x_train'), x_train)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
np.savez_compressed(os.path.join(path_wd, 'y_train'), y_train)
# Extracting datasets for Normalization
x_norm = x_train[::6000]
np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_norm)

# Build VGG11 model
model = build_model_structure()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Train the model
batch_size = 4096
epochs = 1
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model_name = 'VGG11_CIFAR10'
keras.models.save_model(model, os.path.join(path_wd, model_name + '.h5'))

# Save the config file
default_config_path = os.path.join("..", "default_config")

# Load the default config file
default_config = configparser.ConfigParser()
default_config.read(default_config_path)

# Update the config values with new values
default_config['paths']['path_wd'] = path_wd
default_config['paths']['dataset_path'] = path_wd
default_config['paths']['filename_ann'] = model_name
default_config['paths']['filename_snn'] = model_name + '_for_SNN'
default_config['paths']['converted_model'] = path_wd + '/converted_model/'

# SNN configuration
default_config['initial']['w_mag'] = '64.0'
default_config['initial']['th_rate'] = '0.8'

# Define path for the new config file
config_filepath = os.path.join(path_wd, 'config')

# Write the updated config values to a new file named 'config'
with open(config_filepath, 'w') as configfile:
    default_config.write(configfile)

main.run_neuroTB(config_filepath)  # Use run_neuroTB instead of run_n
