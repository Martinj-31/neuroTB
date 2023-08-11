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
from neuroToolbox import parse



# Create a directory for saving outputs
path_wd = os.path.join(parent_dir, 'temp', datetime.now().strftime("%m-%d/%H%M%S"))
os.makedirs(path_wd, exist_ok=True)

# Save the config file
default_config_path = os.path.join("..", "default_config")

# Load the default config file
config = configparser.ConfigParser()
config.read(default_config_path)

# Update the config values with new values
config['train settings'] = {
    'loss': 'sparse_categorical_crossentropy',
    'optimizer': 'adam',
    'metrics': 'accuracy',
    'validation_split': '0.1',
    'callbacks': 'None',
    'batch_size': '128',
    'epochs': '1'
}

config['paths'] = {
    'path_wd': path_wd,
    'dataset_path': path_wd,
    'x_train': os.path.join(path_wd, "x_train.npz"),
    'x_test': os.path.join(path_wd, "x_test.npz"),
    'y_train': os.path.join(path_wd, "y_train.npz"),
    'y_test': os.path.join(path_wd, "y_test.npz"),
    'filename_ann': 'cifar10_vgg16'
}

# Define path for the new config file
config_filepath = os.path.join(path_wd, 'config')

# Write the updated config values to a new file named 'config'
with open(config_filepath, 'w') as configfile:
    config.write(configfile)


# Extract values from the config
loss = config['train settings']['loss']
optimizer_value = config['train settings']['optimizer']
metrics = config['train settings']['metrics']
validation_split = float(config['train settings']['validation_split'])
batch_size = int(config['train settings']['batch_size'])
epochs = int(config['train settings']['epochs'])

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Data preprocessing and normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encode the labels if using 'categorical_crossentropy' loss
if loss == 'categorical_crossentropy':
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Save the preprocessed dataset for later use
np.savez_compressed(os.path.join(path_wd, 'x_train'), x_train)
np.savez_compressed(os.path.join(path_wd, 'y_train'), y_train)
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train[::10])

# VGG16 model, with weights pre-trained on ImageNet.
base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
pre_parser = parse.Parser(base_model, config)
pre_parser.parse()

# Add a new top layer
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)  # This layer is optional
x = keras.layers.Dense(4096, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(4096, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)  # FC3

# This is the model we will train
model = keras.Model(inputs=base_model.input, outputs=outputs)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(loss=loss, optimizer=optimizer_value, metrics=[metrics])

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model_name = config['paths']['filename_ann']
model_path = os.path.join(config['paths']['path_wd'], model_name + '.h5')
keras.models.save_model(model, model_path)

main.run_neuroTB(config_filepath)