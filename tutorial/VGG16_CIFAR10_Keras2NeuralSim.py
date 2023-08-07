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

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Data preprocessing and normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encode the labels
num_classes = 10
y_train = y_train.reshape(-1)  # Convert one-hot encoded labels to categorical labels
y_test = y_test.reshape(-1)  # Convert one-hot encoded labels to categorical labels

# Save the dataset
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
# Extracting datasets for Normalization
np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train[::10])

# VGG16 model, with weights pre-trained on ImageNet.
base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Add a new top layer
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)  # This layer is optional
x = keras.layers.Dense(4096, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(4096, activation='relu')(x)

outputs = keras.layers.Dense(num_classes, activation='softmax')(x)  # FC3

# This is the model we will train
model = keras.Model(inputs=base_model.input, outputs=outputs)

# First: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers to prevent large gradient updates wrecking the learned weights
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Train the model
batch_size = 128
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model_name = 'cifar10_vgg16'
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

# Define path for the new config file
config_filepath = os.path.join(path_wd, 'config')

# Write the updated config values to a new file named 'config'
with open(config_filepath, 'w') as configfile:
    default_config.write(configfile)

main.run_neuroTB(config_filepath)  # Use run_neuroTB instead of run_n
