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

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Load the ResNet50 model
base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = keras.layers.Flatten()(x)
# Add a fully-connected layer
x = keras.layers.Dense(512, activation='relu')(x)
# And a logistic layer for 10 classes
predictions = keras.layers.Dense(10, activation='softmax')(x)

# This is the model we will train
model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the new data for a few epochs
model.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model_name = 'ResNet_CIFAR10'
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
