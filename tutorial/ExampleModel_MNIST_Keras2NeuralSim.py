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
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Data preprocessing and normalization
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# One-hot encode the labels
num_classes = 10
y_train = y_train.reshape(-1)  # Convert one-hot encoded labels to categorical labels
y_test = y_test.reshape(-1)  # Convert one-hot encoded labels to categorical labels

# Save the dataset
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'x_train'), x_train)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
np.savez_compressed(os.path.join(path_wd, 'y_train'), y_train)
# Extracting datasets for Normalization
x_norm = x_train[::100]
np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_norm)

# Define the input layer
inputs = keras.Input(shape=(28, 28, 1))

# Convolutional layers
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', use_bias = False)(inputs)
#x = keras.layers.BatchNormalization(epsilon=1e-5, center = False)(x)  # Add BatchNormalization layer
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(32, activation='relu', use_bias = False)(x)  # Adjusted the dense layer size

outputs = keras.layers.Dense(num_classes, activation='softmax', use_bias = False)(x)

# Create the model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Train the model
batch_size = 128
epochs = 1
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Save the model
model_name = 'MNIST_CNN'
keras.models.save_model(model, os.path.join(path_wd, model_name + '.h5'))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("Summary of", model_name) # Print the summary of the loaded model
model.summary()

result_1 = keras.Model(inputs = model.input, outputs = model.layers[2].output).predict(x_test)
print("This is OG model's BN output : \n", result_1)


# Save the config file
default_config_path = os.path.abspath(os.path.join(current_dir, "..", "default_config"))

# Load the default config file
default_config = configparser.ConfigParser()
default_config.read(default_config_path)

# Update the config values with new values
default_config['paths']['path_wd'] = path_wd
default_config['paths']['dataset_path'] = path_wd
default_config['paths']['filename_ann'] = model_name
default_config['paths']['filename_snn'] = model_name + '_for_SNN'
default_config['paths']['converted_model'] = path_wd + '/converted_model/'
default_config['paths']['evaluation_layers'] = path_wd + '/evaluation_layers/'

# Define path for the new config file
config_filepath = os.path.join(path_wd, 'config')

# Write the updated config values to a new file named 'config'
with open(config_filepath, 'w') as configfile:
    default_config.write(configfile)

main.run_neuroTB(config_filepath)  # Use run_neuroTB instead of run_n
