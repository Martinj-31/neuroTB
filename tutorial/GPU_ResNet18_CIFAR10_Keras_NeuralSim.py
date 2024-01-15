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
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp', str(datetime.now().strftime("%m-%d" + "/" + "%H%M%S"))))
os.makedirs(path_wd)
os.makedirs(path_wd + '/dataset/')
os.makedirs(path_wd + '/model_graph/')

print("path wd: ", path_wd)

##############
def identity_block(input_tensor, kernel_size, filters):
    filters1, filters2 = filters
    
    x = keras.layers.Conv2D(filters1, (1, 1), activation='relu')(input_tensor)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(filters2, kernel_size, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Add()([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    
    return x

def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    filters1, filters2 = filters
    
    x = keras.layers.Conv2D(filters1, (1, 1), strides=strides, activation='relu')(input_tensor)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(filters2, kernel_size, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    
    shortcut = keras.layers.Conv2D(filters2, (1, 1), strides=strides)(input_tensor)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    
    return x

def build_resnet18(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.layers.Input(shape=input_shape)
    
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)

    # ResNet blocks
    x = identity_block(x, 3, [64, 64])
    x = identity_block(x, 3, [64, 64])

    x = conv_block(x, 3, [128, 128])
    x = identity_block(x, 3, [128, 128])

    x = conv_block(x, 3, [256, 256])
    x = identity_block(x, 3, [256, 256])

    x = conv_block(x, 3, [512, 512])
    x = identity_block(x, 3, [512, 512])

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    keras.utils.plot_model(model, os.path.join(path_wd + '/model_graph' + '/input_model.png'), show_shapes=True)

    
    return model
#############


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Data preprocessing and normalization
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 

# Convert labels to one-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build ResNet18 model
model = build_resnet18(input_shape=(32, 32, 3), num_classes=num_classes)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
batch_size = 128
epochs = 1
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model_name = 'ResNet18_CIFAR10'
model.summary()
keras.models.save_model(model, os.path.join(path_wd + '/models/', model_name + '.h5'))

# Save the preprocessed dataset for later use
np.savez_compressed(os.path.join(path_wd + '/dataset/' , 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd + '/dataset/' , 'x_train'), x_train)
np.savez_compressed(os.path.join(path_wd + '/dataset/' , 'y_test'), y_test)
np.savez_compressed(os.path.join(path_wd + '/dataset/' , 'y_train'), y_train)

# Extracting datasets for Normalization
x_norm = x_train[::6000]
np.savez_compressed(os.path.join(path_wd + '/dataset/' , 'x_norm'), x_norm)

# Save the config file
default_config_path = os.path.abspath(os.path.join(current_dir, "..", "default_config"))

# Load the default config file
default_config = configparser.ConfigParser()
default_config.read(default_config_path)

# Update the config values with new values
default_config['paths']['path_wd'] = path_wd
default_config['paths']['dataset_path'] = path_wd + '/dataset/'
default_config['paths']['models'] = path_wd + '/models/'

default_config['names']['input_model'] = model_name
default_config['names']['parsed_model'] = 'parsed_' + model_name
default_config['names']['snn_model'] = 'SNN_' + model_name

default_config['conversion']['refractory'] = '5'
default_config['conversion']['threshold'] = '1'
default_config['conversion']['ratio'] = '1.6'

default_config['test']['data_size'] = '10'

# Define path for the new config file
config_filepath = os.path.join(path_wd, 'config')

# Write the updated config values to a new file named 'config'
with open(config_filepath, 'w') as configfile:
    default_config.write(configfile)

main.run_neuroTB(config_filepath)  # Use run_neuroTB instead of run_n

took = time.time() - start
print(took)