import os
import sys

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import time
from run import main
from datetime import datetime
import numpy as np
import configparser
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp', str(datetime.now().strftime("%m-%d" + "/" + "%H%M%S"))))
os.makedirs(path_wd)

print("path wd: ", path_wd)

# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리 및 정규화
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 레이블을 one-hot 인코딩
num_classes = 10
y_train = y_train.reshape(-1)  # Convert one-hot encoded labels to categorical labels
y_test = y_test.reshape(-1)  # Convert one-hot encoded labels to categorical labels

# 데이터셋 저장
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)

# 입력 레이어 정의
inputs = Input(shape=(32, 32, 3))

# Convolutional 레이어
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization(epsilon=1e-5)(x)  # BatchNormalization 레이어 추가
x = Conv2D(32, (3, 3), activation='relu')(x)
x = AveragePooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization(epsilon=1e-5)(x)  # BatchNormalization 레이어 추가
x = Conv2D(64, (3, 3), activation='relu')(x)
x = AveragePooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(6 * 6 * 64, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(128, activation='relu')(x)

outputs = Dense(num_classes, activation='softmax')(x)

# 모델 생성
model = Model(inputs=inputs, outputs=outputs)

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),  
              metrics=['accuracy'])

# 모델 학습
batch_size = 128
epochs = 1
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# 모델 평가
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 모델 저장
model_name = 'cifar10_cnn'
keras.models.save_model(model, os.path.join(path_wd, model_name + '.h5'))

# config 파일 저장
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
