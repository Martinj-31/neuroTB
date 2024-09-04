import os, sys, configparser, ssl
import numpy as np
import keras.backend as K
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from datetime import datetime
from tensorflow import keras
from keras.layers import Lambda

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neuroToolbox.torchTokeras as torchTokeras

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
os.makedirs(path_wd + '/models/')

print("path wd: ", path_wd)
from run.main import run_neuroTB


# Pytorch MNIST 데이터셋 load
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), # required, otherwise MNIST are in PIL format
])
train = torchvision.datasets.MNIST('./_datafiles/', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('./_datafiles/', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

#torch Model 생성
class CustomTorchModel(nn.Module):
    def __init__(self):
        super(CustomTorchModel, self).__init__()
        #1x28x28
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,3), stride=1, padding='same')
        #4x28x28
        self.batch1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        #4x14x14
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,3), stride=1, padding='same')
        #8x6x6
        self.batch2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        #8x3x3
        self.flatten = nn.Flatten()
        
        self.dense = nn.Linear(784, 100)
        self.relu4 = nn.ReLU()
        self.dense_1 = nn.Linear(100, 10)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        flatten_input = x
        x = self.flatten(x)
        flatten_output = x
        x = self.dense(x)
        x = self.relu4(x)
        x = self.dense_1(x)
        #x = self.softmax(x)
        
        return x, flatten_input, flatten_output
torch_model = CustomTorchModel()

# Training loop
def training_loop(model, optimizer, loss_fn, train_loader, val_loader=None, n_epochs=100):
    best_loss, best_epoch = np.inf, -1
    best_state = model.state_dict()

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for data, target in train_loader:
            output, x1, x2 = model(data)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Validation
        model.eval()
        status = (f"{str(datetime.now())} End of epoch {epoch}, "
                  f"training loss={train_loss/len(train_loader)}")
        if val_loader:
            val_loss = 0
            for data, target in val_loader:
                output, x1, x2 = model(data)
                loss = loss_fn(output, target)
                val_loss += loss.item()
            status += f", validation loss={val_loss/len(val_loader)}"
        print(status)

# Run training
optimizer = optim.Adam(torch_model.parameters())
criterion = nn.CrossEntropyLoss()
training_loop(torch_model, optimizer, criterion, train_loader, test_loader, n_epochs=1)

# Save model
torch_model_name = "torch_model"
torch.save(torch_model, os.path.join(path_wd + '/models/', torch_model_name + '.pt'))

bias_flag = True

torchtokeras = torchTokeras.torchtokeras(torch_model, train_loader, test_loader)

model = torchtokeras.main()

x_train, x_test, y_train, y_test = torchtokeras.generate_train_data()

# Compile the model
model.compile(loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy'])

# %% Save model and parameters.
# Save the model
model_name = 'MNIST_CNN'
model.summary()
keras.models.save_model(model, os.path.join(path_wd + '/models/', model_name + '.h5'))

torchtokeras.evaluate_models(torch_model, model)

# Save the dataset
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'x_train'), x_train)
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'y_test'), y_test)
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'y_train'), y_train)
# Extracting datasets for Normalization
x_norm = x_train[::6000]
np.savez_compressed(os.path.join(path_wd + '/dataset/', 'x_norm'), x_norm)

# Evaluate the model
score = model.evaluate(x_test[:1000], y_test[:1000], verbose=0)
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
default_config['conversion']['firing_range'] = '10'
default_config['conversion']['fp_precision'] = 'FP32'
default_config['conversion']['normalization'] = 'off'
default_config['conversion']['optimizer'] = 'off'
default_config['conversion']['timesteps'] = '1000.0'
default_config['spiking_neuron']['refractory'] = '0'
default_config['spiking_neuron']['threshold'] = '1.0'
default_config['spiking_neuron']['w_mag'] = '1.0'
default_config['options']['bias'] = str(bias_flag)
default_config['test']['data_size'] = '1000'
default_config['result']['input_model_acc'] = str(score[1])


# Define path for the new config file
config_filepath = os.path.join(path_wd, 'config')

# Write the updated config values to a new file named 'config'
with open(config_filepath, 'w') as configfile:
    default_config.write(configfile)
run_neuroTB(config_filepath)  # Use run_neuroTB instead of run_n
