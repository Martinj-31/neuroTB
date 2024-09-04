import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, AveragePooling2D, Flatten, BatchNormalization, ReLU, Softmax, Layer, Input

class VerticalFlatten(Layer):
    def __init__(self, **kwargs):
        super(VerticalFlatten, self).__init__(**kwargs)

    def call(self, inputs):
        shape = tf.shape(inputs)
        x = tf.transpose(inputs, (0, 3, 1, 2))
        output = tf.reshape(x, [tf.shape(x)[0],  shape[1] * shape[2] * shape[3]])
        return output
    
    def compute_output_shape(self, input_shape):
        if input_shape[1] is None or input_shape[2] is None or input_shape[3] is None:
            raise ValueError("Input shape dimensions must be defined except for the batch size.")
        
        return (input_shape[0], input_shape[1] * input_shape[2] * input_shape[3])

class torchtokeras:
    def __init__(self, torch_model, input_train, input_test):
        self.torch_model  = torch_model
        self.keras_model = None
        self.torch_weights = self.torch_model.state_dict()
        self.keras_weights = []
        self.train_loader = input_train
        self.test_loader = input_test

        
        #keras test dataset.
        self.x_train, self.x_test, self.y_train, self.y_test = self.generate_train_data()
    
    def main(self):
        print(f"\n\n ######## Torch to Keras Conversion ######## \n")
        keras_model = self.Creates_keras_model()
        keras_model = self.mapping_param(keras_model)
        return keras_model

    #입력으로 받은 torch 모델에 맞는 keras model 생성
    def Creates_keras_model(self): 
        print(f"\n\n######## Creates a keras model ########")
        input_shape = self.x_train.shape[1:]
        inputs = keras.layers.Input(input_shape)
        x = inputs
        bias_flag = True
        linear_cnt = 0
        for name, layer in self.torch_model.named_modules():
            if isinstance(layer, nn.Conv2d):
                pad = "valid"
                if layer.padding == "same":
                    pad = "same"
                x = keras.layers.Conv2D(layer.out_channels, layer.kernel_size, strides=layer.stride, padding=f"{pad}", dilation_rate=layer.dilation, groups=layer.groups, use_bias=bias_flag)(x)
            elif isinstance(layer, nn.BatchNorm2d):
                axis=1 if keras.backend.image_data_format() == 'channels_first' else -1,
                x = keras.layers.BatchNormalization(epsilon=layer.eps, axis=axis, momentum=layer.momentum, scale=layer.affine, center=layer.affine)(x)
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Sigmoid) or isinstance(layer, nn.Tanh) or isinstance(layer, nn.LeakyReLU) or isinstance(layer, nn.Softmax):
                if layer.__class__.__name__ == "ReLU":
                    x = keras.layers.ReLU()(x)
                if layer.__class__.__name__ == "Softmax":
                    x = keras.layers.Softmax()(x)
            elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                kn_size = (layer.kernel_size, layer.kernel_size)
                pad = "valid"
                if layer.padding == "same":
                    pad = "same"
                if layer.__class__.__name__ == "AvgPool2d":
                    x = keras.layers.AveragePooling2D(pool_size=kn_size, strides=layer.stride, padding=f"{pad}")(x)
            elif isinstance(layer, nn.Flatten):

                x = keras.layers.Flatten()(x)
            elif isinstance(layer, nn.Linear):
                if linear_cnt == 0:
                    x = keras.layers.Dense(units=layer.out_features, use_bias=bias_flag)(x)
                else :
                    x = keras.layers.Dense(units=layer.out_features, use_bias=bias_flag, activation='softmax')(x)
                linear_cnt += 1

        keras_model = keras.Model(inputs=inputs, outputs=x)

        keras_model.build(input_shape)
        print(f"\n\n >>>>> Keras model Creation Completed.\n")
        print(f" ######## Keras model Summary ########\n")
        keras_model.summary(line_length=100)
        return keras_model

    def get_layer_outputs(self, model, input_tensor):
        layer_outputs = {}
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                # If output is a tuple, store the shapes of all its elements
                layer_outputs[module] = [list(o.shape) for o in output]
            else:
                # Otherwise, store the shape of the single output
                layer_outputs[module] = list(output.shape)
        hooks = []
        for layer in model.modules():
            if not isinstance(layer, nn.Sequential):
                hook = layer.register_forward_hook(forward_hook)
                hooks.append(hook)
        with torch.no_grad():
            model(input_tensor)
        for hook in hooks:
            hook.remove()
        
        return layer_outputs
    
    def mapping_param(self, keras_model):
        data, targets = next(iter(self.test_loader))
        data = data.numpy()
        input_tensor = torch.tensor(data[:1])  # 임의의 입력 텐서 (데이터의 배치 크기: 1)
        
        layer_output_shapes = self.get_layer_outputs(self.torch_model, input_tensor)

        print(f"\n\n ######## Start parameter mapping ########\n\n")
        first_dense_factor = 0
        for name, layer in self.torch_model.named_modules():
            if layer.__class__.__name__ == 'Flatten':
                output_shape = layer_output_shapes[previous_layer_instance]
                ch = output_shape[1]
                hei = output_shape[2]
                wid = output_shape[3]
            if hasattr(layer,'weight') and layer.weight is not None: 
                weight = layer.weight.detach().numpy()
                if layer.__class__.__name__ == 'Conv2d':
                    weight = np.moveaxis(weight, [0,1], [-1,-2])
                    self.keras_weights.append(weight)
                elif layer.__class__.__name__ == 'Linear':
                    #print(weight)
                    weight = weight.T
                    if first_dense_factor == 0:

                        new_weight = np.zeros(weight.shape)

                        c = 0
                        for i in range(hei*wid):
                            for j in range(ch):
                                new_weight[c] = weight[i+ j*hei*wid]
                                c += 1
                        first_dense_factor += 1
                        self.keras_weights.append(new_weight)
                    else:
                        self.keras_weights.append(weight)
                else:
                    self.keras_weights.append(weight)
            if hasattr(layer,'bias') and layer.bias is not None: 
                bias = layer.bias.detach().numpy()
                self.keras_weights.append(bias)
            if hasattr(layer,'running_mean') and layer.running_mean is not None: 
                running_mean = layer.running_mean.detach().numpy()
                self.keras_weights.append(running_mean)
            if hasattr(layer,'running_var') and layer.running_var is not None: 
                running_var = layer.running_var.detach().numpy()
                self.keras_weights.append(running_var)
            print(f">> layer : {name} , Mapping Complete.\n")
            previous_layer_instance = layer

        keras_model.set_weights(self.keras_weights)

        #keras model compile
        #keras_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        #keras model save
        #keras_model.save("keras_model.h5")

        self.keras_model = keras_model
            
        return keras_model
    
    #keras dataset으로 변환
    def generate_train_data(self):
        x_train, y_train = self.convert_to_numpy(self.train_loader)
        x_test, y_test = self.convert_to_numpy(self.test_loader)

        x_train = np.transpose(x_train, (0, 2, 3, 1))
        x_test = np.transpose(x_test, (0, 2, 3, 1))

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        return x_train, x_test, y_train, y_test

    def convert_to_numpy(self, data_loader):
        images, labels = [], []
        for batch in data_loader:
            imgs, lbls = batch
            images.append(imgs.numpy())
            labels.append(lbls.numpy())
        
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return images, labels

    #keras, torch model 성능 비교
    def evaluate_models(self, torch_model, keras_model):
        print(f"\nComparison of Keras and Pytorch Accuracy.\n")
        self.torch_model.eval()  # PyTorch 모델을 평가 모드로 설정
        correct_pytorch = 0
        total = 0

        # PyTorch 모델 예측 정확도 계산
        with torch.no_grad():  # Gradient 계산 비활성화
            for data, targets in self.test_loader:
                outputs_pytorch, x1, x2 = torch_model(data)
                _, predicted_pytorch = torch.max(outputs_pytorch.data, 1)
                correct_pytorch += (predicted_pytorch == targets).sum().item()
                total += targets.size(0)

        accuracy_pytorch = 100 * correct_pytorch / total
        print(f'PyTorch Model Accuracy: {accuracy_pytorch:.2f}%')

        # Keras 모델 평가
        loss, accuracy_keras = keras_model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f'Keras Model Accuracy: {accuracy_keras * 100:.2f}%')
        
        if accuracy_pytorch == accuracy_keras*100 :
            print(f"Conversion Complete")