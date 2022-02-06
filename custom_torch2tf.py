import numpy as np
import torch
from torch import nn

import tensorflow as tf
from tensorflow.python.keras.engine import training
from tensorflow.python.keras import layers
# from tensorflow.python.keras.layers import VersionAwareLayers

import models
from handlers.fuse_torch_model import fuse


va_layers = layers  # VersionAwareLayers()


def do_pad(pad_size, input_tensor, name_prefix):
    if sum(pad_size) > 0:
        input_tensor = layers.ZeroPadding2D(padding=pad_size, name=name_prefix + '_pad')(input_tensor)
    return input_tensor


def construct_keras_conv(keras_layers, torch_op, op_name, input_tensor):
    # op_name.split('_')[0][-2:] == '2d'
    use_bias = torch_op.bias is not None
    is_depthwise = torch_op.groups > 1
    input_tensor = do_pad(torch_op.padding, input_tensor, op_name)
    weights = [torch_op.weight.detach().numpy().transpose(2, 3, 0, 1) if is_depthwise else
               torch_op.weight.detach().numpy().transpose(2, 3, 1, 0)]
    if use_bias:
        weights.append(torch_op.bias.detach().numpy())
    if is_depthwise:
        input_tensor = keras_layers.DepthwiseConv2D(kernel_size=torch_op.kernel_size,
                                                    strides=torch_op.stride,
                                                    padding='valid',
                                                    use_bias=use_bias,
                                                    dilation_rate=torch_op.dilation,
                                                    activation=None,
                                                    name=op_name + '_depthconv',
                                                    weights=weights)(input_tensor)
    else:
        input_tensor = keras_layers.Conv2D(filters=torch_op.out_channels,
                                           kernel_size=torch_op.kernel_size,
                                           strides=torch_op.stride,
                                           padding='valid',
                                           use_bias=use_bias,
                                           dilation_rate=torch_op.dilation,
                                           activation=None,
                                           name=op_name + '_conv',
                                           weights=weights)(input_tensor)
    return input_tensor


def construct_keras_linear(keras_layers, torch_op, op_name, input_tensor):
    weights = [torch_op.weight.detach().numpy().transpose()]
    use_bias = torch_op.bias is not None
    if use_bias:
        weights.append(torch_op.bias.detach().numpy())
    input_tensor = keras_layers.Dense(units=torch_op.out_features,
                                      activation=None,
                                      use_bias=use_bias,
                                      name=op_name,
                                      weights=weights,
                                      kernel_regularizer=None,
                                      bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                      bias_constraint=None)(input_tensor)
    return input_tensor


def construct_keras_flatten(keras_layers, torch_op, op_name, input_tensor):
    input_tensor = keras_layers.Permute((3, 2, 1))(input_tensor)
    input_tensor = keras_layers.Flatten(name=op_name)(input_tensor)
    return input_tensor


def construct_maxpool(keras_layers, torch_op, op_name, input_tensor):
    if isinstance(torch_op.padding, tuple):
        if sum(torch_op.padding) > 0:
            input_tensor = keras_layers.ZeroPadding2D(padding=torch_op.padding, name=op_name + '_pad')(input_tensor)
    elif torch_op.padding > 0:
        input_tensor = keras_layers.ZeroPadding2D(padding=torch_op.padding, name=op_name + '_pad')(input_tensor)
    keras_layers.MaxPool2D(pool_size=torch_op.kernel_size, strides=torch_op.stride, padding='valid')
    return input_tensor


def construct_adaptive_avg_pool(keras_layers, torch_op, op_name, input_tensor):
    if isinstance(torch_op.output_size, int):
        output_size = np.array([torch_op.output_size, torch_op.output_size])
    else:
        output_size = np.array(torch_op.output_size)
    inp_hw = np.array(input_tensor.shape[1:-1])
    pool_size = (inp_hw / output_size).astype(int)
    input_tensor = keras_layers.AveragePooling2D(pool_size, name=op_name)(input_tensor)
    return input_tensor


def get_identity(keras_layers, torch_op, op_name, input_tensor):
    return tf.identity(input_tensor)


tf_layers_constructors = {'conv': construct_keras_conv,
                          'linear': construct_keras_linear,
                          'flatten': construct_keras_flatten,
                          'identity': get_identity,
                          'adaptiveavgpool2d': construct_adaptive_avg_pool,
                          }


def construct_layers(torch_model, inputs):
    global _BASIC_LAYERS
    for layer_id, operation in enumerate(torch_model.children()):
        basic_name = str(operation).partition('(')[0].lower()
        print(basic_name)
        layer_name = basic_name + f'_{layer_id}'
        if basic_name == 'sequential':
            inputs = construct_layers(operation, inputs)
        if basic_name in tf_layers_constructors:
            inputs = tf_layers_constructors[basic_name](layers=va_layers, torch_op=operation,
                                                        op_name=layer_name, input_tensor=inputs)
        else:
            inputs = getattr(va_layers,
                             _BASIC_LAYERS[basic_name]['tensorflow'])(name=layer_name)(inputs)
    return inputs


def construct_tf_model(torch_model, input_size):
    model_inputs = va_layers.Input(shape=input_size)
    x = construct_layers(torch_model, model_inputs)
    model = training.Model(inputs=model_inputs, outputs=x)
    return model


if __name__ == '__main__':
    model_torch = models.SimpleTorchModel()  # models.resnet18()
    model_torch.eval()
    fuse(model_torch)
    print(model_torch)

    img_size = (256, 512, 3)
    tf_model = construct_tf_model(model_torch, (*img_size[:2][::-1], img_size[-1]))
    tf_model.summary()

    with torch.no_grad():
        # N, C, H, W
        rand_sample = torch.rand(1, img_size[-1], *img_size[:2])

        torch_out = model_torch(rand_sample)
        torch_out = torch_out.numpy()
        # N, H, W, C
        torch_out = torch_out.transpose(0, 2, 3, 1)

    # TF
    # N, W, H, C
    x = tf.constant(rand_sample.detach().numpy().transpose(0, 3, 2, 1))
    # N, H, W, C
    tf_out = np.array(tf_model(x))
    tf_out = np.transpose(tf_out, (0, 2, 1, 3))

    print(torch_out.shape, tf_out.shape)
    print(np.abs(torch_out - tf_out).sum())
