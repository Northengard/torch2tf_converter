import numpy as np

import tensorflow as tf
from tensorflow.python.keras.engine import training
from tensorflow.python.keras import layers
# from tensorflow.python.keras.layers import VersionAwareLayers

from handlers.onnx_graph_parser import OnnxInterModel
from tensorflow.keras import activations


def do_pad(pad_size, input_tensor, name_prefix):
    if sum(pad_size) > 0:
        input_tensor = layers.ZeroPadding2D(padding=pad_size, name=name_prefix + '_pad')(input_tensor)
    return input_tensor


def get_conv1d(keras_layers, conv_parameters, is_depthwise, use_bias, input_tensor):
    raise NotImplementedError('TBA')


def get_conv2d(keras_layers, conv_parameters, is_depthwise, use_bias, input_tensor):
    weights = [conv_parameters['weights'][0].transpose(2, 3, 0, 1) if is_depthwise else
               conv_parameters['weights'][0].transpose(2, 3, 1, 0)]
    if use_bias:
        weights.append(conv_parameters['weights'][1])

    if is_depthwise:
        input_tensor = keras_layers.DepthwiseConv2D(kernel_size=conv_parameters['kernel_shape'],
                                                    strides=conv_parameters['strides'],
                                                    padding='valid',
                                                    use_bias=use_bias,
                                                    dilation_rate=conv_parameters['dilations'],
                                                    activation=None,
                                                    name=conv_parameters['name'],
                                                    weights=weights)(input_tensor)
    else:
        out_channels = weights[0].shape[-1]
        input_tensor = keras_layers.Conv2D(filters=out_channels,
                                           kernel_size=conv_parameters['kernel_shape'],
                                           strides=conv_parameters['strides'],
                                           padding='valid',
                                           use_bias=use_bias,
                                           dilation_rate=conv_parameters['dilations'],
                                           activation=None,
                                           name=conv_parameters['name'],
                                           weights=weights)(input_tensor)
    return input_tensor


def get_conv3d(keras_layers, conv_parameters, is_depthwise, use_bias, input_tensor):
    raise NotImplementedError('TBA')


def get_conv(keras_layers, op_params, op_name, input_tensor):
    use_bias = len(op_params['weights']) > 1
    is_depthwise = op_params['group'] > 1

    input_tensor = do_pad(op_params['pads'], input_tensor, op_name)
    # op_name = conv_parameters['op_type'].capitalize()
    weights_size = len(op_params['weights'][0].shape)
    if weights_size == 3:
        # op_name += '1D'
        input_tensor = get_conv1d(keras_layers, op_params, is_depthwise, use_bias, input_tensor)
    elif weights_size == 4:
        # op_name += '2D'
        input_tensor = get_conv2d(keras_layers, op_params, is_depthwise, use_bias, input_tensor)
    elif weights_size == 5:
        # op_name += '3D'
        input_tensor = get_conv3d(keras_layers, op_params, is_depthwise, use_bias, input_tensor)

    return input_tensor


def get_dence(keras_layers, op_params, op_name, input_tensor):
    weights = [w.transpose() for w in op_params['weights']]
    use_bias = len(weights) > 1
    out_features = weights[0].shape[1]
    input_tensor = keras_layers.Dense(units=out_features,
                                      activation=None,
                                      use_bias=use_bias,
                                      name=op_name,
                                      weights=weights,
                                      kernel_regularizer=None,
                                      bias_regularizer=None,
                                      activity_regularizer=None,
                                      kernel_constraint=None,
                                      bias_constraint=None)(input_tensor)
    return input_tensor


def get_flatten(keras_layers, op_params, op_name, input_tensor):
    # input_tensor = keras_layers.Permute((3, 2, 1))(input_tensor)
    input_tensor = keras_layers.Flatten(name=op_name)(input_tensor)
    return input_tensor


def construct_maxpool(keras_layers, op_params, op_name, input_tensor):
    if isinstance(op_params['padding'], tuple):
        if sum(op_params['padding']) > 0:
            input_tensor = keras_layers.ZeroPadding2D(padding=op_params['padding'], name=op_name + '_pad')(input_tensor)
    elif op_params['padding'] > 0:
        input_tensor = keras_layers.ZeroPadding2D(padding=op_params['padding'], name=op_name + '_pad')(input_tensor)
    keras_layers.MaxPool2D(pool_size=op_params.kernel_size, strides=op_params.stride, padding='valid')
    return input_tensor


def get_global_avg_pool(keras_layers, op_params, op_name, input_tensor):
    return keras_layers.GlobalAvgPool2D(name=op_name)(input_tensor)


def get_maxpool(keras_layers, op_params, op_name, input_tensor):
    input_tensor = do_pad(op_params['pads'], input_tensor, op_name)
    input_tensor = keras_layers.MaxPool2D(pool_size=op_params['kernel_shape'],
                                          padding='valid',
                                          strides=op_params['strides'])(input_tensor)
    return input_tensor


def get_identity(keras_layers, torch_op, op_name, input_tensor):
    return tf.identity(input_tensor)


def get_add(inp1, inp2):
    x = layers.Add()([inp1, inp2])
    return x


def get_concat(inp1, inp2, axis=-1):
    x = layers.Concatenate(axis=axis)([inp1, inp2])
    return x


def get_activation(inp, layer_properties):
    activation_name = layer_properties['op_type']
    kwargs = {}
    values = layer_properties.get('values', None)
    if values is not None:
        kwargs = {'max_value': values[-1]}
    if activation_name == 'clip':
        activation_name = 'relu'
    inp = getattr(activations, activation_name)(inp, **kwargs)
    return inp


def get_upsample(keras_layers, op_params, op_name, input_tensor):
    if np.all(op_params['scale'] >= 1):
        input_tensor = keras_layers.UpSampling2D(size=op_params['scale'], name=op_name,
                                                 interpolation=op_params['mode'])(input_tensor)
    else:
        size = (int(input_tensor.shape[1] * op_params['scale'][0]),
                int(input_tensor.shape[2] * op_params['scale'][1]))
        input_tensor = tf.image.resize(images=input_tensor,
                                       size=size,
                                       method=op_params['mode'],
                                       name=op_name)
    return input_tensor


tf_layers_constructors = {'conv': get_conv,
                          'gemm': get_dence,
                          'flatten': get_flatten,
                          'identity': get_identity,
                          'globalaveragepool': get_global_avg_pool,
                          'maxpool': get_maxpool,
                          'add': get_add,
                          'concat': get_concat,
                          'upsample': get_upsample,
                          }


def construct_layers(layers_parameters, connections, inputs):
    return inputs


def construct_tf_model(parsed_model):
    model_layers, model_connections = parsed_model.nodes, parsed_model.adj_list

    input_name, input_size = list(parsed_model.model_input.items())[0]
    output_names = list(parsed_model.model_output.keys())
    # TODO: findout how to deal with N dimention because for keras Input it is redundant
    input_size = [dim for dim in input_size if dim > 1]
    # to HWC
    input_size = input_size[1:] + input_size[:1]
    model_inputs = layers.Input(shape=input_size, name=input_name)
    model_outputs = list()
    x = model_inputs
    last_node_output = model_layers[0]['output']
    skip_connections = dict()
    for layer in model_layers:
        layer_name = layer['name']
        layer_id = layer['node_id']
        if not np.any(np.isin(last_node_output, layer['input'])):
            if layer_id in skip_connections.keys():
                skip = x
                skip_connections[model_connections[layer_id - 1][0]] = skip
                x = skip_connections.pop(layer_id)
        if len(layer['input']) > 1:
            x = tf_layers_constructors[layer['op_type']](x, skip_connections.pop(layer_id))
        else:
            if layer['op_type'] in tf_layers_constructors:
                x = tf_layers_constructors[layer['op_type']](keras_layers=layers, op_params=layer,
                                                             op_name=layer_name, input_tensor=x)
            else:
                x = get_activation(x, layer)
        if len(model_connections[layer_id]) > 1:
            for connect in model_connections[layer_id][1:]:
                skip = x
                skip_connections[connect] = skip
        if layer['output'][0] in output_names:
            model_outputs.append(x)
            x = skip_connections.get(layer_id + 1, None)
        last_node_output = layer['output']
    model = training.Model(inputs=model_inputs, outputs=model_outputs)
    return model


def to_tflite(tf_model_name, tflite_optimization=None):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_name)
    if tflite_optimization is not None:
        if 'optimizations' in tflite_optimization:
            converter.optimizations = tflite_optimization['optimizations']
            print("TFLite optimizations: ", converter.optimizations)
        if 'target_spec.supported_types' in tflite_optimization:
            converter.target_spec.supported_types = tflite_optimization['target_spec.supported_types']
            print("TFLite target_spec.supported_types: ", converter.target_spec.supported_types)
        if 'representative_dataset' in tflite_optimization:
            converter.representative_dataset = tflite_optimization['representative_dataset']
            print("Assigned representative_dataset")
    tflite_model = converter.convert()

    # Save the model.
    tflite_model_name = tf_model_name.split('.')[0] + '.tflite'
    # tflite_model_name += '.tflite'
    # check_filename(tflite_model_name)
    with open(tflite_model_name, 'wb') as f:
        f.write(tflite_model)
