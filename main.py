import os
from sys import argv
from argparse import ArgumentParser

import torch

from handlers import OnnxInterModel, fuse

import models
from converters.torch2onnx import to_onnx
from converters.weight_transfer import construct_tf_model, to_tflite


def parse_args(arg_list):
    parser = ArgumentParser(description='model creation script')
    parser.add_argument('-m', '--model-name', type=str, help='torchvision model name', default='resnet18')
    parser.add_argument('-w', '--model-weights', type=str, help='torch model weights', default='resnet18.pt')
    parser.add_argument('-i', '--input-shape', type=int, default=[3, 224, 224],
                        help='input tensors shape in CHW format, '
                             'if model has multiple inputs just pass '
                             'the shapes in required order one by one', nargs='+')
    parser.add_argument('-o', '--output-path', type=str, default=f'.{os.path.sep}',
                        help='path to save translated weights')
    arguments = parser.parse_args(arg_list)
    return arguments


if __name__ == '__main__':
    args = parse_args(argv[1:])
    torch_model = getattr(models, args.model_name)()
    input_tensor = torch.rand(1, *args.input_shape)
    if len(args.model_weights) > 0:
        snapshot = torch.load(args.model_weights, map_location='cpu')
        if 'state_dict' in snapshot:
            snapshot = snapshot['state_dict']
        torch_model.load_state_dict(snapshot)
    torch_model.eval()
    fuse(torch_model)
    outputs = torch_model(input_tensor)

    to_onnx(torch_model, os.path.join(args.output_path, args.model_name), input_tensor, outputs, ['input'],
            output_names=[f'output{i}' for i in range(len(outputs))])

    model_parser = OnnxInterModel(onnx_path=os.path.join(args.output_path, args.model_name + '.onnx'),
                                  use_simplify=False)

    tf_model = construct_tf_model(model_parser)
    tf_model_path = os.path.join(args.output_path, args.model_name + '.pb')

    tf_model.save(tf_model_path)
    to_tflite(tf_model_path)
    print('done')
