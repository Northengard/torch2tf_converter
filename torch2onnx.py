import os.path
from sys import argv
from argparse import ArgumentParser
import torch
import models


def parse_args(arg_list):
    parser = ArgumentParser(description='model creation script')
    parser.add_argument('-m', '--model-name', type=str, help='torchvision model name', default='resnet18')
    parser.add_argument('-i', '--input-shape', type=int, default=[3, 224, 224],
                        help='input tensors shape in CHW format, '
                             'if model has multiple inputs just pass '
                             'the shapes in required order one by one', nargs='+')
    parser.add_argument('-o', '--output-path', type=str, default=f'.{os.path.sep}',
                        help='path to save translated weights')
    arguments = parser.parse_args(arg_list)
    return arguments


def to_onnx(torch_model, model_name, inputs, outputs, input_names=('input',), output_names=('output',)):
    d_axes = {**{f'input{i}': {0: 'batch_size'} for i in range(len(input_names))},
              **{f'output{i}': {0: 'batch_size'} for i in range(len(output_names))}}
    torch.onnx.export(torch_model,  # model being run
                      args=inputs,  # model input (or a tuple for multiple inputs)
                      example_outputs=outputs,
                      # where to save the model (can be a file or file-like object)
                      f=f'{model_name}.onnx',
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=input_names,  # the model's input names
                      output_names=output_names,  # the model's output names
                      dynamic_axes=d_axes)


if __name__ == '__main__':
    if len(argv) > 1:
        args = parse_args(argv[1:])
    else:
        args = parse_args([])
    model = getattr(models, args.model_name)(pretrained=True)
    model.eval()

    print('model listing\n' + '-' * 90)
    print(model)

    dummy = torch.rand(1, *args.input_shape)

    model_output = model(dummy)

    to_onnx(model, args.model_name, dummy, outputs=model_output, input_names=['input'],
            output_names=[f'output{i}' for i in range(len(model_output))])
    print(f'model {args.model_name} converted to ONNX')

    torch.save(model, os.path.join(args.output_path, f'{args.model_name}.pth'))
