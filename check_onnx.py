from sys import argv
from argparse import ArgumentParser
import onnxruntime as ort
import numpy as np


def parse_args(arg_list):
    parser = ArgumentParser(description='check onnx weights script')
    parser.add_argument('-m', '--model-path', type=str, default='resnet18.onnx')
    parser.add_argument('-i', '--input-shape', nargs='+', default=[3, 224, 224],
                        help='input tensors shape in CHW format, '
                             'if model has multiple inputs just pass the shapes in required order one by one')
    arguments = parser.parse_args(arg_list)
    return arguments


if __name__ == '__main__':
    if len(argv) > 1:
        args = parse_args(argv[1:])
    else:
        args = parse_args([])

    session = ort.InferenceSession(args.model_path)
    input_names = [inp.name for inp in session.get_inputs()]
    input_shape = args.input_shape
    dummy = [np.random.randn(1, *t_shape).astype(np.float32)
             for t_shape in np.array(input_shape).reshape(-1, 3).astype(int)]
    outputs = session.run([], dict(zip(input_names, dummy)))
    # print(outputs)
    print('output shapes:')
    print([out.shape for out in outputs])
