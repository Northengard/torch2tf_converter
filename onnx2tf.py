import os
from sys import argv
from argparse import ArgumentParser

import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


def parse_args(arg_list):
    def str2bool(flag):
        if flag.lower() in ['true', 't', 'yes', '1']:
            return True
        else:
            return False

    export2tf_choices = ['True', 'False', 'T', 'F', 'Yes', 'No', '1', '0']
    export2tf_choices = export2tf_choices + [flag.lower() for flag in export2tf_choices]
    parser = ArgumentParser(description='model creation script')
    parser.add_argument('-m', '--model-path', type=str, help='onnx model weights path')
    parser.add_argument('-o', '--output-path', type=str, default='',
                        help='path to save translated weights')
    parser.add_argument('-e', '--export-to-lite', type=str, choices=export2tf_choices, default='no',
                        help='additionally export created tensorflow model to tf-lite')
    arguments = parser.parse_args(arg_list)
    arguments.export_to_lite = str2bool(arguments.export_to_lite)
    return arguments


def export_tf_to_tflite(tf_lite_save_path, tf_model_load_path):
    # make a converter object from the saved tensorflow file
    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(tf_model_load_path)

    converter.experimental_new_converter = True

    # I had to explicitly state the ops
    converter.target_spec.supported_ops = [tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.compat.v1.lite.OpsSet.SELECT_TF_OPS]

    tf_lite_model = converter.convert()
    # Save the model.
    with open(tf_lite_save_path, 'wb') as f:
        f.write(tf_lite_model)
    print(f"Tensorflow-Lite model {os.path.basename(tf_lite_save_path).split('.')[0]}"
          f" has been created at {tf_lite_save_path}")


if __name__ == '__main__':
    if len(argv) > 1:
        args = parse_args(argv[1:])
    else:
        args = parse_args([])
    onnx_model = onnx.load(args.model_path)
    onnx.checker.check_model(onnx_model)
    print('ONNX loaded')

    model_name = os.path.basename(args.model_path).split('.')[0]
    tf_model_save_path = os.path.join(args.output_path, model_name + '.pb')
    tf_rep = prepare(onnx_model)  # creating TensorflowRep object
    tf_rep.export_graph(tf_model_save_path)  # save it

    print(f'Model {model_name} has been converted to Tensorflow')

    if args.export_to_lite:
        tf_lite_path = os.path.join(args.output_path, model_name + '.tflite')
        export_tf_to_tflite(tf_lite_path, tf_model_save_path)
