# Torch2TF
This repo contains:
1) Torch --> ONNX --> TF (via onnx-tf) 
2) Torch --> ONNX --> TF via model reconstruction from onnx proto. (proposed method)

### Advantages of the second approach:
1) does not requires [onnx-tf](https://github.com/onnx/onnx-tensorflow) (but steel use onnx for now (to get model 
structure with weights))
2) do not produce redundant permutation layers, because it permutes network input to HWC format
instead of permute layers input every time.
3) exploit [layers fusion](https://pytorch.org/tutorials/recipes/fuse.html) without accuracy loss

## Additional features:
1) It exploits onnx opset version 9. (There is no particular reason, it's just the start point, selected because of 
convenience)
2) As it use onnx so 
   1) intermediate onnx optimisation may be applied, such as 
   [onnx_simplifier](https://github.com/daquexian/onnx-simplifier), etc.
   2) onnx runtime checker is also available

# How it works
Technically it uses the same way to export model as [onnx-tf](https://github.com/onnx/onnx-tensorflow),
but Torch2TF avoids additional nodes production and does not turn model into 
[tf-proto](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/how_tos/tool_developers/index.md) 
format.

## Why not just use [onnx-tf](https://github.com/onnx/onnx-tensorflow)?
As it has been [mentioned](#advantages-of-the-second-approach), [onnx-tf](https://github.com/onnx/onnx-tensorflow) 
produces  additional layers that are not presented in the model originally, so its performance becomes worse.
The diagram below shows an example of resnet-18 translated with [onnx-tf](https://github.com/onnx/onnx-tensorflow) 
and method, proposed in this repo:

![onnx-tf_vs_proposed_example](figures/onnx-tf_vs_proposed_example.png)

It's easy to see that onnx-tf tries to keep the input dimensions the same in the pretty wierd manner, by inserting the 
permutation (transpose) layer each time the channels sensitive (HWC or CHW matters) operator need to be applied.

So, just imagine how it slows the performance, by the way it's possible to find it out, because both method of model
conversion are available, check [Running](#running) for details.

## Supported operators
For now, it converts model using **channels last format only** because it's the most common for tensorflow.<br>
Supported for now operators list(refers to proposed method):
1) operators:
   1) conv2d
      1) common
      2) depthwise
   2) dense (linear)
   3) pooling
      1) maxpool
      2) global avg pool (adaptive avg pool)
   4) activations
   5) add
   6) concat
   7) flatten
   8) upsample
2) skip-connections
3) branching

TODO:
1) support multi-input models
2) expand number of supported operators
3) support others onnx opsets or get rid of onnx (find another way to turn model into proto format)
4) conversion check

# Prerequierements
This code was developed and tested on python V 3.8.10+.
(actually 3.8.10 and 3.9.5)

# Installation
There is no any 'special' requierements yet. All dependencies may be installed via pip:

```pip install -r requierements.txt```

# Running
## Traditional way (using onnx-tf)
### How to run:
1) add your model to models module (more details below)
2) convert model to onnx first:
   1) To get script help simply run ```python converters/torch2onnx.py -h```
   2) To run the script you should basically define:
      1) model name (torchvision model name particularly, Please, see below how to add custom model) 
      2) its input tensor shape in C H W format
      3) output path
3) use ```traditional/onnx2tf.py``` to make onnx-tf convertion 
   1) you should define model path and output path to save tf model
#### Example
```Shell
python torch2onnx.py --model-name resnet18 --input-shape 3 224 224 --output-path ./
```
## Proposed method
Check [main.py](main.py) for details

# Testing 
Onnx Runtime test scripts is also available. It pass the dummy input through the onnx model so its possible to see it 
fails or not. Later adequate output tensors comparison would be added.
It can be launched like this:
```Shell
python torch2onnx.py --model-path ./resnet18.onnx --input-shape 3 224 224
```

Check [check_onnx.py](handlers/check_onnx.py) for more details

# Add custom model
to add the custom model support you should make your own model 'visible' from 'models' subpackage.
To deal with it simply add your model import to ```__init__``` properly or write the model inside this module and add 
related import into init.
### Example
TBA
