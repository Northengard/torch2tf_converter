# Torch2TF
What's done:
1) torch --> ONNX with simple onnx runtime checker
2) Torch --> TF via model construction with weight initialisation. But it steel use onnx as intermediate representation, 
although it is not required onnx-tf package anymore.

What is expected to be in this repo: <br>
It should contain at least 2 ways of weights mapping:
1) Torch --> ONNX --> TF
2) Torch --> TF via model construction with weight initialisation.

What to do next:
1) it's super important to dial with onnx opsets, for know 9 is used but 12 (or newer is available)
2) add support of various types of operators

# Prerequierements
This code was developed and tested on python V 3.8.10+.
(actually 3.8.10 and 3.9.5)

# Installation
There is no any special requierements yet. All dependencies may be installed via pip:

```pip install -r requierements.txt```

# Running
For now there is only torch2onnx.py translation available. 
### How to run:
1) To get script help simply run ```python torch2onnx.py -h```
2) To run the script you should basically define:
   1) model name (torchvision model name particularly, Please, see below how to add custom model) 
   2) its input tensor shape in C H W format
   3) output path
#### Example
```Shell
python torch2onnx.py --model-name resnet18 --input-shape 3 224 224 --output-path ./
```

# Testing 
Onnx Runtime test scripts is also available. It pass the dummy input through the onnx model so its possible to see it fails or not.

It can be launched like this:
```Shell
python torch2onnx.py --model-path ./resnet18.onnx --input-shape 3 224 224
```

Check [check_onnx.py](handlers/check_onnx.py) for more details
# Add custom model
to add the custom model support you should make your own model 'visible' from 'models' subpackage.
To deal with it simply add your model import to ```__init__``` properly or write the model inside this module and add related import into init.
### Example
TBA
