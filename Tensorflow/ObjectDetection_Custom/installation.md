# Installation

**Adapted from [Tensorflow Objection Detection API installation document](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

## Dependencies

Tensorflow Object Detection API depends on the following libraries:

*   Protobuf 3.0.0
*   Python-tk
*   Pillow 1.0
*   lxml
*   tf Slim (which is included in the "tensorflow/models/research/" checkout)
*   Jupyter notebook
*   Matplotlib
*   Tensorflow 1.14
*   Cython
*   contextlib2
*   cocoapi

For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/). A typical user can install
Tensorflow using one of the following commands:

``` bash
# For CPU
pip install tensorflow==1.14
# For GPU
pip install tensorflow-gpu==1.14
```

The remaining libraries can be installed on Ubuntu 18.04 using via apt-get and pip install:

``` bash
sudo apt-get install protobuf-compiler 
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
```

**Note**: To install protobuf-compiler on Windows and Mac, see [other installation](#protobuf-compiler installation on Windows and MacOS) installation.

## COCO API installation

Download the
[cocoapi](https://github.com/cocodataset/cocoapi) and
copy the pycocotools subfolder to the tensorflow/models/research directory if
you are interested in using COCO evaluation metrics. The default metrics are
based on those used in Pascal VOC evaluation. To use the COCO object detection
metrics add `metrics_set: "coco_detection_metrics"` to the `eval_config` message
in the config file. To use the COCO instance segmentation metrics add
`metrics_set: "coco_mask_metrics"` to the `eval_config` message in the config
file.

**Ubuntu/MacOS instructions**

``` bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/
```
**Windows instruction**

``` bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
Note that, according to the [package’s instructions](https://github.com/philferriere/cocoapi#this-clones-readme), Visual C++ 2015 build tools must be installed and on your path. If they are not, make sure to install them from [here](https://go.microsoft.com/fwlink/?LinkId=691126).
```

## Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the tensorflow/models/research/ directory:

``` bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

**Note**: If you're getting errors while compiling, you might be using an incompatible protobuf compiler. If that's the case, use the following manual installation

## protobuf-compiler installation on Windows and MacOS

**If you are on Windows:**

Head to the [protoc releases page](https://github.com/protocolbuffers/protobuf/releases) and download the latest *-win32.zip release (e.g. protoc-3.5.1-win32.zip)

Create a folder in e.g. C:\protobuf. 

Extract the contents of the downloaded *-win32.zip, inside C:\protobuf

Add C:\protobuf\bin to your Path environment variable 

**If you are on MacOS:**

If you have homebrew, download and install the protobuf with
```brew install protobuf```

Run the compilation process again:

``` bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

## Add Libraries to PYTHONPATH

When running locally, the tensorflow/models/research/ and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
tensorflow/models/research/:


``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file, replacing \`pwd\` with the absolute path of
tensorflow/models/research on your system.

# Testing the Installation

You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:

```bash
python object_detection/builders/model_builder_test.py
```