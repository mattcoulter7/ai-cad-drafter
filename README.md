# STRUCTURAL DRAFTING

## 1. Walls
Train AI to identify outline of walls from architectural drawing
Extract lines/polylines from architectural walls and draw on **S-WALLS-UNDER** layer using **POLYLINE** command
Draw walls as a **block** with colour, linetype, lineweight properties all **By Block**

## 2. Openings and Lintels
"Openings": Doors, windows, etc. Lintels are required over windows/doors to support loads bearing on opening.
Train AI to recognise openings from Architectural .dwg/.dxf file
	All doors can be identified by the arch swing geometry
	All windows can be identified by 3 lines
Delete architectural window/door OR draw new line over top and delete architecturals later?
Draft line using **LINE** command in place of opening on **LINTEL** layer

## 3. Stud/Column
"Stud/Column": Timber studs to support the lintel at each end
Train AI to draw 90x90mm square using **RECTANGLE** command on **S-COL-POST** layer

# DEVELOPMENT ENVIRONMENT
To set up an Anaconda environment with Python 3.7 and TensorFlow GPU support, follow these steps:

## 1. Install Anaconda:
If you don't have Anaconda installed, download and install it from the official Anaconda website: https://www.anaconda.com/products/distribution

## 2. Create a New Environment:
Open a terminal or Anaconda Prompt and create a new environment named "tf_gpu_env" with Python 3.7 by running the following command:

```bash
conda create -n tf_gpu_env python=3.7
```

## 3. Activate the Environment:
Activate the newly created environment using the following command:
```bash
conda activate tf_gpu_env
```

## 4. Install Tensorflow
Following the instructions here to get Tensorflow installed https://www.tensorflow.org/install/pip

## 5. Verify Installation:
You can verify that TensorFlow has been installed successfully by running a Python script within your activated environment:

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
```
The `print(tf.config.list_physical_devices('GPU'))` line should output information about the available GPUs if TensorFlow GPU support is properly configured.

## 6. Install Requirements
```bash
pip install -r requirements_dev.txt
```

# RUNNING THE CODE
## TODO
