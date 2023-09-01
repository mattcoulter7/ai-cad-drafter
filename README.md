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

## 1. Setup Conda Environment and Install Tensorflow 
Following the instructions here to get Tensorflow installed https://www.tensorflow.org/install/pip#windows-native_1

## 2. Install Requirements
```bash
pip install -r requirements_dev.txt
```

# RUNNING THE CODE
## TODO
