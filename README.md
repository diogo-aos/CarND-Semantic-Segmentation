# Semantic Segmentation
The aim of this project is to label the pixels belonging to the class "road", in a given image.
This task is called _semantic segmentation_.
A Fully Convolutional Network (FCN) was used.


## Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.


## Architecture

crop
no crop
regulizer
no regulizer

## Parameters
### 05_04_2019__10_25
epochs=50
learning_rate=0.001
drpout_rate = 0.5
batc-size = 10
no scaling no regulizer

### 03_04_2019__15_14
epochs=50
learning_rate=0.001
drpout_rate = 0.5
batc-size = 10
scaling, no regulizer

![image](report/03_inference.gif?raw=true "Title")

### 04_04_2019__15_14
epochs=50
learning_rate=0.001
drpout_rate = 0.5
batc-size = 10
with scaling and regularization fixed


## Results
### Dataset

### Collected video
foz
neuronios

## Considerations


### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note:** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

#### Example Outputs
Here are examples of a sufficient vs. insufficient output from a trained network:

Sufficient Result          |  Insufficient Result
:-------------------------:|:-------------------------:
![Sufficient](./examples/sufficient_result.png)  |  ![Insufficient](./examples/insufficient_result.png)

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
