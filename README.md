# Semantic Segmentation
The aim of this project is to label the pixels belonging to the class "road", in a given image.
This task is called _semantic segmentation_.
A Fully Convolutional Network (FCN) was used.


## Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.


## Architecture

## Parameters

Model A = 05_04_2019__10_25
Model B = 03_04_2019__15_14
Model C = 04_04_2019__15_14

| Parameter     | Model A | Model B | Model C |
| ---------     | -----   | ------- | ------- |
| Epochs        | 50      | 50      | 50      |
| Learning rate | 0.001   | 0.001   | 0.001   |
| Dropout rate  | 0.5     | 0.5     | 0.5     |
| Batch size    | 10      | 10      | 10      |
| Scaling layer | No      | Yes     | Yes     |
| Regulizer     | No      | No      | Yes     |


## Results
### Dataset
![Model A](report/05_inference.gif?raw=true "Title")

![Model B](report/03_inference.gif?raw=true "Title")

![Model C](report/04_inference.gif?raw=true "Title")

### Collected video
#### Foz
![Model A](report/05_processed_model50_foz_crop.gif?raw=true "Title")
![Model B](report/03_processed_model50_foz_crop.gif?raw=true "Title")
![Model C](report/04_processed_model50_foz_crop.gif?raw=true "Title")

#### Neuronios
![Model A](report/05_processed_model50_neuronios_crop.gif?raw=true "Title")
![Model B](report/03_processed_model50_neuronios_crop.gif?raw=true "Title")
![Model C](report/04_processed_model50_neuronios_crop.gif?raw=true "Title")

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
 
