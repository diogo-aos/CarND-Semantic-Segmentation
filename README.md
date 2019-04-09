# Semantic Segmentation
The aim of this project is to label the pixels belonging to the class "road", in a given image.
This task is called _semantic segmentation_.
A Fully Convolutional Network (FCN) was used.


## Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.


## Architecture
Essentially, this project reproduced the architecture of the original paper [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).
X.
A VGG16 network pre-trained on ImageNet was converted to a Fully Convolutional Network.
The last fully connected layers were replaced by 1x1 convolutions with depth equal to the number of classes (in this case 2, road/not-road).
Afterwards, transposed convolutions are used for upsampling.
Inbetween transposed convolutions, skip layers are used by adding 1x1 convolutions of layers from VGG16.
A kernel regulizer was added to each convolution and transposed convolution.
Different values for this regularization were tried.
The final, better performing model, does not include these regulizers in the loss function.

## Implementation
The model is implemented in `model.py`.
To use it, we only need to import this module's functions.
The unit tests are still executed in `main.py`.
The model parameters are now received as command line arguments.
The available arguments are: epochs, dropout rate, learning rate, batch size, directory to save results (metadata of model, loss at each epoch, the model itself).

```
> python main.py --help
usage: main.py [-h] -E EPOCHS -L LEARNING -B BSIZE -D DROPOUT -M RDIR -N NOTES
               [--validate VALIDATE]

optional arguments:
  -h, --help           show this help message and exit
  -E EPOCHS            epochs
  -L LEARNING          dropout rate
  -B BSIZE             batch size
  -D DROPOUT           dropout rate
  -M RDIR              directory for results
  -N NOTES             notes
  --validate VALIDATE  validate model with tests
```


A `main2.py` file was developed to make it easier to test the model on different stages of the training.
From 20 epochs, the model is saved every 10 epochs.
A `process.py` file receives models from a given architecture and videos to be processed.


## Parameters
<!---
Model A = 05_04_2019__10_25
Model B = 03_04_2019__15_14
Model C = 04_04_2019__15_14
--->

Several parameters were experimented with.
This report will focus on 3 architectures, described on the table below.
Models for epochs 20,30,40 were also saved and tested on.
Their performance was inferior and the results are not included.


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
![Model A loss](report/05_loss.png?raw=true "Title")
![Model B loss](report/03_loss.png?raw=true "Title")
![Model C loss](report/04_loss.png?raw=true "Title")


Inference results on the test dataset can be observed in this [Youtube video](https://youtu.be/7sT-jydE_E8).

The best results were produced by *model B*.
*Model A* produced similar results, but the labeled pixels were more disperse.
*Model C* produced results with a lot more false positives.

### Collected video
2 videos were recorded using 20MP Sony Exmor RS sensor.
Both videos were recorded with the phone handheld.
The windshield is dirty on both videos.
These videos are called `Foz` and `Neuronios`.
The light differs significantly on both videos.
The performance of the three models above in both videos can be seen in this [Youtube video](https://youtu.be/0TAYRCinuVs).

The performance in the `Neuronios` video is awful.
Model A and B perform a little better in the sense there are a lot fewer false positives.
The light conditions in `Foz` are more similar to those in the training dataset, which might explain the better results.
Model B performed best in `Foz`, as in the test dataset.
Model A performs slightly worse than B, but the labeled pixels are more disperse, as previously observed.
Model C, once again, has lots of false positives.

Both videos had to be cropped to keep the aspect ratio after they go through the pipeline.
Without maintaining the aspect ratio, the results are completely unusable.

## Reflections
Some tools, that don't involve changing the model itself, could prove useful improving the performance.
The first obvious tool is to use some pre-processing.
I believe pre-processing would be specially impactful on the collected dataset (the 2 homemade videos).
In this project, no pre-processing was used.
Another tool is dataset augmentation.



