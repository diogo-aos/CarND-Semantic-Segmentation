# Semantic Segmentation
The aim of this project is to label the pixels belonging to the class "road", in a given image.
This task is called _semantic segmentation_.
A Fully Convolutional Network (FCN) was used.


## Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.


## Architecture
Essentially, this project reproduced the architecture of X.
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

```bash
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


A `main2.py` file was developed to make it easier to 


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

The best results were produced by *model B*.
*Model A* produced similar results, but the labeled pixels were more disperse.
*Model C* produced results with 

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


