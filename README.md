# CS5242_Project - Vehicle detection

## Problem statement 
The integration of machine learning and vehicular perception enables various active safety measures in autonomous driving. However, numerous issues in a dynamic urban environment, such as suboptimal weather conditions, performance of objection detection, perception and decision making accuracy in autonomous driving, makes it dangerous and costly to collect data from the real world. Therefore, we wish to investigate the feasibility of training vehicle detection via the use of image data generated from video games.

## Data collection / Annotation
[Use this drive for data](https://drive.google.com/drive/folders/1VC1MVYZWxWdbyIPAjdu7vfWq-vLzg06k?usp=sharing)


In this work, we aim to collect a traffic image dataset from a 2013 open world video game “Grand Theft Auto V” (GTA V). GTA V has detailed and diverse game scenes, as it’s open world was modelled after the city of Los Angeles. Image dataset can be obtained by recording a gameplay video and converting it into image sequences. To perform supervised learning for vehicle detection, we shall label the data using bounding box annotation, which defines the location of the target vehicle by rectangular boxes. 

**TODO**:
- Collect gameplay videos of GTA V driving scenes; 
- Convert video into images; 
- Apply bounding box data annotation/labelling; 
- Select suitable data augmentation algorithms and use them in our collected dataset; 
- Annotate the augmented dataset.

## Data Augmentations
Since we only have 500 labelled images, it is essential to incorporate data augmentations to create more variability in available data. The following list provides the types of image augmentations performed.

- Blur (Gaussian, Average, Median)
- Brightness variation with per-channel probability
- Adding Gaussian Noise with per-channel probability
- Random dropout of pixels

Examples of image blur, brightness/ color jitter and Dropout are shown below.
[ Attach figure here]()

## Data Split
Next, we shall split the collected dataset into 80% training set and 20% testing set, and implement a vehicle detection algorithm using Convolutional Neural Network on the training set. 

## Performance evaluation
We use mean average precision (mAP) as the performance metric here.  
**Average Precision:** It is the average precision over multiple IoU values.  
**mAP:** It is average of AP over all the object categories.  

## Neural Network Architecture
We experimented with MLP, CNN, R-CNN. Network architectures/ deep learning frameworks for each of the models are as follows:
- [MLP framework figure]()
- [CNN framework figure]()
- [R-CNN framework figure]()

| Model      	| Number of layers  	| Average Precision   |  mAP               	| Running time  |   Device (cpu/gpu)  |
|-----------	|-------------------	|-------------------- |---------------------|---------------|-------------------	|
| MLP         |                     |                     |                     |               |                     |
| CNN         |                     |                     |                     |               |                     |
| R-CNN       |                     |                     |                     |               |                     |


## Performance Analysis
- Explain the overall performance;
- Why model A is the best?
- Why model B is the worst?
- Compare running time, pros and cons of each model.
- What is the effect of the number of layers to the accuracy of object detection?
- Explain the loss againest iterations;
- Overfitting or Underfitting?
- Future insights 
  - Accurate object detection relies on big data to avoid overfitting, while we have only a limited dataset. To handle this issue, we shall implement several data augmentation algorithms: a data-space solution to the problem of limited data.
  - Most object detection models were trained from relatively ideal scenarios, such as sunny days. Therefore, we will also need to quantify how our model performs under various bad weather conditions: heavy rain, snowy, foggy.

## Agnostic learning 
Perform agnostic study, eg. adding rain drops onto the original dataset; Evaluate its impact on performance.


## Prerequisites
**Python 3.6+**
   - `argpase`
   - `itertools`
   - `numpy`
   - `scipy`
   - `sklearn`
   - `matplotlib`
   - `torch`


## Reproducbility 
- src 
- data


- **Data**  - Collected data 
- `data.py` - Data pre-processing 
- `evl.py` - algorithm accuracy evaluation
- `config.py` - Set parameters for our model (eg. hyperparameters, number of hidden layers, sample size, etc. )
- `methods.py` - the implementation of all algorithms
- `main.py` - Run some examples of our algorithm
- 'requirement.md' - package requirement
- readme
- etc. 

## Parameters

| Parameter    | Type | Description                      | Options            |
| -------------|------| ---------------------------------| -------------      |
| `n`          |  int |  number of images for training   |      -             |
| `s`          |  int |  number of samples               |      -             |
| `batch`      |  int |  number of batch size            | - |
| `choice`     |  str |  choice which model to run       |`MLP`, `CNN`, `R-CNN`|

## Running a simple demo

The simplest way to try out CauchyEst is to run a simple example:
```bash
$ git clone https://github.com/_/_.git
$ cd  _/
$ python demo.py
```

## Runing as a command

```bash
$ pip install git+git://github.com/_/_
$ cd _
$ python main.py 
```

## References
[1] https://github.com/keshik6/KITTI-2d-object-detection
[2] http://www.cvlibs.net/datasets/kitti/
