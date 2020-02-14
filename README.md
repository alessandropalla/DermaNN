# DermaNN

Skin cancer is one of the most aggressive tumor, with over 5 million newly diagnosed cases in the United States each year. Melanoma is the deadliest form of skin cancer, responsible for over 9,000 deaths each year.
An expert dermatologist can detect if a nevus is malignant or not. However, Melanoma can be easily treated only if it is detected in early stage. 

## Definitions (from ISIC):

Melanoma – malignant skin tumor, derived from melanocytes (melanocytic)
Nevus – benign skin tumor, derived from melanocytes (melanocytic)
Seborrheic keratosis – benign skin tumor, derived from keratinocytes (non-melanocytic)

## Neural Network
 
The aim of this project is to build an accurate skin lesion classification from images based on Neural Network. The strategy is to fine tuning the currently state-of-the-art convolutional Neural Networks to the specific task.

The InceptionV3 and the ResNet NN has been used and initialized with the ImageNet weigths. On top of that, we have built a two fully connected layers with a dorpout layer (dropout prob = 0.75) 

The output of the neural network is a three vector, one-hot encoded that represent the inferred class.

```
# add a global spatial average pooling layer
self.x = self.base_model.output
self.x = Dense(100, activation='relu')(self.x)
self.x = Dropout(0.75)(self.x)
# and a softmax layer with 3 classes
self.predictions = Dense(3, activation='softmax')(self.x)
```


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

- Python 3
- Tensorflow
- Keras
- Numpy
- Scikit

The code has been tested on Windows 10. Please report any issue with different setups.

### Dataset

The Neural network has been tested using the ISIC 2017: Skin Lesion Analysis Towards Melanoma Detection
 dataset. In order to make it works, you need to download the dataset and store it in the db db folder. You may want to split the images in train, validation and test.
 The ISIC dataset contains:
 - 2000 train images divided as following: 374 melanoma, 254 seborrheic keratosis and 1372 benign nevi
 - 150 validation images
 - 600 test images

## Running the tests

simply open a terminal an type:

'''
python DermaNN.python
'''

The script create a new neural network and perform training. It is possible also to load and train a previous saved network. 


## Deployment

TODO: 
- additional proprocessing
- lesion segmentation

## License

This project is licensed under the Apache2 License - see the [LICENSE.md](LICENSE.md) file for details

