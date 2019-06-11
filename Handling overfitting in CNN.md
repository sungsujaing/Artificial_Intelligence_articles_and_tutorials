# Handling overfitting in CNN using keras ImageDataGenerator

During the CNN training, one of the most typical problems that one might occur includes the model overfitting. Also referred to as a high variance problem, overfitting represents when the model becomes "too complex" to fit the training data, thereby failing to generalize to the new set of data that has not been seen by the model during the training. 

As described in **<u>my other article</u>** "Model evaluation", the overfitting behavior can be detected by many diagnostic methods. During the CNN training using Keras framework, one may directly detect the model overfitting real-time by carefully monitoring various metrics of the learning model.

To handle the model overfitting, one may consider to implement:

- regularization and dropout
- batchnormalization
- data augmentation

Here in this article, detailed usage of the Keras data augmentation method is explored. The `ImageDataGenerator` class invovles ***data preprocessing, on-the-fly data augmentation and feeding in one step***. While these steps can be carried rather manually using other frameworks like openCV, the `ImageDataGenerator` class in Keras provides a really easy way to achieve the goal with the minimal efforts.

## Importing the module

```python
from keras.preprocessing.image import ImageDataGenerator
```

`ImageDataGenerator` generates batches of tensor with real-time data augmentation. The data will be looped over in batches. There are two typical types of a generator created based on the nature/structure of the available dataset.

1. When all images are already resized and loaded in memory (***in-memory***)
2. When images are directly fed from disk and resized on-the-fly (***on-disk***)

## Define a *in-memory* generator

Prior to building a in-memory generator, the dataset needs to be resized and splited into training and test sets. While the full list of arguments for preprocessing and augmentation can be found in the official keras document ([here](https://keras.io/preprocessing/image/)), some popular arguments for the in-memory generator include:

* `rotation_range`: the range of random rotation degree
* `horizontal_flip` and `vertical_flip`: True or False for random flips
* `preprocessing_funtion`: function that will be implied on each input after all other resize and augmentation. Typically used when building transfer learning models

example code snippit:

```python
train_datagen = ImageDataGenerator(rotation_range=20,
                                width_shift_range=0.2,
                                rotation_range=20,
                                horizontal_flip=True)
train_datagen.fit(X_train)  #(1)   
```

(1) necessary in flow?

## ImageDataGenerator`

n-memory (i.e., numpy arrays) and
on-disk images with the class ImageDataGenerator. 

`ImageDataGenerator`



* `rescale`: rescaling factor which is multiplied after applying all other transformation. (i.e. 1./255 for image normalization)