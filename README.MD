# Handwritten Math Problem Calculator - ML

## Description
The current project represents the machine learning part of a mobile application for recognition and solution of handwritten mathematical problems. It uses two datasets of handwritten symbols images for the training of a convolutional neural network. The mathematical symbols data is a subset of a dataset from  [Kaggle](https://www.kaggle.com/xainano/handwrittenmathsymbols/) and can be downloaded as used in this project from [here](https://drive.google.com/open?id=1IfTHWxdJQng55tjyheBnqu4GbX9lkcnN). The digit symbols are part of the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

## Structure
The repository contains the following files:

* split_dataset.py - script for splitting the raw dataset (when downloaded directly from Kaggle)
* train_math_symbols.py - script for training a convolutional neural network for the Kaggle dataset (from already split data in data/train and data/test directories
* model_mnist.h5 - the latest trained model from the train_mnist.py script
* model_math_symbols.h5 - the latest trained model from the train_math_symbols.py script
* label_map_math_symbols.npy - dictionary with indices of different classes mapped to their named label for the Kaggle dataset
* best_single_model.h5 - the highest accuracy trained single model (only from Kaggle for digits and symbols) so far
* best_single_model.h5.txt - the console output of the best trained model (only from Kaggle for digits and symbols) so far
* plot_model_math_symbols.py - script for creating a graphical representation of the convolutional neural network for the Kaggle dataset
* plot_model_mnist.py - script for creating a graphical representation of the convolutional neural network for the MNIST dataset

* server/server.py - script for segmenting and predicting each digit from already trained model