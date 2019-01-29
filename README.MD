# Handwritten Math Problem Calculator - ML

## Description
The current project represents the machine learning part of a mobile application for recognition and solution of handwritten mathematical problems. It uses a dataset of handwritten symbols images for the training of a convolutional neural network. The data is a subset of a dataset from  [Kaggle](https://www.kaggle.com/xainano/handwrittenmathsymbols/) and can be downloaded as used in this project from [here](https://drive.google.com/open?id=1IfTHWxdJQng55tjyheBnqu4GbX9lkcnN).
## Structure
The repository contains the following files:

* split_dataset.py - script for splitting the raw dataset (when downloaded directly from Kaggle)
* train.py - script for training a convolutional neural network (from already split data in data/train and data/test directories
* segment.py - script for segmenting and predicting each digit from already trained model
* model.h5 - the latest trained model from the train.py script
* label_map.npy - dictionary with indices of different classes mapped to their named label
* photo_1.jpg - test image for executing the current segment.py script
* best_model.h5 - the highest accuracy trained model so far
* best_training.txt - the console output of the best trained model so far
