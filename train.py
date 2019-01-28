from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np

# dimensions of our images.
img_width, img_height = 45, 45
channels = 1

train_data_dir = 'data/train'
validation_data_dir = 'data/test'
nb_train_samples = 3000
nb_validation_samples = 800
epochs = 50
batch_size = 16
classes_count = 15

if K.image_data_format() == 'channels_first':
    input_shape = (channels, img_width, img_height)
else:
    input_shape = (img_width, img_height, channels)

cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), input_shape=input_shape, kernel_initializer='random_uniform'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(64, (5, 5)))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(128, (7, 7)))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Flatten())
cnn.add(Dense(128))
cnn.add(Activation('relu'))
cnn.add(Dropout(0.4))

cnn.add(Dense(classes_count))
# cnn.add(BatchNormalization())
cnn.add(Activation('softmax'))

cnn.compile(optimizer=Adam(lr=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1./127,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./127)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

cnn.fit_generator(
    train_generator,
    verbose=1,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

label_map = (train_generator.class_indices)
np.save('label_map.npy', label_map) 

cnn.save('model.h5')
