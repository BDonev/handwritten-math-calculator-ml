from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model

import numpy as np
import os

def get_samples_count(data_dir):
    files_count = sum([len(files) for r, d, files in os.walk(data_dir)])
    return files_count

# dimensions of our images.
img_width, img_height = 45, 45
channels = 1

train_data_dir = 'data/math_symbols/train'
validation_data_dir = 'data/math_symbols/test'
train_samples = get_samples_count(train_data_dir)
validation_samples = get_samples_count(validation_data_dir)
print("Number of training samples: " + str(train_samples))
print("Number of validation samples: " + str(validation_samples))
epochs = 12
batch_size = 64
classes_count = 5
train_iterations = train_samples / batch_size
validation_iterations = validation_samples / batch_size

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
cnn.add(BatchNormalization())
cnn.add(Activation('softmax'))

cnn.compile(optimizer=Adam(lr=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

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
    steps_per_epoch=train_iterations,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_iterations)

label_map = (train_generator.class_indices)
np.save('label_map_math_symbols.npy', label_map)
cnn.save('model_math_symbols.h5')

score = cnn.evaluate_generator(validation_generator, steps=validation_iterations, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(cnn.summary())

plot_model(cnn, to_file='model_math_symbols.png')
