import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
import random
import shutil
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model

# Function to generate data for training and validation


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24, 24), class_mode='categorical'):
    """
    Generates batches of tensor image data with real-time data augmentation.

    Parameters:
        dir (str): Directory where the images are stored.
        gen (ImageDataGenerator): Generator for data augmentation and preprocessing.
        shuffle (bool): Whether to shuffle the images.
        batch_size (int): Number of images in each batch.
        target_size (tuple): Desired dimension (height, width) to resize images.
        class_mode (str): Determines the type of label arrays that are returned.

    Returns:
        DirectoryIterator: Iterator over the data generated from the directory.
    """
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)


# Batch size and target size configuration
BS = 32  # Batch size for the network
TS = (24, 24)  # Target image dimensions

# Creating training and validation generators
train_batch = generator('data/train', shuffle=True,
                        batch_size=BS, target_size=TS)
valid_batch = generator('data/valid', shuffle=True,
                        batch_size=BS, target_size=TS)

# Calculating steps per epoch and validation steps
SPE = len(train_batch.classes) // BS  # Steps per epoch
VS = len(valid_batch.classes) // BS  # Validation steps
print(SPE, VS)  # Print steps to monitor setup

# Building the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.25),  # Dropout for regularization
    Flatten(),  # Flatten the output before feeding into dense layers
    Dense(128, activation='relu'),
    Dropout(0.5),  # Further dropout for regularization
    # Output layer with softmax activation for 2 classes
    Dense(2, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit_generator(train_batch, validation_data=valid_batch,
                    epochs=15, steps_per_epoch=SPE, validation_steps=VS)

# Saving the trained model to disk
model.save('models/cnnCat2.h5', overwrite=True)
