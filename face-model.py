# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Sequential

# Helper libraries
# import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
import os
import random
import pathlib

# Set the path of the input folder

dataset = "https://drive.google.com/uc?export=download&id=1IXx1TynW-WmUlNNbJEvnExXDrmDvQlT3"
directory = tf.keras.utils.get_file('caras', origin=dataset, untar=True)
data = pathlib.Path(directory)
folders = os.listdir(data)
#print(folders)

# Import the images and resize them to a 128*128 size
# Also generate the corresponding labels

image_names = []
train_labels = []
train_images = []

size = 64,64
print('folders')
#folders.remove("LICENSE.txt")
print(folders)

for folder in folders:
    for file in os.listdir(os.path.join(data,folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else:
            continue

# Transform the image array to a numpy type

train = np.array(train_images)
print(train.shape)

# Reduce the RGB values between 0 and 1
train = train.astype('float32') / 255.0
# Extract the labels
label_dummies = pandas.get_dummies(train_labels)
labels =  label_dummies.values.argmax(1)
pandas.unique(train_labels)
print(pandas.unique(labels))

# Shuffle the labels and images randomly for better results

union_list = list(zip(train, labels))
random.shuffle(union_list)
train,labels = zip(*union_list)

# Convert the shuffled list to numpy array type

train = np.array(train)
labels = np.array(labels)


# Develop a sequential model using tensorflow keras
#model = keras.Sequential([
    #keras.layers.Flatten(input_shape=(64,64,3)),
    #keras.layers.Dense(128, activation=tf.nn.tanh),
    #keras.layers.Dense(10, activation=tf.nn.softmax)
#])

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(train_labels), activation='softmax'))

# Compute the model parameters

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train,labels, epochs=5)

export_path = 'face-model/1/'
tf.saved_model.save(model, os.path.join('./',export_path))
