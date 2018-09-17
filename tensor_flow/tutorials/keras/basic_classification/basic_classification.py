#!/bin/env python3

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# The images are 28x28 NumPy arrays, with pixel values ranging between 0 and 255.
# The labels are an array of integers, ranging from 0 to 9.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Let's explore the format of the dataset before training the model.
# The following shows there are 60,000 images in the training set, with each
# image represented as 28 x 28 pixels:
print("Input training images shape: " + str(train_images.shape))
print("Output training labels   nb: " + str(len(train_labels)))

# test: 10000 examples
print("Input testing images shape: " + str(test_images.shape))
print("Output testing labels   nb: " + str(len(test_labels)))

# all input in [0:1]
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
