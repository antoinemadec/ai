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

# -------------------------------------------------------------
# build model
# -------------------------------------------------------------
# relu:     rectified linear
# softmax:  sigmoid for multiple classes
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Loss function —Error function
# Optimizer —This is how the model is updated based on the data it sees and its
# loss function.
# Metrics —Used to monitor the training and testing steps. The following
# example uses accuracy, the fraction of the images that are correctly
# classified.
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------------------------------------------
# train model
# -------------------------------------------------------------
# To start training, call the model.fit method—the model is "fit" to the
# training data:
model.fit(train_images, train_labels, epochs=5)

# -------------------------------------------------------------
# evaluate accuracy
# -------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# -------------------------------------------------------------
# predict
# -------------------------------------------------------------
predictions = model.predict(test_images)
# A prediction is an array of 10 numbers. These describe the "confidence" of
# the model that the image corresponds to each of the 10 different articles of
# clothing. We can see which label has the highest confidence value:
print(np.argmax(predictions[0]))

# plots
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# finally we can predict a single image

# tf.keras models are optimized to make predictions on a batch, or collection,
# of examples at once. So even though we're using a single image, we need to
img = (np.expand_dims(test_images[0],0))
predictions_single = model.predict(img)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# model.predict returns a list of lists, one for each image in the batch of
# data. Grab the predictions for our (only) image in the batch:
print(np.argmax(predictions_single[0]))
