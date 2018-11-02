#!/bin/env python3

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Rather than using an embedding as in the previous notebook, here we will
# multi-hot encode the sentences. This model will quickly overfit to the
# training set. It will be used to demonstrate when overfitting occurs, and how
# to fight it.

# Multi-hot-encoding our lists means turning them into vectors of 0s and 1s.
# Concretely, this would mean for instance turning the sequence [3, 5] into a
# 10,000-dimensional vector that would be all-zeros except for indices 3 and 5,
# which would be ones.

# -------------------------------------------------------------
# dataset
# -------------------------------------------------------------
NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = \
    keras.datasets.imdb.load_data(num_words=NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# plt.plot(train_data[0])
# plt.show()

# -------------------------------------------------------------
# demonstrate overfitting
# -------------------------------------------------------------
# The simplest way to prevent overfitting is to reduce the size of the model,
# i.e. the number of learnable parameters in the model (which is determined by
# the number of layers and the number of units per layer). In deep learning,
# the number of learnable parameters in a model is often referred to as the
# model's "capacity". Intuitively, a model with more parameters will have more
# "memorization capacity" and therefore will be able to easily learn a perfect
# dictionary-like mapping between training samples and their targets, a mapping
# without any generalization power, but this would be useless when making
# predictions on previously unseen data.

# Always keep this in mind: deep learning models tend to be good at fitting to
# the training data, but the real challenge is generalization, not fitting.

# On the other hand, if the network has limited memorization resources, it will
# not be able to learn the mapping as easily. To minimize its loss, it will
# have to learn compressed representations that have more predictive power. At
# the same time, if you make your model too small, it will have difficulty
# fitting to the training data. There is a balance between "too much capacity"
# and "not enough capacity".

# Unfortunately, there is no magical formula to determine the right size or
# architecture of your model (in terms of the number of layers, or what the
# right size for each layer). You will have to experiment using a series of
# different architectures.

# To find an appropriate model size, it's best to start with relatively few
# layers and parameters, then begin increasing the size of the layers or adding
# new layers until you see diminishing returns on the validation loss. Let's
# try this on our movie review classification network.

# We'll create a simple model using only Dense layers as a baseline, then
# create smaller and larger versions, and compare them.

# ------------------------------
# create a baseline model
# ------------------------------
baseline_model = keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()

baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)

# ------------------------------
# create a smaller model
# ------------------------------
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()

smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)

# ------------------------------
# create a bigger model
# ------------------------------
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'binary_crossentropy'])

bigger_model.summary()

bigger_history = bigger_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

# ------------------------------
# plot the training and validation loss
# ------------------------------
# The solid lines show the training loss, and the dashed lines show the
# validation loss (remember: a lower validation loss indicates a better model).
# Here, the smaller network begins overfitting later than the baseline model
# (after 6 epochs rather than 4) and its performance degrades much more slowly
# once it starts overfitting.


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])


plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])

plt.show()

# Notice that the larger network begins overfitting almost right away, after
# just one epoch, and overfits much more severely. The more capacity the
# network has, the quicker it will be able to model the training data
# (resulting in a low training loss), but the more susceptible it is to
# overfitting (resulting in a large difference between the training and
# validation loss).
