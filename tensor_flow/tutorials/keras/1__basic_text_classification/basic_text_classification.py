#!/bin/env python3

import tensorflow as tf
from tensorflow import keras
# import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# --------------------------------------------------------------
# explore the data
# --------------------------------------------------------------
print("Training entries: %d, labels: %d" %
      (len(train_data), len(train_labels)))
print(type(train_data[0]))
# The text of reviews have been converted to integers, where each integer
# represents a specific word in a dictionary. Here's what the first review
# looks like:
print(train_data[0])

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# The first indices are reserved
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))

# --------------------------------------------------------------
# prepare the data
# --------------------------------------------------------------
# Alternatively, we can pad the arrays so they all have the same length, then
# create an integer tensor of shape max_length * num_reviews. We can use an
# embedding layer capable of handling this shape as the first layer in our
# network.

# In this tutorial, we will use the second approach.

# Since the movie reviews must be the same length, we will use the
# pad_sequences function to standardize the lengths:
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# Let's look at the length of the examples now:
len(train_data[0]), len(train_data[1])
# And inspect the (now padded) first review:
print(train_data[0])

# --------------------------------------------------------------
# build the model
# --------------------------------------------------------------
# The neural network is created by stacking layers—this requires two main
# architectural decisions:
# How many layers to use in the model?
# How many hidden units to use for each layer?

# In this example, the input data consists of an array of word-indices. The
# labels to predict are either 0 or 1. Let's build a model for this problem:

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()
