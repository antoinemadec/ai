#!/bin/env python3

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

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
# We can pad the arrays so they all have the same length, then
# create an integer tensor of shape max_length * num_reviews. We can use an
# embedding layer capable of handling this shape as the first layer in our
# network.

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
# The first layer is an Embedding layer. This layer takes the integer-encoded
# vocabulary and looks up the embedding vector for each word-index. These
# vectors are learned as the model trains. The vectors add a dimension to the
# output array. The resulting dimensions are: (batch, sequence, embedding).
model.add(keras.layers.Embedding(vocab_size, 16))
# Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for
# each example by averaging over the sequence dimension. This allows the model
# can handle input of variable length, in the simplest way possible. The
# resulting dimensions are: (batch, embedding).
model.add(keras.layers.GlobalAveragePooling1D())
# This fixed-length output vector is piped through a fully-connected (Dense)
# layer with 16 hidden units.
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# The last layer is densely connected with a single output node. Using the
# sigmoid activation function, this value is a float between 0 and 1,
# representing a probability, or confidence level.
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()
plot_model(model, to_file='model.png', show_shapes='True')

# Loss function and optimizer
# A model needs a loss function and an optimizer for training. Since this is a
# binary classification problem and the model outputs of a probability (a
# single-unit layer with a sigmoid activation), we'll use the
# binary_crossentropy loss function.

# This isn't the only choice for a loss function, you could, for instance,
# choose mean_squared_error. But, generally, binary_crossentropy is better for
# dealing with probabilities—it measures the "distance" between probability
# distributions, or in our case, between the ground-truth distribution and the
# predictions.

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --------------------------------------------------------------
# train the model
# --------------------------------------------------------------
# validation/train set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model for 40 epochs in mini-batches of 512 samples. This is 40
# iterations over all samples in the x_train and y_train tensors. While
# training, monitor the model's loss and accuracy on the 10,000 samples from
# the validation set:
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# --------------------------------------------------------------
# evaluate the model
# --------------------------------------------------------------
results = model.evaluate(test_data, test_labels)
print(results)

# --------------------------------------------------------------
# create a graph of accuracy and loss over time
# --------------------------------------------------------------
# In this plot, the dots represent the training loss and accuracy, and the
# solid lines are the validation loss and accuracy.

# Notice the training loss decreases with each epoch and the training accuracy
# increases with each epoch. This is expected when using a gradient descent
# optimization—it should minimize the desired quantity on every iteration.

# This isn't the case for the validation loss and accuracy—they seem to peak
# after about twenty epochs. This is an example of overfitting: the model
# performs better on the training data than it does on data it has never seen
# before. After this point, the model over-optimizes and learns representations
# specific to the training data that do not generalize to test data.

# For this particular case, we could prevent overfitting by simply stopping the
# training after twenty or so epochs. Later, you'll see how to do this
# automatically with a callback.
history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
