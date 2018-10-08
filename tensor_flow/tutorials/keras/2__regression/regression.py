#!/bin/env python3

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)


# -------------------------------------------------------------
# dataset
# -------------------------------------------------------------
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features

# The dataset contains 13 different features:
#   Per capita crime rate.
#   The proportion of residential land zoned for lots over 25,000 square feet.
#   The proportion of non-retail business acres per town.
#   Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
#   Nitric oxides concentration (parts per 10 million).
#   The average number of rooms per dwelling.
#   The proportion of owner-occupied units built before 1940.
#   Weighted distances to five Boston employment centers.
#   Index of accessibility to radial highways.
#   Full-value property-tax rate per $10,000.
#   Pupil-teacher ratio by town.
#   1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
#   Percentage lower status of the population.

# use the pandas library to display the first few rows of the dataset in a nicely formatted table:
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(train_data, columns=column_names)
print(df.head())


# -------------------------------------------------------------
# normalize features
# -------------------------------------------------------------
# It's recommended to normalize features that use different scales and ranges.
# For each feature, subtract the mean of the feature and divide by the standard
# deviation:

# Test data is *not* used when calculating the mean and std

print("first training example, before normalization:\n%s" % train_data[0])

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print("first training example, after normalizatio:\n %s" % train_data[0])


# -------------------------------------------------------------
# create the model
# -------------------------------------------------------------
# Let's build our model. Here, we'll use a Sequential model with two densely
# connected hidden layers, and an output layer that returns a single,
# continuous value. The model building steps are wrapped in a function,
# build_model, since we'll create a second model, later on.


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model


model = build_model()
model.summary()


# -------------------------------------------------------------
# train the model
# -------------------------------------------------------------
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

# Visualize the model's training progress using the stats stored in the history
# object. We want to use this data to determine how long to train before the
# model stops making progress.


def plot_history(history, block=False):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show(block=block)
    plt.pause(0.001)
    if not block:
        input("\nPress [Enter] to continue")


plot_history(history)

# -------------------------------------------------------------
# re-train model, stop when no improvement on validation set
# -------------------------------------------------------------
# this graph shows little improvement in the model after about 200 epochs.
# let's update the model.fit method to automatically stop training when the
# validation score doesn't improve. we'll use a callback that tests a training
# condition for every epoch. if a set amount of epochs elapses without showing
# improvement, then automatically stop the training.

model = build_model()
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

# flatten() transform (102,1) -> (102,)
test_predictions = model.predict(test_data).flatten()
plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
print(test_labels.shape)
plt.figure()
plt.hist(error, bins=50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")

plt.show()


# -------------------------------------------------------------
# conclusion
# -------------------------------------------------------------
# This notebook introduced a few techniques to handle a regression problem.
#   - Mean Squared Error (MSE) is a common loss function used for regression
#       problems (different than classification problems).
#   - Similarly, evaluation metrics used for regression differ from
#       classification. A common regression metric is Mean Absolute Error (MAE).
#   - When input data features have values with different ranges, each feature
#       should be scaled independently.
#   - If there is not much training data, prefer a small network with few
#       hidden layers to avoid overfitting.
#   - Early stopping is a useful technique to prevent overfitting.
