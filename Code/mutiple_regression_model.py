import os
import sys
sys.path.append(os.getcwd())

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import Code.preprocess
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd

rng = np.random
# Parameters
learning_rate = 0.05
training_epochs = 1000
display_step = 50

data1, data2 = Code.preprocess.readData()

# Normalizing full dataframe
data_d1 = data1.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(data_d1)
data1 = pd.DataFrame(x_scaled, columns=data1.columns)

data_d2 = data2.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(data_d2)
data2 = pd.DataFrame(y_scaled, columns=data2.columns)


# Training Data
train_X = data1.loc[:, ["TEMP", "DEWP", "PRES", "RAIN", "wd", "WSPM"]]
train_Y = data1.loc[:, "PM2.5"]

size = train_X.shape[0]
n = train_X.shape[1]

# tf Graph Input
X = tf.placeholder("float", [None, n])
Y = tf.placeholder("float", [None])

# Set model weights
# W = tf.Variable(tf.constant(rng.randn(), shape=[n, 1]), name="weight")
b = tf.Variable(rng.randn(), [n], name="bias")
W = tf.cast(tf.Variable(np.random.randn(n, 1), name="weight"), tf.float32)

# Construct a linear model

pred = tf.add(tf.matmul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*size)

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        index = 0
        for (x, y) in zip(np.asarray(train_X), np.asarray(train_Y)):
            xformatted = np.array([x])
            yformatted = np.array([y])
            sess.run(optimizer, feed_dict={X: xformatted, Y: yformatted})
        # sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    # plt.plot(train_X, train_Y, 'ro', label='Original data')
    # plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()

    test_X = data2.loc[:, ["TEMP", "DEWP", "PRES", "RAIN", "wd", "WSPM"]]
    test_Y = data2.loc[:, "PM2.5"]

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    # plt.plot(test_X, test_Y, 'bo', label='Testing data')
    # plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()