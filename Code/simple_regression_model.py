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
learning_rate = 0.1
training_epochs = 500
display_step = 50

elements = ["TEMP", "DEWP", "PRES", "RAIN", "wd", "WSPM"]

def single_regression_process(data1, data2, attribute):
    # Training Data
    train_X = data1.loc[:, attribute]
    train_Y = data1.loc[:, "PM2.5"]
    size = train_X.shape[0]

    # tf Graph Input
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)



    W1 = tf.Variable(tf.random_normal([1]), name="weight_1")
    W2 = tf.Variable(tf.random_normal([1]), name="weight_2")
    W3 = tf.Variable(tf.random_normal([1]), name="weight_3")
    b = tf.Variable(tf.random_normal([1]), name="d")
    # Cubic equation
    pred_deg3 = tf.multiply(W3, tf.pow(X, 3)) + tf.multiply(W2, tf.pow(X, 2)) + tf.multiply(W1, X) + b
    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred_deg3 - Y, 2)) / (2 * size)

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
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                      "W1=", sess.run(W1),"W2=", sess.run(W2),"W3=", sess.run(W3), \
                      "b=", sess.run(b))

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Training cost=", training_cost, "W1=", sess.run(W1),"W2=", sess.run(W2),"W3=", sess.run(W3), \
                       "b=", sess.run(b), '\n')

        # Graphic display
        plt.plot(train_X, train_Y, 'ro', label='Original data')
        pred = sess.run(pred_deg3, feed_dict={X: train_X})
        plt.plot(train_X, pred, label='Fitted line')
        plt.legend()
        plt.show()

        test_X = data2.loc[:, attribute]
        test_Y = data2.loc[:, "PM2.5"]

        print("Testing... (Mean square loss Comparison)")
        testing_cost = sess.run(
            tf.reduce_sum(tf.pow(pred_deg3 - Y, 2)) / (2 * test_X.shape[0]),
            feed_dict={X: test_X, Y: test_Y})  # same function as cost above
        print("Testing cost=", testing_cost)
        print("Absolute mean square loss difference:", abs(
            training_cost - testing_cost))

        plt.plot(test_X, test_Y, 'bo', label='Testing data')
        plt.plot(train_X, pred, label='Fitted line')
        plt.legend()
        plt.show()

if __name__ == '__main__':

    data1, data2 = Code.preprocess.readData()

    # Normalizing full dataframe
    data_d1 = data1.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data_d1)
    data1 = pd.DataFrame(x_scaled, columns=data1.columns)

    data_d2 = data2.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    y_scaled = min_max_scaler.fit_transform(data_d2)
    data2 = pd.DataFrame(y_scaled, columns=data2.columns)

    for attribute in elements:
        print("start testing: "+attribute)
        single_regression_process(data1, data2, attribute)