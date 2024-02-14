#read input data sets
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import os
import sys
import ssl

import mltracker
import tensorflow as tf

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
if hasattr(ssl, '_create_unverified_context'):
      ssl._create_default_https_context = ssl._create_unverified_context
mnist = read_data_sets('./mnist', one_hot=True)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Call saved model for testing
save_filename = 'mnist_model.ckpt'
save_path = ('/gpfs-volume/{}'.format(save_filename))
saver = tf.train.Saver()

# For mltracker start
mltracker.start_run()
mltracker.log_param("img_shape", n_input)

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_path)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))

# For mltracker metric  information
mltracker.log_metric("Test_Accuracy", test_accuracy)

# For mltracker end
mltracker.end_run()
print('End of Testing')
