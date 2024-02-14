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

# Import MNIST data
if hasattr(ssl, '_create_unverified_context'):
      ssl._create_default_https_context = ssl._create_unverified_context
mnist = read_data_sets('./mnist', one_hot=True)

learning_rate = 0.5
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)
saver = tf.train.Saver()
# Define loss and optimizer
cost = tf.reduce_mean( \
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) \
    .minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

import math

# Save path and model name
save_filename = 'mnist_model.ckpt'
save_path = ('/gpfs-volume/{}'.format(save_filename))
log_path = ('/gpfs-volume/{}.data-00000-of-00001'.format(save_filename))

batch_size = 256
n_epochs = 30

n_epochs = os.environ.get('epoch')
if n_epochs is None:
  n_epochs = 30
n_epochs = int(n_epochs)
print('n_epochs : ' + str(n_epochs))

saver = tf.train.Saver()

# For Tracking model param and metric
mltracker.start_run()
# save param key, value
mltracker.log_param("batch_size", batch_size)
mltracker.log_param("learning_rate", learning_rate)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(
                optimizer,
                feed_dict={features: batch_features, labels: batch_labels})

        # Print status for every 10 epochs
        if epoch % 10 == 0:
            valid_accuracy = sess.run(
                accuracy,
                feed_dict={
                    features: mnist.validation.images,
                    labels: mnist.validation.labels})

            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))
            # save metric key, value
            mltracker.log_metric("Validation Accuracy", valid_accuracy)

    # Save the model
    saver.save(sess, save_path)
    # Save
    mltracker.log_file(log_path)
    mltracker.end_run()
    print('Trained Model Saved.')
