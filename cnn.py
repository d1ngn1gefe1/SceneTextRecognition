import tensorflow as tf
import numpy as np
from spatial_transformer import transformer


def CNN(x, height, width, depth, output_size, keep_prob, keep_prob_transformer):
  # x: batch_size x height x width x depth

  x_reshape = tf.reshape(x, [-1, height*width*depth])

  # two-layer localisation network
  with tf.variable_scope('loc1') as scope:
    W_fc_loc1 = tf.get_variable('Weight', [height*width*depth, 20],
        initializer=tf.constant_initializer(0))
    b_fc_loc1 = tf.get_variable('Bias', [20],
        initializer=tf.constant_initializer(0))
    h_fc_loc1 = tf.nn.tanh(tf.matmul(x_reshape, W_fc_loc1) + b_fc_loc1)
    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob_transformer)
  with tf.variable_scope('loc2') as scope:
    W_fc_loc2 = tf.get_variable('Weight', [20, 6],
        initializer=tf.constant_initializer(0))
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    b_fc_loc2 = tf.get_variable('Bias', initializer=tf.constant(initial))
    h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

  with tf.variable_scope('transformer') as scope:
    h_trans = transformer(x, h_fc_loc2, (height, width))

  with tf.variable_scope('conv1') as scope:
    W_conv1 = tf.get_variable('Weight', [5, 5, depth, 32],
        initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.get_variable('Bias', [32],
        initializer=tf.constant_initializer(0.1))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(h_trans, W_conv1, strides=[1, 1, 1, 1],
        padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

  with tf.variable_scope('conv2') as scope:
    W_conv2 = tf.get_variable('Weight', [5, 5, 32, 64],
        initializer=tf.contrib.layers.xavier_initializer())
    b_conv2 = tf.get_variable('Bias', [64],
        initializer=tf.constant_initializer(0.1))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
        padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

  with tf.variable_scope('fc1') as scope:
    W_fc1 = tf.get_variable('Weight', [7*7*64, 1024],
        initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.get_variable('Bias', [1024],
        initializer=tf.constant_initializer(0.1))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.variable_scope('fc2') as scope:
    W_fc2 = tf.get_variable('Weight', [1024, output_size],
        initializer=tf.contrib.layers.xavier_initializer())
    b_fc2 = tf.get_variable('Bias', [output_size],
        initializer=tf.constant_initializer(0.1))
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  variables_spatial_transformer = [W_fc_loc1, b_fc_loc1, W_fc_loc2, b_fc_loc2]
  variables_CNN = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]

  return logits, variables_spatial_transformer, variables_CNN, h_trans
