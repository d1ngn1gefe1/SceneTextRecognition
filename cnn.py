import tensorflow as tf
import numpy as np
from spatial_transformer import transformer


def CNN(x, height, width, depth, output_size, keep_prob, keep_prob_transformer):
  # x: batch_size x height x width x depth

  # localisation network
  with tf.variable_scope('loc-conv1') as scope:
    W_loc_conv1 = tf.get_variable('Weight', [5, 5, depth, 20],
        initializer=tf.contrib.layers.xavier_initializer())
    b_loc_conv1 = tf.get_variable('Bias', [20],
        initializer=tf.constant_initializer(0))
    h_loc_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_loc_conv1, strides=[1, 1, 1, 1],
        padding='SAME') + b_loc_conv1)
    h_loc_pool1 = tf.nn.max_pool(h_loc_conv1, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

  with tf.variable_scope('loc-conv2') as scope:
    W_loc_conv2 = tf.get_variable('Weight', [5, 5, 20, 20],
        initializer=tf.contrib.layers.xavier_initializer())
    b_loc_conv2 = tf.get_variable('Bias', [20],
        initializer=tf.constant_initializer(0))
    h_loc_conv2 = tf.nn.relu(tf.nn.conv2d(h_loc_pool1, W_loc_conv2,
        strides=[1, 1, 1, 1], padding='SAME') + b_loc_conv2)
    h_loc_pool2 = tf.nn.max_pool(h_loc_conv2, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

  with tf.variable_scope('loc-fc1') as scope:
    W_loc_fc1 = tf.get_variable('Weight', [height*width*20/16, 20],
        initializer=tf.constant_initializer(0))
    b_loc_fc1 = tf.get_variable('Bias', [20],
        initializer=tf.constant_initializer(0))
    h_loc_pool2_flat = tf.reshape(h_loc_pool2, [-1, height*width*20/16])
    h_loc_fc1 = tf.nn.relu(tf.matmul(h_loc_pool2_flat, W_loc_fc1) + b_loc_fc1)
    h_loc_fc1_drop = tf.nn.dropout(h_loc_fc1, keep_prob_transformer)

  with tf.variable_scope('loc-fc2') as scope:
    W_loc_fc2 = tf.get_variable('Weight', [20, 6],
        initializer=tf.constant_initializer(0))
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    b_loc_fc2 = tf.get_variable('Bias', initializer=tf.constant(initial))
    h_loc_fc2 = tf.matmul(h_loc_fc1_drop, W_loc_fc2) + b_loc_fc2

  with tf.variable_scope('transformer') as scope:
    x_trans = transformer(x, h_loc_fc2, (height, width))

  # CNN for char recognition
  with tf.variable_scope('conv1') as scope:
    W_conv1 = tf.get_variable('Weight', [5, 5, depth, 32],
        initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.get_variable('Bias', [32],
        initializer=tf.constant_initializer(0))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_trans, W_conv1, strides=[1, 1, 1, 1],
        padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

  with tf.variable_scope('conv2') as scope:
    W_conv2 = tf.get_variable('Weight', [5, 5, 32, 32],
        initializer=tf.contrib.layers.xavier_initializer())
    b_conv2 = tf.get_variable('Bias', [32],
        initializer=tf.constant_initializer(0))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
        padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

  with tf.variable_scope('conv3') as scope:
    W_conv3 = tf.get_variable('Weight', [5, 5, 32, 64],
        initializer=tf.contrib.layers.xavier_initializer())
    b_conv3 = tf.get_variable('Bias', [64],
        initializer=tf.constant_initializer(0))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1],
        padding='SAME') + b_conv3)
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

  with tf.variable_scope('fc1') as scope:
    W_fc1 = tf.get_variable('Weight', [height*width*64/64, 256],
        initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.get_variable('Bias', [256],
        initializer=tf.constant_initializer(0))
    h_pool3_flat = tf.reshape(h_pool3, [-1, height*width*64/64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.variable_scope('fc2') as scope:
    W_fc2 = tf.get_variable('Weight', [256, output_size],
        initializer=tf.contrib.layers.xavier_initializer())
    b_fc2 = tf.get_variable('Bias', [output_size],
        initializer=tf.constant_initializer(0))
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  variables_spatial_transformer = [W_loc_conv1, b_loc_conv1, W_loc_conv2, b_loc_conv2, W_loc_fc1, b_loc_fc1, W_loc_fc2, b_loc_fc2]
  variables_CNN = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]

  return logits, variables_spatial_transformer, variables_CNN, x_trans
