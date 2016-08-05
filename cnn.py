import tensorflow as tf
import numpy as np
from spatial_transformer import transformer


def CNN(x, height, width, depth, keep_prob, keep_prob_transformer):
  # x: batch_size x height x width x depth

  eps = 1e-5

  # localisation network
  with tf.variable_scope('loc-conv1') as scope:
    W_loc_conv1 = tf.get_variable('Weight', [5, 5, depth, 20],
        initializer=tf.contrib.layers.xavier_initializer())
    b_loc_conv1 = tf.get_variable('Bias', [20],
        initializer=tf.constant_initializer(0))
    z_loc_conv1 = tf.nn.conv2d(x, W_loc_conv1, strides=[1, 1, 1, 1],
        padding='SAME') + b_loc_conv1
    mean_loc_conv1, variance_loc_conv1 = tf.nn.moments(z_loc_conv1, [0, 1, 2], keep_dims=False)
    gamma_loc_conv1 = tf.get_variable('Gamma', [20], initializer=tf.constant_initializer(1))
    beta_loc_conv1 = tf.get_variable('Beta', [20], initializer=tf.constant_initializer(0))
    bn_loc_conv1 = tf.nn.batch_normalization(z_loc_conv1, mean_loc_conv1, variance_loc_conv1, beta_loc_conv1, gamma_loc_conv1, eps)
    a_loc_conv1 = tf.nn.relu(bn_loc_conv1)
    h_loc_pool1 = tf.nn.max_pool(a_loc_conv1, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

  with tf.variable_scope('loc-conv2') as scope:
    W_loc_conv2 = tf.get_variable('Weight', [5, 5, 20, 20],
        initializer=tf.contrib.layers.xavier_initializer())
    b_loc_conv2 = tf.get_variable('Bias', [20],
        initializer=tf.constant_initializer(0))
    z_loc_conv2 = tf.nn.conv2d(h_loc_pool1, W_loc_conv2, strides=[1, 1, 1, 1],
        padding='SAME') + b_loc_conv2
    mean_loc_conv2, variance_loc_conv2 = tf.nn.moments(z_loc_conv2, [0, 1, 2], keep_dims=False)
    gamma_loc_conv2 = tf.get_variable('Gamma', [20], initializer=tf.constant_initializer(1))
    beta_loc_conv2 = tf.get_variable('Beta', [20], initializer=tf.constant_initializer(0))
    bn_loc_conv2 = tf.nn.batch_normalization(z_loc_conv2, mean_loc_conv2, variance_loc_conv2, beta_loc_conv2, gamma_loc_conv2, eps)
    a_loc_conv2 = tf.nn.relu(bn_loc_conv2)
    h_loc_pool2 = tf.nn.max_pool(a_loc_conv2, ksize=[1, 2, 2, 1],
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
    b_loc_fc2 = tf.get_variable('Bias', [6],
        initializer=tf.constant_initializer(initial))
    h_loc_fc2 = tf.matmul(h_loc_fc1_drop, W_loc_fc2) + b_loc_fc2

  with tf.variable_scope('transformer') as scope:
    x_trans = transformer(x, h_loc_fc2, (height, width))

  # CNN for char recognition
  with tf.variable_scope('conv1') as scope:
    W_conv1 = tf.get_variable('Weight', [5, 5, depth, 32],
        initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.get_variable('Bias', [32],
        initializer=tf.constant_initializer(0))
    z_conv1 = tf.nn.conv2d(x_trans, W_conv1, strides=[1, 1, 1, 1],
        padding='SAME') + b_conv1
    mean_conv1, variance_conv1 = tf.nn.moments(z_conv1, [0, 1, 2], keep_dims=False)
    gamma_conv1 = tf.get_variable('Gamma', [32], initializer=tf.constant_initializer(1))
    beta_conv1 = tf.get_variable('Beta', [32], initializer=tf.constant_initializer(0))
    bn_conv1 = tf.nn.batch_normalization(z_conv1, mean_conv1, variance_conv1, beta_conv1, gamma_conv1, eps)
    a_conv1 = tf.nn.relu(bn_conv1)
    h_pool1 = tf.nn.max_pool(a_conv1, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

  with tf.variable_scope('conv2') as scope:
    W_conv2 = tf.get_variable('Weight', [5, 5, 32, 32],
        initializer=tf.contrib.layers.xavier_initializer())
    b_conv2 = tf.get_variable('Bias', [32],
        initializer=tf.constant_initializer(0))
    z_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
        padding='SAME') + b_conv2
    mean_conv2, variance_conv2 = tf.nn.moments(z_conv2, [0, 1, 2], keep_dims=False)
    gamma_conv2 = tf.get_variable('Gamma', [32], initializer=tf.constant_initializer(1))
    beta_conv2 = tf.get_variable('Beta', [32], initializer=tf.constant_initializer(0))
    bn_conv2 = tf.nn.batch_normalization(z_conv2, mean_conv2, variance_conv2, beta_conv2, gamma_conv2, eps)
    a_conv2 = tf.nn.relu(bn_conv2)
    h_pool2 = tf.nn.max_pool(a_conv2, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

  with tf.variable_scope('conv3') as scope:
    W_conv3 = tf.get_variable('Weight', [5, 5, 32, 64],
        initializer=tf.contrib.layers.xavier_initializer())
    b_conv3 = tf.get_variable('Bias', [64],
        initializer=tf.constant_initializer(0))
    z_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1],
        padding='SAME') + b_conv3
    mean_conv3, variance_conv3 = tf.nn.moments(z_conv3, [0, 1, 2], keep_dims=False)
    gamma_conv3 = tf.get_variable('Gamma', [64], initializer=tf.constant_initializer(1))
    beta_conv3 = tf.get_variable('Beta', [64], initializer=tf.constant_initializer(0))
    bn_conv3 = tf.nn.batch_normalization(z_conv3, mean_conv3, variance_conv3, beta_conv3, gamma_conv3, eps)
    a_conv3 = tf.nn.relu(bn_conv3)
    h_pool3 = tf.nn.max_pool(a_conv3, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

  with tf.variable_scope('fc1') as scope:
    W_fc1 = tf.get_variable('Weight', [height*width*64/64, 256],
        initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.get_variable('Bias', [256],
        initializer=tf.constant_initializer(0))
    h_pool3_flat = tf.reshape(h_pool3, [-1, height*width*64/64])
    logits = tf.matmul(h_pool3_flat, W_fc1) + b_fc1

  variables_STN = [W_loc_conv1, b_loc_conv1, gamma_loc_conv1,
      beta_loc_conv1, W_loc_conv2, b_loc_conv2, gamma_loc_conv2, beta_loc_conv2,
      W_loc_fc1, b_loc_fc1, W_loc_fc2, b_loc_fc2]
  variables_CNN = [W_conv1, b_conv1, gamma_conv1, beta_conv1, W_conv2, b_conv2,
      gamma_conv2, beta_conv2, W_conv3, b_conv3, gamma_conv3, beta_conv3, W_fc1,
      b_fc1]

  saver_STN = tf.train.Saver({'W_loc_conv1': W_loc_conv1, 'b_loc_conv1': b_loc_conv1,
                              'gamma_loc_conv1': gamma_loc_conv1, 'beta_loc_conv1': beta_loc_conv1,
                              'W_loc_conv2': W_loc_conv2, 'b_loc_conv2': b_loc_conv2,
                              'gamma_loc_conv2': gamma_loc_conv2, 'beta_loc_conv2': beta_loc_conv2,
                              'W_loc_fc1': W_loc_fc1, 'b_loc_fc1': b_loc_fc1,
                              'W_loc_fc2': W_loc_fc2, 'b_loc_fc2': b_loc_fc2
                            })

  saver_CNN = tf.train.Saver({'W_conv1': W_conv1, 'b_conv1': b_conv1,
                              'gamma_conv1': gamma_conv1, 'beta_conv1': beta_conv1,
                              'W_conv2': W_conv2, 'b_conv2': b_conv2,
                              'gamma_conv2': gamma_conv2, 'beta_conv2': beta_conv2,
                              'W_conv3': W_conv3, 'b_conv3': b_conv3,
                              'gamma_conv3': gamma_conv3, 'beta_conv3': beta_conv3,
                              'W_fc1': W_fc1, 'b_fc1': b_fc1
                            })

  return logits, variables_STN, variables_CNN, saver_STN, saver_CNN, x_trans
