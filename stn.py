import numpy as np
from spatial_transformer import transformer
import tensorflow as tf


def STN(x, dropout, height, width):

  eps = 1e-5

  # input: 32 x 32 x 1
  with tf.variable_scope('loc-conv1') as scope:
    W_loc_conv1 = tf.get_variable('Weight', [3, 3, 1, 64], initializer=tf.contrib.layers.xavier_initializer())
    b_loc_conv1 = tf.get_variable('Bias', [64], initializer=tf.constant_initializer(0))
    z_loc_conv1 = tf.nn.conv2d(x, W_loc_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_loc_conv1
    mean_loc_conv1, variance_loc_conv1 = tf.nn.moments(z_loc_conv1, [0, 1, 2], keep_dims=False)
    gamma_loc_conv1 = tf.get_variable('Gamma', [64], initializer=tf.constant_initializer(1))
    beta_loc_conv1 = tf.get_variable('Beta', [64], initializer=tf.constant_initializer(0))
    bn_loc_conv1 = tf.nn.batch_normalization(z_loc_conv1, mean_loc_conv1, variance_loc_conv1, beta_loc_conv1, gamma_loc_conv1, eps)
    a_loc_conv1 = tf.nn.relu(bn_loc_conv1)
    h_loc_pool1 = tf.nn.max_pool(a_loc_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_loc_pool1_drop = tf.nn.dropout(h_loc_pool1, 1-dropout*0.1)

  # input: 16 x 16 x 64
  with tf.variable_scope('loc-conv2') as scope:
    W_loc_conv2 = tf.get_variable('Weight', [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    b_loc_conv2 = tf.get_variable('Bias', [128], initializer=tf.constant_initializer(0))
    z_loc_conv2 = tf.nn.conv2d(h_loc_pool1_drop, W_loc_conv2, strides=[1, 1, 1, 1], padding='SAME')+b_loc_conv2
    mean_loc_conv2, variance_loc_conv2 = tf.nn.moments(z_loc_conv2, [0, 1, 2], keep_dims=False)
    gamma_loc_conv2 = tf.get_variable('Gamma', [128], initializer=tf.constant_initializer(1))
    beta_loc_conv2 = tf.get_variable('Beta', [128], initializer=tf.constant_initializer(0))
    bn_loc_conv2 = tf.nn.batch_normalization(z_loc_conv2, mean_loc_conv2, variance_loc_conv2, beta_loc_conv2, gamma_loc_conv2, eps)
    a_loc_conv2 = tf.nn.relu(bn_loc_conv2)
    h_loc_pool2 = tf.nn.max_pool(a_loc_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_loc_pool2_drop = tf.nn.dropout(h_loc_pool2, 1-dropout*0.1)

  # input: 8 x 8 x 128
  with tf.variable_scope('loc-conv3') as scope:
    W_loc_conv3 = tf.get_variable('Weight', [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    b_loc_conv3 = tf.get_variable('Bias', [256], initializer=tf.constant_initializer(0))
    z_loc_conv3 = tf.nn.conv2d(h_loc_pool2_drop, W_loc_conv3, strides=[1, 1, 1, 1], padding='SAME')+b_loc_conv3
    mean_loc_conv3, variance_loc_conv3 = tf.nn.moments(z_loc_conv3, [0, 1, 2], keep_dims=False)
    gamma_loc_conv3 = tf.get_variable('Gamma', [256], initializer=tf.constant_initializer(1))
    beta_loc_conv3 = tf.get_variable('Beta', [256], initializer=tf.constant_initializer(0))
    bn_loc_conv3 = tf.nn.batch_normalization(z_loc_conv3, mean_loc_conv3, variance_loc_conv3, beta_loc_conv3, gamma_loc_conv3, eps)
    a_loc_conv3 = tf.nn.relu(bn_loc_conv3)
    h_loc_pool3 = tf.nn.max_pool(a_loc_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_loc_pool3_drop = tf.nn.dropout(h_loc_pool3, 1-dropout*0.2)
    h_loc_pool3_drop_flat = tf.reshape(h_loc_pool3_drop, [-1, 4*4*256])

  # input: 4 x 4 x 256
  with tf.variable_scope('loc-fc4') as scope:
    W_loc_fc4 = tf.get_variable('Weight', [4*4*256, 512], initializer=tf.constant_initializer(0))
    b_loc_fc4 = tf.get_variable('Bias', [512], initializer=tf.constant_initializer(0))
    h_loc_fc4 = tf.nn.relu(tf.matmul(h_loc_pool3_drop_flat, W_loc_fc4)+b_loc_fc4)
    h_loc_fc4_drop = tf.nn.dropout(h_loc_fc4, 1-dropout*0.5)

  # input: 1 x 1 x 512
  with tf.variable_scope('loc-fc5') as scope:
    W_loc_fc5 = tf.get_variable('Weight', [512, 64], initializer=tf.constant_initializer(0))
    b_loc_fc5 = tf.get_variable('Bias', [64], initializer=tf.constant_initializer(0))
    h_loc_fc5 = tf.nn.relu(tf.matmul(h_loc_fc4_drop, W_loc_fc5)+b_loc_fc5)
    h_loc_fc5_drop = tf.nn.dropout(h_loc_fc5, 1-dropout*0.5)

  # input: 1 x 1 x 64
  with tf.variable_scope('loc-fc6') as scope:
    W_loc_fc6 = tf.get_variable('Weight', [64, 6], initializer=tf.constant_initializer(0))
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    b_loc_fc6 = tf.get_variable('Bias', [6], initializer=tf.constant_initializer(initial))
    h_loc_fc6 = tf.matmul(h_loc_fc5_drop, W_loc_fc6)+b_loc_fc6

  with tf.variable_scope('transformer') as scope:
    x_trans = transformer(x, h_loc_fc6, (height, width))

  # outputs
  variables_STN = [W_loc_conv1, b_loc_conv1, gamma_loc_conv1, beta_loc_conv1,
                   W_loc_conv2, b_loc_conv2, gamma_loc_conv2, beta_loc_conv2,
                   W_loc_conv3, b_loc_conv3, gamma_loc_conv3, beta_loc_conv3,
                   W_loc_fc4, b_loc_fc4, W_loc_fc5, b_loc_fc5, W_loc_fc6, b_loc_fc6]

  saver_STN = tf.train.Saver({'W_loc_conv1': W_loc_conv1, 'b_loc_conv1': b_loc_conv1,
                              'gamma_loc_conv1': gamma_loc_conv1, 'beta_loc_conv1': beta_loc_conv1,
                              'W_loc_conv2': W_loc_conv2, 'b_conv2': b_loc_conv2,
                              'gamma_loc_conv2': gamma_loc_conv2, 'beta_loc_conv2': beta_loc_conv2,
                              'W_loc_conv3': W_loc_conv3, 'b_loc_conv3': b_loc_conv3,
                              'gamma_loc_conv3': gamma_loc_conv3, 'beta_loc_conv3': beta_loc_conv3,
                              'W_loc_fc4': W_loc_fc4, 'b_loc_fc4': b_loc_fc4,
                              'W_loc_fc5': W_loc_fc5, 'b_loc_fc5': b_loc_fc5,
                              'W_loc_fc6': W_loc_fc6, 'b_loc_fc6': b_loc_fc6,
                            })

  return x_trans, variables_STN, saver_STN
