import numpy as np
import tensorflow as tf

"""
  Input:
    x: sum(time) x height x window_size x 1
    dropout: dropout for training = 1, dropour for testing = 0
             the actually dropout is calculated based on this multiplier
    height: 32 by default (you might need to change the architecture if you change the dimension)
    width: 32 by default (you might need to change the architecture if you change the dimension)
  Output:
    logits: image features of size 128, fc6 output
    variables: variable for CNN, return so that we can set the learning rate
    saver: saver for CNN, return so that we can save CNN variables separately
"""
def CNN(x, dropout, height, width):

  eps = 1e-5

  # input: 32 x 32 x 1 (default)
  with tf.variable_scope('conv1') as scope:
    W_conv1 = tf.get_variable('Weight', [3, 3, 1, 64], initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.get_variable('Bias', [64], initializer=tf.constant_initializer(0))
    z_conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1
    mean_conv1, variance_conv1 = tf.nn.moments(z_conv1, [0, 1, 2], keep_dims=False)
    gamma_conv1 = tf.get_variable('Gamma', [64], initializer=tf.constant_initializer(1))
    beta_conv1 = tf.get_variable('Beta', [64], initializer=tf.constant_initializer(0))
    bn_conv1 = tf.nn.batch_normalization(z_conv1, mean_conv1, variance_conv1, beta_conv1, gamma_conv1, eps)
    a_conv1 = tf.nn.relu(bn_conv1)
    h_pool1 = tf.nn.max_pool(a_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_pool1_drop = tf.nn.dropout(h_pool1, 1-dropout*0.1)

  # input: 16 x 16 x 64 (default)
  with tf.variable_scope('conv2') as scope:
    W_conv2 = tf.get_variable('Weight', [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    b_conv2 = tf.get_variable('Bias', [128], initializer=tf.constant_initializer(0))
    z_conv2 = tf.nn.conv2d(h_pool1_drop, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+b_conv2
    mean_conv2, variance_conv2 = tf.nn.moments(z_conv2, [0, 1, 2], keep_dims=False)
    gamma_conv2 = tf.get_variable('Gamma', [128], initializer=tf.constant_initializer(1))
    beta_conv2 = tf.get_variable('Beta', [128], initializer=tf.constant_initializer(0))
    bn_conv2 = tf.nn.batch_normalization(z_conv2, mean_conv2, variance_conv2, beta_conv2, gamma_conv2, eps)
    a_conv2 = tf.nn.relu(bn_conv2)
    h_pool2 = tf.nn.max_pool(a_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_pool2_drop = tf.nn.dropout(h_pool2, 1-dropout*0.1)

  # input: 8 x 8 x 128 (default)
  with tf.variable_scope('conv3') as scope:
    W_conv3 = tf.get_variable('Weight', [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    b_conv3 = tf.get_variable('Bias', [256], initializer=tf.constant_initializer(0))
    z_conv3 = tf.nn.conv2d(h_pool2_drop, W_conv3, strides=[1, 1, 1, 1], padding='SAME')+b_conv3
    mean_conv3, variance_conv3 = tf.nn.moments(z_conv3, [0, 1, 2], keep_dims=False)
    gamma_conv3 = tf.get_variable('Gamma', [256], initializer=tf.constant_initializer(1))
    beta_conv3 = tf.get_variable('Beta', [256], initializer=tf.constant_initializer(0))
    bn_conv3 = tf.nn.batch_normalization(z_conv3, mean_conv3, variance_conv3, beta_conv3, gamma_conv3, eps)
    a_conv3 = tf.nn.relu(bn_conv3)
    h_pool3 = tf.nn.max_pool(a_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_pool3_drop = tf.nn.dropout(h_pool3, 1-dropout*0.2)

  # input: 4 x 4 x 256 (default)
  with tf.variable_scope('conv4') as scope:
    W_conv4 = tf.get_variable('Weight', [3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
    b_conv4 = tf.get_variable('Bias', [512], initializer=tf.constant_initializer(0))
    z_conv4 = tf.nn.conv2d(h_pool3_drop, W_conv4, strides=[1, 1, 1, 1], padding='SAME')+b_conv4
    mean_conv4, variance_conv4 = tf.nn.moments(z_conv4, [0, 1, 2], keep_dims=False)
    gamma_conv4 = tf.get_variable('Gamma', [512], initializer=tf.constant_initializer(1))
    beta_conv4 = tf.get_variable('Beta', [512], initializer=tf.constant_initializer(0))
    bn_conv4 = tf.nn.batch_normalization(z_conv4, mean_conv4, variance_conv4, beta_conv4, gamma_conv4, eps)
    a_conv4 = tf.nn.relu(bn_conv4)
    h_pool4 = tf.nn.max_pool(a_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_pool4_drop = tf.nn.dropout(h_pool4, 1-dropout*0.2)
    h_pool4_drop_flat = tf.reshape(h_pool4_drop, [-1, height/16*width/16*512])

  # input: 2 x 2 x 512 (default)
  with tf.variable_scope('fc5') as scope:
    W_fc5 = tf.get_variable('Weight', [height/16*width/16*512, 512], initializer=tf.contrib.layers.xavier_initializer())
    b_fc5 = tf.get_variable('Bias', [512], initializer=tf.constant_initializer(0))
    a_fc5 = tf.nn.relu(tf.matmul(h_pool4_drop_flat, W_fc5)+b_fc5)
    a_fc5_drop = tf.nn.dropout(a_fc5, 1-dropout*0.5)

  # input: 1 x 1 x 512 (default)
  with tf.variable_scope('fc6') as scope:
    W_fc6 = tf.get_variable('Weight', [512, 128], initializer=tf.contrib.layers.xavier_initializer())
    b_fc6 = tf.get_variable('Bias', [128], initializer=tf.constant_initializer(0))
    logits = tf.matmul(a_fc5_drop, W_fc6)+b_fc6

  # outputs
  variables_CNN = [W_conv1, b_conv1, gamma_conv1, beta_conv1,
                   W_conv2, b_conv2, gamma_conv2, beta_conv2,
                   W_conv3, b_conv3, gamma_conv3, beta_conv3,
                   W_conv4, b_conv4, gamma_conv4, beta_conv4,
                   W_fc5, b_fc5, W_fc6, b_fc6]

  saver_CNN = tf.train.Saver({'W_conv1': W_conv1, 'b_conv1': b_conv1,
                              'gamma_conv1': gamma_conv1, 'beta_conv1': beta_conv1,
                              'W_conv2': W_conv2, 'b_conv2': b_conv2,
                              'gamma_conv2': gamma_conv2, 'beta_conv2': beta_conv2,
                              'W_conv3': W_conv3, 'b_conv3': b_conv3,
                              'gamma_conv3': gamma_conv3, 'beta_conv3': beta_conv3,
                              'W_conv4': W_conv4, 'b_conv4': b_conv4,
                              'gamma_conv4': gamma_conv4, 'beta_conv4': beta_conv4,
                              'W_fc5': W_fc5, 'b_fc5': b_fc5,
                              'W_fc6': W_fc6, 'b_fc6': b_fc6,
                            })

  return logits, variables_CNN, saver_CNN
