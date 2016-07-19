import tensorflow as tf

def CNN(x, depth, output_size, keep_prob):
  with tf.variable_scope('conv1') as scope:
    W_conv1 = tf.get_variable('Weight', [5, 5, depth, 32],
        initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.get_variable('Bias', [32],
        initializer=tf.constant_initializer(0.1))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1],
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

  with tf.variable_scope('dropout') as scope:
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.variable_scope('fc2') as scope:
    W_fc2 = tf.get_variable('Weight', [1024, output_size],
        initializer=tf.contrib.layers.xavier_initializer())
    b_fc2 = tf.get_variable('Bias', [output_size],
        initializer=tf.constant_initializer(0.1))
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  saver = tf.train.Saver({"W_conv1": W_conv1, "b_conv1": b_conv1,
                          "W_conv2": W_conv2, "b_conv2": b_conv2,
                          "W_fc1": W_fc1, "b_fc1": b_fc1,
                          "W_fc2": W_fc2, "b_fc2": b_fc2})

  return logits, saver
