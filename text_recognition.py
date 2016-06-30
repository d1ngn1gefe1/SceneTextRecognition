import tensorflow as tf
import numpy as np
import utils
import cv2
import h5py
import math
import os
import time
import json

debug = True

class Config():
  def __init__(self):
    with open('config.json', 'r') as json_file:
      json_data = json.load(json_file)
      self.dataset_dir = json_data['dataset_dir']
      self.height = json_data['height']
      self.window_size = json_data['window_size']
      self.depth = json_data['depth']
      self.embed_size = json_data['embed_size']
      self.stride = json_data['stride']

  lr = 1e-2
  lstm_size = 64
  num_epochs = 20000
  batch_size = 10

  # for debugging
  debug_size = 40
  take = 8

class DTRN_Model():
  def __init__(self, config):
    self.config = config
    self.load_data(debug)
    self.add_placeholders()
    self.rnn_outputs = self.add_model()
    self.outputs = self.add_projection(self.rnn_outputs)
    self.loss = self.add_loss_op(self.outputs)
    self.train_op = self.add_training_op(self.loss)

  def load_data(self, debug=False):
    filename_train = os.path.join(self.config.dataset_dir, 'train.hdf5')
    filename_test = os.path.join(self.config.dataset_dir, 'test.hdf5')

    f_train = h5py.File(filename_train, 'r')
    f_test = h5py.File(filename_test, 'r')

    # load train data
    self.imgs_train = np.array(f_train.get('imgs'), dtype=np.uint8)
    self.words_embed_train = \
        np.array(f_train.get('words_embed'), dtype=np.uint8)
    self.time_train = np.array(f_train.get('time'), dtype=np.int32)
    self.words_length_train = \
        np.array(f_train.get('words_length'), dtype=np.int32)

    self.imgs_test = np.array(f_test.get('imgs'), dtype=np.uint8)
    self.words_embed_test = np.array(f_test.get('words_embed'), dtype=np.uint8)
    self.time_test = np.array(f_test.get('time'), dtype=np.int32)
    self.words_length_test = \
        np.array(f_test.get('words_length'), dtype=np.int32)

    if debug:
      self.imgs_train = self.imgs_train[:self.config.debug_size]
      self.words_embed_train = self.words_embed_train[:self.config.debug_size]
      self.time_train = self.time_train[:self.config.debug_size]
      self.words_length_train = self.words_length_train[:self.config.debug_size]

      self.imgs_test = self.imgs_test[:self.config.debug_size]
      self.words_embed_test = self.words_embed_test[:self.config.debug_size]
      self.time_test = self.time_test[:self.config.debug_size]
      self.words_length_test = self.words_length_test[:self.config.debug_size]

    self.max_time = max(np.amax(self.time_train), np.amax(self.time_test))
    self.max_words_length = max(np.amax(self.words_length_train), \
        np.amax(self.words_length_test))
    print 'max_time = ', self.max_time
    print 'max_words_length = ', self.max_words_length

    if debug:
      self.imgs_train = self.imgs_train[:, :self.max_time]
      self.words_embed_train = self.words_embed_train[:, :self.max_words_length]

      self.imgs_test = self.imgs_test[:, :self.max_time]
      self.words_embed_test = self.words_embed_test[:, :self.max_words_length]

    f_train.close()
    f_test.close()

  def add_placeholders(self):
    # max_time x batch_size x height x width x depth
    self.inputs_placeholder = tf.placeholder(tf.float32, \
        shape=[self.max_time, self.config.batch_size, self.config.height, \
        self.config.window_size, self.config.depth])

    # max_time x batch_size x embed_size (63)
    self.labels_placeholder = tf.sparse_placeholder(tf.int32)

    # batch_size
    self.sequence_length_placeholder = tf.placeholder(tf.int32, \
        shape=[self.config.batch_size])

  def CNN(self, images):
    # images: 4D tensor of size [batch_size, height, width, depth]

    with tf.variable_scope('conv1') as scope:
      kernel = utils.variable_with_weight_decay('weights', \
          shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = utils.variable_on_cpu('biases', [64], \
          tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(bias, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], \
        padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, \
        name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
      kernel = utils.variable_with_weight_decay('weights', \
          shape=[5, 5, 64, 64], stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = utils.variable_on_cpu('biases', [64], \
          tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(bias, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, \
        name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], \
        strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
      # Move everything into depth so we can perform a single matrix multiply.
      reshape = tf.reshape(pool2, \
          [self.max_time*self.config.batch_size, -1])
      dim = reshape.get_shape()[1].value
      weights = utils.variable_with_weight_decay('weights', shape=[dim, 384], \
          stddev=0.04, wd=0.004)
      biases = utils.variable_on_cpu('biases', [384], \
          tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
      weights = utils.variable_with_weight_decay('weights', shape=[384, 192], \
          stddev=0.04, wd=0.004)
      biases = utils.variable_on_cpu('biases', [192], \
          tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    return local4

  def add_model(self):
    self.cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_size, \
        state_is_tuple=True)

    with tf.variable_scope('LSTM') as scope:
      # inputs_placeholder: max_time x batch_size x height x window_size x depth
      # data_cnn: max_time*batch_size x height x window_size x depth
      data_cnn = tf.reshape(self.inputs_placeholder, \
          [self.max_time*self.config.batch_size, self.config.height, \
          self.config.window_size, self.config.depth])

      img_features = self.CNN(data_cnn)

      # img_features: max_time*batch_size x feature_size (192)
      # data_encoder: a length max_time list, each a tensor of shape
      #               [batch_size, feature_size]
      data_encoder = tf.reshape(img_features, (self.max_time, \
          self.config.batch_size, -1))
      data_encoder = tf.split(0, self.max_time, data_encoder)
      data_encoder = [tf.squeeze(datum) for datum in data_encoder]

      rnn_outputs, _, _ = tf.nn.bidirectional_rnn(self.cell, self.cell, \
          data_encoder, sequence_length=self.sequence_length_placeholder, \
          dtype=tf.float32)

      # rnn_outputs: a length max_time list, each a tensor of shape
      #              [batch_size, 2*lstm_size]
      return rnn_outputs

  def add_projection(self, rnn_outputs):
    with tf.variable_scope('Projection'):
      W = tf.get_variable('Weight', [2*self.config.lstm_size, \
          self.config.embed_size], \
          initializer=tf.random_normal_initializer(0.0, 1e-3))
      b = tf.get_variable('Bias', [self.config.embed_size],
          initializer=tf.constant_initializer(0.0))
      outputs = [tf.matmul(rnn_output, W)+b for rnn_output in rnn_outputs]
    return outputs

  def add_loss_op(self, outputs):
    loss = tf.contrib.ctc.ctc_loss(outputs, self.labels_placeholder, \
        self.sequence_length_placeholder)
    loss = tf.reduce_mean(loss)
    return loss

  def add_training_op(self, loss):
    optimizer = tf.train.AdamOptimizer(self.config.lr)
    train_op = optimizer.minimize(loss)
    return train_op

def main():
  config = Config()
  model = DTRN_Model(config)

  init = tf.initialize_all_variables()

  with tf.Session() as session:
    session.run(init)

    iterator = utils.data_iterator( \
        model.imgs_train, model.words_embed_train,
        model.time_train, model.words_length_train,
        model.config.num_epochs, model.config.batch_size, \
        model.max_time, model.max_words_length)

    for step, (inputs, labels_indices, labels_values, labels_shape, \
        sequence_length) in enumerate(iterator):

      feed = {model.inputs_placeholder: inputs,
              model.labels_placeholder: (labels_indices, labels_values, \
                  labels_shape),
              model.sequence_length_placeholder: sequence_length}

      ret = session.run([model.train_op, model.loss], feed_dict=feed)

      print 'loss: ', ret[1]

if __name__ == '__main__':
  main()
