import tensorflow as tf
import numpy as np
import utils
import cv2
import h5py
import math
import os
import time
import json
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')

fh = logging.FileHandler('debug.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

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
      self.lr = json_data['lr']
      self.lstm_size = json_data['lstm_size']
      self.num_epochs = json_data['num_epochs']
      self.batch_size = json_data['batch_size']
      self.debug = json_data['debug']
      self.debug_size = json_data['debug_size']
      self.load_ckpt = json_data['load_ckpt']
      self.ckpt_dir = json_data['ckpt_dir']
      self.save_every_n_steps = json_data['save_every_n_steps']
      self.test_every_n_steps = json_data['test_every_n_steps']
      self.test_size = json_data['test_size']

class DTRN_Model():
  def __init__(self, config):
    self.config = config
    self.load_data(self.config.debug)
    self.add_placeholders()
    self.rnn_outputs = self.add_model()
    self.outputs = self.add_projection(self.rnn_outputs)
    self.loss = self.add_loss_op(self.outputs)
    self.pred = self.add_decoder(self.outputs)
    self.train_op = self.add_training_op(self.loss)

  def load_data(self, debug=False):
    filename_train = os.path.join(self.config.dataset_dir, 'train.hdf5')
    filename_test = os.path.join(self.config.dataset_dir, 'test.hdf5')

    # load train data
    f_train = h5py.File(filename_train, 'r')
    self.imgs_train = np.array(f_train.get('imgs'), dtype=np.uint8)
    self.words_embed_train = f_train.get('words_embed')[()].tolist()
    self.time_train = np.array(f_train.get('time'), dtype=np.uint8)
    logger.info('loading training data (%d examples)', self.imgs_train.shape[0])
    f_train.close()

    f_test = h5py.File(filename_test, 'r')
    self.imgs_test = np.array(f_test.get('imgs'), dtype=np.uint8)
    self.words_embed_test = f_test.get('words_embed')[()].tolist()
    self.time_test = np.array(f_test.get('time'), dtype=np.uint8)
    logger.info('loading test data (%d examples)', self.imgs_test.shape[0])
    f_test.close()

    if self.config.debug:
      self.imgs_train = self.imgs_train[:self.config.debug_size]
      self.words_embed_train = self.words_embed_train[:self.config.debug_size]
      self.time_train = self.time_train[:self.config.debug_size]

    if self.imgs_test.shape[0] > self.config.test_size:
      self.imgs_test = self.imgs_test[:self.config.test_size]
      self.words_embed_test = self.words_embed_test[:self.config.test_size]
      self.time_test = self.time_test[:self.config.test_size]

    self.max_time = max(np.amax(self.time_train), np.amax(self.time_test))
    self.imgs_train = self.imgs_train[:, :self.max_time]
    self.imgs_test = self.imgs_test[:, :self.max_time]


  def add_placeholders(self):
    # batch_size x max_time x height x width x depth
    self.inputs_placeholder = tf.placeholder(tf.float32,
        shape=[self.config.batch_size, self.max_time, self.config.height,
        self.config.window_size, self.config.depth])

    # batch_size x max_time x embed_size (63)
    self.labels_placeholder = tf.sparse_placeholder(tf.int32)

    # batch_size
    self.sequence_length_placeholder = tf.placeholder(tf.int32,
        shape=[self.config.batch_size])

  def CNN(self, images):
    # images: 4D tensor of size [batch_size, height, width, depth]

    with tf.variable_scope('conv1') as scope:
      kernel = utils.variable_with_weight_decay('weights',
          shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = tf.get_variable('biases', [64],
          initializer=tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(bias, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
        padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
      kernel = utils.variable_with_weight_decay('weights',
          shape=[5, 5, 64, 64], stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = tf.get_variable('biases', [64],
          initializer=tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(bias, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
      # Move everything into depth so we can perform a single matrix multiply.
      reshape = tf.reshape(pool2,
          [self.max_time*self.config.batch_size, -1])
      dim = reshape.get_shape()[1].value
      weights = utils.variable_with_weight_decay('weights', shape=[dim, 384],
          stddev=0.04, wd=0.004)
      biases = tf.get_variable('biases', [384],
          initializer=tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
      weights = utils.variable_with_weight_decay('weights', shape=[384, 192],
          stddev=0.04, wd=0.004)
      biases = tf.get_variable('biases', [192],
          initializer=tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    return local4

  def add_model(self):
    self.cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_size,
        state_is_tuple=True)

    with tf.variable_scope('LSTM') as scope:
      # inputs_placeholder: batch_size x max_time x height x window_size x depth
      # data_cnn: batch_size*max_time x height x window_size x depth
      data_cnn = tf.reshape(self.inputs_placeholder,
          [self.max_time*self.config.batch_size, self.config.height,
          self.config.window_size, self.config.depth])

      img_features = self.CNN(data_cnn)

      # img_features: batch_size*max_time x feature_size (192)
      # data_encoder: batch_size x max_time x feature_size
      data_encoder = tf.reshape(img_features, (self.config.batch_size,
          self.max_time, -1))

      rnn_outputs, _ = tf.nn.dynamic_rnn(self.cell, data_encoder,
          sequence_length=self.sequence_length_placeholder, dtype=tf.float32)

      # rnn_outputs: batch_size x max_time x lstm_size
      return rnn_outputs

  def add_projection(self, rnn_outputs):
    with tf.variable_scope('Projection'):
      W = tf.get_variable('Weight', [self.config.lstm_size,
          self.config.embed_size],
          initializer=tf.random_normal_initializer(0.0, 1e-3))
      b = tf.get_variable('Bias', [self.config.embed_size],
          initializer=tf.constant_initializer(0.0))

      rnn_outputs_reshape = tf.reshape(rnn_outputs,
          (self.config.batch_size*self.max_time, self.config.lstm_size))
      outputs = tf.matmul(rnn_outputs_reshape, W)+b
      outputs = tf.reshape(outputs, (self.config.batch_size, self.max_time, -1))
      outputs = tf.transpose(outputs, perm=[1, 0, 2])

    # outputs: max_time x batch_size x embed_size
    return outputs

  def add_loss_op(self, outputs):
    loss = tf.contrib.ctc.ctc_loss(outputs, self.labels_placeholder,
        self.sequence_length_placeholder, ctc_merge_repeated=False)
    loss = tf.reduce_mean(loss)
    return loss

  def add_decoder(self, outputs):
    decoded, _ = tf.contrib.ctc.ctc_greedy_decoder(outputs,
        self.sequence_length_placeholder, merge_repeated=False)
    pred = tf.sparse_tensor_to_dense(decoded[0])
    return pred

  def add_training_op(self, loss):
    optimizer = tf.train.RMSPropOptimizer(self.config.lr)
    train_op = optimizer.minimize(loss)
    return train_op

def main():
  config = Config()
  model = DTRN_Model(config)
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  with tf.Session() as session:
    session.run(init)
    if model.config.load_ckpt:
      saver.restore(session, model.config.ckpt_dir+'model.ckpt')
      logger.info('model restored')

    iterator_train = utils.data_iterator(
        model.imgs_train, model.words_embed_train, model.time_train,
        model.config.num_epochs, model.config.batch_size, model.max_time)

    iterator_test = utils.data_iterator(
        model.imgs_test, model.words_embed_test, model.time_test, 1,
        model.config.test_size, model.max_time)

    num_examples = model.imgs_train.shape[0]
    num_steps = int(math.ceil(
        num_examples*model.config.num_epochs/model.config.batch_size))

    for step, (inputs, labels_sparse, sequence_length,
        epoch) in enumerate(iterator_train):
      logger.info('epoch %d/%d, step %d/%d', epoch, model.config.num_epochs,
          step, num_steps)

      feed = {model.inputs_placeholder: inputs,
              model.labels_placeholder: labels_sparse,
              model.sequence_length_placeholder: sequence_length}

      ret = session.run([model.train_op, model.loss], feed_dict=feed)
      logger.info('loss: %f', ret[1])

      if step%model.config.save_every_n_steps == 0:
        save_path = saver.save(session, model.config.ckpt_dir+'model.ckpt')
        logger.info('model saved in file: %s', save_path)

      if step%model.config.test_every_n_steps == 0:
        ret = session.run([model.pred], feed_dict=feed)
        print ret[0]

if __name__ == '__main__':
  main()
