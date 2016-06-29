from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import seq2seq
import tensorflow as tf
import numpy as np
import utils
import cv2
import h5py
import math
import os

dataset_dir = '/home/local/ANT/zelunluo/Documents/IIIT5K/'
# img = cv2.imread(dataset_path + '/train/27_1.png', 0)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()'

class Config():
  lr = 1e-3

  batch_size = 32
  debug_size = 32
  num_epochs = 20000
  window_size = 32
  lstm_size = 64
  take = 12
  depth = 3

class DTRN_Model():
  def __init__(self, config):
    self.config = config
    self.load_data(debug=True)
    self.add_placeholders()
    self.rnn_outputs = self.add_model()
    self.outputs = self.add_projection(self.rnn_outputs)
    self.loss = self.add_loss_op(self.outputs)
    self.train_op = self.add_training_op(self.loss)

  def load_data(self, debug=False):
    filename_train = os.path.join(dataset_dir, 'train.hdf5')
    filename_test = os.path.join(dataset_dir, 'test.hdf5')

    f_train = h5py.File(filename_train, 'r')
    f_test = h5py.File(filename_test, 'r')

    # load train data
    self.imgs_train = np.array(f_train.get('imgs'), dtype=np.uint8)
    self.words_train = np.array(f_train.get('words'))
    self.imgs_length_train = \
        np.array(f_train.get('imgs_length'), dtype=np.int32)
    self.words_length_train = \
        np.array(f_train.get('words_length'), dtype=np.uint8)

    self.imgs_test = np.array(f_test.get('imgs'), dtype=np.uint8)
    self.words_test = np.array(f_test.get('words'))
    self.imgs_length_test = \
        np.array(f_test.get('imgs_length'), dtype=np.int32)
    self.words_length_test = \
        np.array(f_test.get('words_length'), dtype=np.uint8)

    if debug:
      stop = np.sum(self.imgs_length_train[:self.config.debug_size])
      self.imgs_train = self.imgs_train[:stop]
      self.words_train = self.words_train[:self.config.debug_size]
      self.imgs_length_train = self.imgs_length_train[:self.config.debug_size]
      self.words_length_train = self.words_length_train[:self.config.debug_size]

      stop = np.sum(self.imgs_length_test[:self.config.debug_size])
      self.imgs_test = self.imgs_test[:stop]
      self.words_test = self.words_test[:self.config.debug_size]
      self.imgs_length_test = self.imgs_length_test[:self.config.debug_size]
      self.words_length_test = self.words_length_test[:self.config.debug_size]

    self.height = self.imgs_train.shape[1]

    imgs_length_max = max(np.amax(self.imgs_length_train), \
        np.amax(self.imgs_length_test))
    words_length_max = max(np.amax(self.words_length_train), \
        np.amax(self.words_length_test))

    # timesteps for LSTM
    self.max_time_encoder = int(math.ceil(
        imgs_length_max/self.config.window_size))
    self.max_time_decoder = words_length_max-1
    print 'encoder length: ', self.max_time_encoder
    print 'decoder length: ', self.max_time_decoder

    f_train.close()

  def add_placeholders(self):
    # batch_size x max_time_encoder x height x width x depth
    self.inputs_encoder_placeholder = tf.placeholder(tf.float32, \
        shape=[self.config.batch_size, self.max_time_encoder, self.height, \
        self.config.window_size, self.config.depth])
    # batch_size x max_time_decoder x 62
    self.inputs_decoder_placeholder = tf.placeholder(tf.float32, \
        shape=[self.config.batch_size, self.max_time_decoder, 62])
    # batch_size x max_time_decoder x 62
    self.labels_placeholder = tf.placeholder(tf.float32, \
        shape=[self.config.batch_size, self.max_time_decoder, 62])
    # batch_size
    self.inputs_length_placeholder = tf.placeholder(tf.int32, \
        shape=[self.config.batch_size])
    # batch_size x 62
    self.labels_mask_placeholder = tf.placeholder(tf.float32, \
        shape=[self.config.batch_size, self.max_time_decoder])

  def CNN(self, images):
    # images: 4D tensor of size [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]

    with tf.variable_scope('conv1') as scope:
      kernel = utils.variable_with_weight_decay('weights', \
          shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = tf.get_variable('biases', [64], tf.constant_initializer(0.0))
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
      biases = tf.get_variable('biases', [64], tf.constant_initializer(0.1))
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
          [self.config.batch_size*self.max_time_encoder, -1])
      dim = reshape.get_shape()[1].value
      weights = utils._variable_with_weight_decay('weights', shape=[dim, 384], \
          stddev=0.04, wd=0.004)
      biases = tf.get_variable('biases', [384], tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
      weights = utils._variable_with_weight_decay('weights', shape=[384, 192], \
          stddev=0.04, wd=0.004)
      biases = tf.get_variable('biases', [192], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    return local4

  def add_model(self):
    self.cell = rnn_cell.BasicLSTMCell(self.config.lstm_size)

    with tf.variable_scope('LSTM') as scope:
      # inputs_encoder_placeholder: a training batch of size batch_size
      #                             x max_time_encoder x height x window_size
      #                             x depth
      # data_cnn: batch_size*max_time_encoder x height x window_size x depth
      data_cnn = tf.reshape(self.inputs_encoder_placeholder, \
          [self.config.batch_size*self.max_time_encoder, self.height, \
          self.config.window_size, self.config.depth])

      img_features = self.CNN(data_cnn)

      # img_features: batch_size x max_time_encoder x 192
      # data_encoder: a length max_time_encoder list of inputs, each a tensor
      #               of shape [batch_size, input_size]
      data_encoder = tf.reshape(data_encoder, (self.config.batch_size, \
          self.max_time_encoder, -1))
      data_encoder = tf.split(1, self.max_time_encoder, data_encoder)
      data_encoder = [tf.squeeze(datum) for datum in data_encoder]

      # inputs_decoder_placeholder: batch_size x max_time_decoder x 62
      # data_decoder: a length max_time_decoder list of inputs, each a tensor
      #               of shape [batch_size, input_size]
      data_decoder = tf.split(1, self.max_time_decoder, \
          self.inputs_decoder_placeholder)
      data_decoder = [tf.squeeze(datum) for datum in data_decoder]

      # encoder and decoder use the same RNN cell, but don't share parameters
      _, state_decoder = rnn.rnn(self.cell, data_encoder, dtype=tf.float32, \
          sequence_length=self.inputs_length_placeholder)
      outputs, _ = seq2seq.rnn_decoder(data_decoder, state_decoder, self.cell)

      return outputs

  def add_projection(self, rnn_outputs):
    with tf.variable_scope('Projection'):
      W = tf.get_variable('Weight', [self.config.lstm_size, 62],
          initializer=tf.random_normal_initializer(0.0, 1e-3))
      b = tf.get_variable('Bias', [62],
          initializer=tf.constant_initializer(0.0))
      outputs = [tf.matmul(rnn_output, W)+b for rnn_output in rnn_outputs]
    return outputs

  def add_loss_op(self, outputs):
    labels = tf.split(1, self.max_time_decoder, self.labels_placeholder)
    labels = [tf.squeeze(label) for label in labels]

    assert len(labels) == len(outputs)

    loss = 0.0
    for i in range(self.max_time_decoder):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(outputs[i], \
          labels[i])
      cross_entropy *= self.labels_mask_placeholder[:, i]
      #losses = tf.reduce_sum(cross_entropy)
      loss += tf.reduce_sum(cross_entropy)

    loss /= tf.reduce_sum(self.labels_mask_placeholder)

    self.out1 = self.inputs_decoder_placeholder[self.config.take, 0, :]
    self.out2 = tf.reshape(tf.concat(1, labels), \
        [-1, self.max_time_decoder, 62])[self.config.take, :, :]
    self.out3 = tf.reshape(tf.concat(1, outputs), \
        [-1, self.max_time_decoder, 62])[self.config.take, :, :]
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
        model.imgs_train, model.words_train, model.imgs_length_train, \
        model.words_length_train, model.config.batch_size, \
        model.config.window_size, model.config.num_epochs, \
        model.max_time_encoder, model.max_time_decoder)

    for step, (inputs_encoder, inputs_decoder, labels, inputs_length, \
        labels_mask) in enumerate(iterator):
      feed = {model.inputs_encoder_placeholder: inputs_encoder,
              model.inputs_decoder_placeholder: inputs_decoder,
              model.labels_placeholder: labels,
              model.inputs_length_placeholder: inputs_length,
              model.labels_mask_placeholder: labels_mask}

      ret = session.run([model.train_op, model.loss, model.out1, model.out2, \
          model.out3, model.rnn_outputs[3]], feed_dict=feed)

      out1 = utils.index2char(np.argmax(ret[2]))
      out2 = utils.index2char(np.argmax(ret[2]))
      for i in range(int(np.sum(labels_mask[model.config.take, :]))):
        out1 += utils.index2char(np.argmax(ret[3][i]))
        out2 += utils.index2char(np.argmax(ret[4][i]))

      print 'loss: ', ret[1]
      print 'label: ', out1
      print 'predict: ', out2
      print '\n'

      #print ret[4]

if __name__ == '__main__':
  main()
