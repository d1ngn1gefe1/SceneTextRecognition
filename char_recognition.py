import tensorflow as tf
import numpy as np
import utils
from utils import logger
from cnn import CNN
import h5py
import math
import os
import time
import json
import sys

class Config():
  def __init__(self):
    with open('config.json', 'r') as json_file:
      json_data = json.load(json_file)
      self.dataset_dir = json_data['dataset_dir']
      self.height = json_data['height']
      self.window_size = json_data['window_size']
      self.depth = json_data['depth']
      self.embed_size = json_data['embed_size']
      self.lr = json_data['lr']
      self.keep_prob = json_data['keep_prob']
      self.num_epochs = json_data['num_epochs']
      self.batch_size = json_data['batch_size']
      self.debug = json_data['debug']
      self.debug_size = json_data['debug_size']
      self.test_only = json_data['test_only']

class CNN_Model():
  def __init__(self, config):
    self.config = config
    self.load_data(self.config.debug, self.config.test_only)
    self.add_placeholders()
    self.logits = self.add_model()
    self.loss = self.add_loss_op(self.logits)
    self.train_op = self.add_training_op(self.loss)

  def load_data(self, debug=False, test_only=False):
    filename_test = os.path.join(self.config.dataset_dir, 'test.hdf5')
    f_test = h5py.File(filename_test, 'r')
    self.char_imgs_test = f_test.get('char_imgs')
    self.chars_embed_test = f_test.get('chars_embed')
    logger.info('loading test data (%d examples)', self.char_imgs_test.shape[0])
    f_test.close()

    filename_train = os.path.join(self.config.dataset_dir, 'train.hdf5')
    f_train = h5py.File(filename_train, 'r')
    self.char_imgs_train = np.array(f_train.get('char_imgs'), dtype=np.uint8)
    self.chars_embed_train = np.array(f_train.get('chars_embed'), dtype=np.uint8)
    logger.info('loading training data (%d characters)', self.char_imgs_train.shape[0])
    f_train.close()

    if self.config.debug:
      self.char_imgs_train = self.char_imgs_train[:self.config.debug_size]
      self.chars_embed_train = self.chars_embed_train[:self.config.debug_size]

  def add_placeholders(self):
    # batch_size x height x width x depth
    self.inputs_placeholder = tf.placeholder(tf.float32,
        shape=[self.config.batch_size, self.config.height,
        self.config.window_size, self.config.depth])

    # batch_size
    self.labels_placeholder = tf.placeholder(tf.float32, shape=[self.config.batch_size, self.config.embed_size])

    # float
    self.keep_prob_placeholder = tf.placeholder(tf.float32)

  def add_model(self):
    with tf.variable_scope('CNN') as scope:
      logits = CNN(self.inputs_placeholder, self.config.depth, self.config.embed_size, self.keep_prob_placeholder)

    return logits

  def add_loss_op(self, logits):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits, self.labels_placeholder)
    loss = tf.reduce_mean(losses)
    return loss

  def add_training_op(self, loss):
    optimizer = tf.train.AdamOptimizer(self.config.lr)
    train_op = optimizer.minimize(loss)
    return train_op

def main():
  config = Config()
  model = CNN_Model(config)
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  with tf.Session() as session:
    session.run(init)

    iterator_train = utils.data_iterator_char(model.char_imgs_train, model.chars_embed_train, model.config.num_epochs, model.config.batch_size, model.config.embed_size)

    num_chars = model.char_imgs_train.shape[0]

    losses_train = []
    cur_epoch = 0
    step_epoch = 0
    for step, (inputs_train, labels_train, epoch_train) in enumerate(iterator_train):
      # new epoch, calculate average loss from last epoch
      if epoch_train != cur_epoch:
        logger.info('average training loss in epoch %d: %f\n', cur_epoch,
            np.mean(losses_train[step_epoch:]))
        step_epoch = step
        cur_epoch = epoch_train

      feed_train = {model.inputs_placeholder: inputs_train,
                    model.labels_placeholder: labels_train,
                    model.keep_prob_placeholder: model.config.keep_prob}

      ret_train = session.run([model.train_op, model.loss], feed_dict=feed_train)
      losses_train.append(ret_train[1])
    #   logger.info('epoch %d, step %d: training loss = %f', epoch_train, step,
    #       ret_train[1])

if __name__ == '__main__':
  main()
