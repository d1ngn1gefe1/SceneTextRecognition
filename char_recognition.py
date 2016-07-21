import tensorflow as tf
import numpy as np
import utils
from utils import logger
import cnn
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
      self.jittering = json_data['jittering']
      self.lr = json_data['lr']
      self.keep_prob = json_data['keep_prob']
      self.num_epochs = json_data['num_epochs']
      self.batch_size = json_data['batch_size']
      self.debug = json_data['debug']
      self.debug_size = json_data['debug_size']
      self.cnn_load_ckpt = json_data['cnn_load_ckpt']
      self.ckpt_dir = json_data['ckpt_dir']
      self.save_every_n_steps = json_data['save_every_n_steps']
      self.test_only = json_data['test_only']
      self.test_every_n_steps = json_data['test_every_n_steps']
      self.test_size = json_data['test_size']
      self.test_size = self.test_size-self.test_size%self.batch_size

class CNN_Model():
  def __init__(self, config):
    self.config = config
    self.load_data(self.config.debug, self.config.test_only)
    self.add_placeholders()
    self.logits, self.saver = self.add_model()
    self.loss = self.add_loss_op(self.logits)
    self.train_op = self.add_training_op(self.loss)

  def load_data(self, debug=False, test_only=False):
    filename_test = os.path.join(self.config.dataset_dir, 'test.hdf5')
    f_test = h5py.File(filename_test, 'r')
    self.char_imgs_test = np.array(f_test.get('char_imgs'), dtype=np.uint8)
    self.chars_embed_test = np.array(f_test.get('chars_embed'), dtype=np.uint8)
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
      logits, saver, variables = cnn.CNN(self.inputs_placeholder,
          self.config.depth, self.config.embed_size, self.keep_prob_placeholder)

    return logits, saver

  def add_loss_op(self, logits):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits, self.labels_placeholder)
    loss = tf.reduce_mean(losses)

    self.diff = tf.argmax(logits, 1)-tf.argmax(self.labels_placeholder, 1)

    return loss

  def add_training_op(self, loss):
    optimizer = tf.train.AdamOptimizer(self.config.lr)
    train_op = optimizer.minimize(loss)
    return train_op

def main():
  config = Config()
  model = CNN_Model(config)
  init = tf.initialize_all_variables()

  if not os.path.exists(model.config.ckpt_dir):
    os.makedirs(model.config.ckpt_dir)

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=config) as session:
    session.run(init)

    # restore previous session
    if model.config.cnn_load_ckpt or model.config.test_only:
      if os.path.isfile(model.config.ckpt_dir+'model_cnn.ckpt'):
        model.saver.restore(session, model.config.ckpt_dir+'model_cnn.ckpt')
        logger.info('model restored')

    iterator_train = utils.data_iterator_char(model.char_imgs_train,
        model.chars_embed_train, model.config.num_epochs,
        model.config.batch_size, model.config.embed_size,
        model.config.jittering)

    num_chars = model.char_imgs_train.shape[0]

    losses_train = []
    accuracies_train = []
    cur_epoch = 0
    step_epoch = 0
    for step, (inputs_train, labels_train, epoch_train) in enumerate(iterator_train):
      # test
      if step%model.config.test_every_n_steps == 0:
        losses_test = []
        accuracies_test = []
        iterator_test = utils.data_iterator_char(model.char_imgs_test,
            model.chars_embed_test, 1, model.config.batch_size,
            model.config.embed_size, model.config.jittering)

        for step_test, (inputs_test, labels_test, epoch_test) in enumerate(iterator_test):
          feed_test = {model.inputs_placeholder: inputs_test,
                       model.labels_placeholder: labels_test,
                       model.keep_prob_placeholder: 1.0}

          ret_test = session.run([model.loss, model.diff],feed_dict=feed_test)
          losses_test.append(ret_test[0])
          accuracies_test.append(float(np.sum(ret_test[1] == 0))/ret_test[1].shape[0])

        logger.info('<-------------------->')
        logger.info('average test loss: %f (#batches = %d)',
            np.mean(losses_test), len(losses_test))
        logger.info('average test accuracy: %f (#batches = %d)',
            np.mean(accuracies_test), len(accuracies_test))
        logger.info('<-------------------->')

        if model.config.test_only:
          return

      # new epoch, calculate average loss from last epoch
      if epoch_train != cur_epoch:
        logger.info('average training loss in epoch %d: %f', cur_epoch,
            np.mean(losses_train[step_epoch:]))
        logger.info('average training accuracy in epoch %d: %f', cur_epoch,
            np.mean(accuracies_train[step_epoch:]))
        step_epoch = step
        cur_epoch = epoch_train

      feed_train = {model.inputs_placeholder: inputs_train,
                    model.labels_placeholder: labels_train,
                    model.keep_prob_placeholder: model.config.keep_prob}

      ret_train = session.run([model.train_op, model.loss, model.diff], feed_dict=feed_train)
      losses_train.append(ret_train[1])
      accuracies_train.append(float(np.sum(ret_train[2] == 0))/ret_train[2].shape[0])

      if step%model.config.save_every_n_steps == 0:
        save_path = model.saver.save(session, model.config.ckpt_dir+'model_cnn.ckpt')
        logger.info('cnn model saved in file: %s', save_path)

if __name__ == '__main__':
  main()
