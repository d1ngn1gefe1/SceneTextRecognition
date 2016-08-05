from utils import logger
import tensorflow as tf
import numpy as np
import utils
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
      self.jittering_size = int(json_data['jittering_percent']*self.height)
      self.lr = json_data['lr']
      self.keep_prob = json_data['keep_prob']
      self.keep_prob_transformer = json_data['keep_prob_transformer']
      self.num_epochs = json_data['num_epochs']
      self.batch_size = json_data['batch_size']
      self.debug = json_data['debug']
      self.debug_size = json_data['debug_size']
      self.load_char_ckpt = json_data['load_char_ckpt']
      self.ckpt_dir = json_data['ckpt_dir']
      self.test_only = json_data['test_only']
      self.test_and_save_every_n_steps = \
          json_data['test_and_save_every_n_steps']
      self.visualize = json_data['visualize']
      self.visualize_dir = json_data['visualize_dir']

class CHAR_Model():
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
    self.char_imgs_test = np.array(f_test.get('char_imgs'), dtype=np.uint8)
    self.chars_embed_test = np.array(f_test.get('chars_embed'), dtype=np.uint8)
    logger.info('loading test data (%d examples)', self.char_imgs_test.shape[0])
    f_test.close()

    filename_train = os.path.join(self.config.dataset_dir, 'train.hdf5')
    f_train = h5py.File(filename_train, 'r')
    self.char_imgs_train = np.array(f_train.get('char_imgs'), dtype=np.uint8)
    self.chars_embed_train = np.array(f_train.get('chars_embed'),
        dtype=np.uint8)
    logger.info('loading training data (%d characters)',
        self.char_imgs_train.shape[0])
    f_train.close()

    if self.config.debug:
      self.char_imgs_train = self.char_imgs_train[:self.config.debug_size]
      self.chars_embed_train = self.chars_embed_train[:self.config.debug_size]

  def add_placeholders(self):
    # batch_size x height x width x depth
    self.inputs_placeholder = tf.placeholder(tf.float32,
        shape=[self.config.batch_size, self.config.height,
        self.config.window_size, self.config.depth])

    # batch_size x embed_size
    self.labels_placeholder = tf.placeholder(tf.float32,
        shape=[self.config.batch_size, self.config.embed_size])

    # float
    self.keep_prob_placeholder = tf.placeholder(tf.float32)
    self.keep_prob_transformer_placeholder = tf.placeholder(tf.float32)

  def add_model(self):
    with tf.variable_scope('CHAR') as scope:
      logits, self.variables_STN, self.variables_CNN, self.saver_STN, \
          self.saver_CNN, self.x_trans = cnn.CNN(self.inputs_placeholder,
          self.config.height, self.config.window_size, self.config.depth,
          self.keep_prob_placeholder, self.keep_prob_transformer_placeholder)

      with tf.variable_scope('fc1') as scope:
        h_fc1 = tf.nn.relu(logits)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      with tf.variable_scope('fc2') as scope:
        W_fc2 = tf.get_variable('Weight', [256, self.config.embed_size],
            initializer=tf.contrib.layers.xavier_initializer())
        b_fc2 = tf.get_variable('Bias', [self.config.embed_size],
            initializer=tf.constant_initializer(0))
        logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

      self.variables_FC = [W_fc2, b_fc2]
      self.saver_FC = tf.train.Saver({'W_fc2': W_fc2, 'b_fc2': b_fc2})

    return logits

  def add_loss_op(self, logits):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits,
        self.labels_placeholder)
    loss = tf.reduce_mean(losses)

    self.diff = tf.argmax(logits, 1)-tf.argmax(self.labels_placeholder, 1)

    return loss

  def add_training_op(self, loss):
    train_op1 = tf.train.AdamOptimizer(0.5*self.config.lr).minimize(loss,
        var_list=self.variables_STN)
    train_op2 = tf.train.AdamOptimizer(self.config.lr).minimize(loss,
        var_list=self.variables_CNN)
    train_op3 = tf.train.AdamOptimizer(self.config.lr).minimize(loss,
        var_list=self.variables_FC)
    train_op = tf.group(train_op1, train_op2, train_op3)

    return train_op

def main():
  config = Config()
  model = CHAR_Model(config)
  init = tf.initialize_all_variables()

  if not os.path.exists(model.config.ckpt_dir):
    os.makedirs(model.config.ckpt_dir)

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 1.0

  with tf.Session(config=config) as session:
    session.run(init)
    best_loss = float('inf')
    corresponding_accuracy = 0 # accuracy corresponding to the best loss
    best_accuracy = 0
    corresponding_loss = float('inf') # loss corresponding to the best accuracy

    # restore previous session
    if model.config.load_char_ckpt or model.config.test_only:
      if os.path.isfile(model.config.ckpt_dir+'model_best_accuracy_stn.ckpt'):
        model.saver_STN.restore(session, model.config.ckpt_dir+'model_best_accuracy_stn.ckpt')
        model.saver_CNN.restore(session, model.config.ckpt_dir+'model_best_accuracy_cnn.ckpt')
        model.saver_FC.restore(session, model.config.ckpt_dir+'model_best_accuracy_fc.ckpt')
        logger.info('<-------------------->')
        logger.info('model restored')
      if os.path.isfile(model.config.ckpt_dir+'char_best_loss.npy'):
        best_loss = np.load(model.config.ckpt_dir+'char_best_loss.npy')
        logger.info('best loss: '+str(best_loss))
      if os.path.isfile(model.config.ckpt_dir+'char_corr_accuracy.npy'):
        corresponding_accuracy = np.load(model.config.ckpt_dir+ \
            'char_corr_accuracy.npy')
        logger.info('corresponding accuracy: '+str(corresponding_accuracy))
      if os.path.isfile(model.config.ckpt_dir+'char_best_accuracy.npy'):
        best_accuracy = np.load(model.config.ckpt_dir+'char_best_accuracy.npy')
        logger.info('best accuracy: '+str(best_accuracy))
      if os.path.isfile(model.config.ckpt_dir+'char_corr_loss.npy'):
        corresponding_loss = np.load(model.config.ckpt_dir+'char_corr_loss.npy')
        logger.info('corresponding loss: '+str(corresponding_loss))
      logger.info('<-------------------->')

    iterator_train = utils.data_iterator_char(model.char_imgs_train,
        model.chars_embed_train, model.config.num_epochs,
        model.config.batch_size, model.config.embed_size,
        model.config.jittering_size, False)

    num_chars = model.char_imgs_train.shape[0]

    losses_train = []
    accuracies_train = []
    cur_epoch = 0
    step_epoch = 0

    # each step corresponds to one batch
    for step, (inputs_train, labels_train, epoch_train) in \
        enumerate(iterator_train):

      # test & save model
      if step%model.config.test_and_save_every_n_steps == 0:
        losses_test = []
        accuracies_test = []
        iterator_test = utils.data_iterator_char(model.char_imgs_test,
            model.chars_embed_test, 1, model.config.batch_size,
            model.config.embed_size, model.config.jittering_size, True)

        for step_test, (inputs_test, labels_test, epoch_test) in \
            enumerate(iterator_test):
          feed_test = {model.inputs_placeholder: inputs_test,
                       model.labels_placeholder: labels_test,
                       model.keep_prob_placeholder: 1.0,
                       model.keep_prob_transformer_placeholder: 1.0}

          ret_test = session.run([model.loss, model.diff, model.x_trans],
              feed_dict=feed_test)
          losses_test.append(ret_test[0])
          accuracies_test.append(float(np.sum(ret_test[1] == 0))/\
              ret_test[1].shape[0])

          # visualize the STN results
          if model.config.visualize and step_test < 10:
            utils.save_imgs(inputs_test, model.config.visualize_dir,
                'original'+str(step_test)+'-')
            utils.save_imgs(ret_test[2], model.config.visualize_dir,
                'trans'+str(step_test)+'-')

        cur_loss = np.mean(losses_test)
        cur_accuracy = np.mean(accuracies_test)

        if model.config.test_only:
          return

        # save three models: current model, model with the lowest loss, model
        # with the highest accuracy
        if cur_loss >= best_loss and cur_accuracy <= best_accuracy:
          model.saver_STN.save(session, model.config.ckpt_dir+'model_stn.ckpt')
          model.saver_CNN.save(session, model.config.ckpt_dir+'model_cnn.ckpt')
          model.saver_FC.save(session, model.config.ckpt_dir+'model_fc.ckpt')
          logger.info('cnn model saved')
        if cur_loss < best_loss:
          best_loss = cur_loss
          corresponding_accuracy = cur_accuracy
          model.saver_STN.save(session, model.config.ckpt_dir+'model_best_loss_stn.ckpt')
          model.saver_CNN.save(session, model.config.ckpt_dir+'model_best_loss_cnn.ckpt')
          model.saver_FC.save(session, model.config.ckpt_dir+'model_best_loss_fc.ckpt')
          logger.info('best loss model saved')
          np.save(model.config.ckpt_dir+'char_best_loss.npy', np.array(best_loss))
          np.save(model.config.ckpt_dir+'char_corr_accuracy.npy', np.array(corresponding_accuracy))
        if cur_accuracy > best_accuracy:
          best_accuracy = cur_accuracy
          corresponding_loss = cur_loss
          model.saver_STN.save(session, model.config.ckpt_dir+'model_best_accuracy_stn.ckpt')
          model.saver_CNN.save(session, model.config.ckpt_dir+'model_best_accuracy_cnn.ckpt')
          model.saver_FC.save(session, model.config.ckpt_dir+'model_best_accuracy_fc.ckpt')
          logger.info('best accuracy model saved')
          np.save(model.config.ckpt_dir+'char_best_accuracy.npy', np.array(best_accuracy))
          np.save(model.config.ckpt_dir+'char_corr_loss.npy', np.array(corresponding_loss))

        logger.info('<-------------------->')
        logger.info('test loss: %f (#batches = %d)',
            cur_loss, len(losses_test))
        logger.info('test accuracy: %f (#batches = %d)',
            cur_accuracy, len(accuracies_test))
        logger.info('best test loss: %f, corresponding accuracy: %f',
            best_loss, corresponding_accuracy)
        logger.info('best test accuracy: %f, corresponding loss: %f',
            best_accuracy, corresponding_loss)
        logger.info('<-------------------->')

      # new epoch, calculate average training loss and accuracy from last epoch
      if epoch_train != cur_epoch:
        logger.info('training loss in epoch %d, step %d: %f', cur_epoch, step,
            np.mean(losses_train[step_epoch:]))
        logger.info('training accuracy in epoch %d, step %d: %f', cur_epoch, step,
            np.mean(accuracies_train[step_epoch:]))
        step_epoch = step
        cur_epoch = epoch_train

      # train
      feed_train = {model.inputs_placeholder: inputs_train,
                    model.labels_placeholder: labels_train,
                    model.keep_prob_placeholder: model.config.keep_prob,
                    model.keep_prob_transformer_placeholder: \
                        model.config.keep_prob_transformer}

      ret_train = session.run([model.train_op, model.loss, model.diff],
          feed_dict=feed_train)
      losses_train.append(ret_train[1])
      accuracies_train.append(float(np.sum(ret_train[2] == 0))/\
          ret_train[2].shape[0])

if __name__ == '__main__':
  main()
