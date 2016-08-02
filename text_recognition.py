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
      self.jittering_size = int(json_data['jittering_percent']*self.height)
      self.stride = json_data['stride']
      self.lr = json_data['lr']
      self.keep_prob = json_data['keep_prob']
      self.keep_prob_transformer = json_data['keep_prob_transformer']
      self.lstm_size = json_data['lstm_size']
      self.num_epochs = json_data['num_epochs']
      self.batch_size = json_data['batch_size']
      self.debug = json_data['debug']
      self.debug_size = json_data['debug_size']
      self.text_load_char_ckpt = json_data['text_load_char_ckpt']
      self.load_text_ckpt = json_data['load_text_ckpt']
      self.ckpt_dir = json_data['ckpt_dir']
      self.test_only = json_data['test_only']
      self.test_and_save_every_n_steps = json_data['test_and_save_every_n_steps']
      self.test_size = json_data['test_size']
      self.test_size = self.test_size-self.test_size%self.batch_size
      self.gpu = json_data['gpu']

class TEXT_Model():
  def __init__(self, config):
    self.config = config
    self.load_data(self.config.debug, self.config.test_only)
    self.add_placeholders()
    self.rnn_outputs = self.add_model()
    self.outputs = self.add_projection(self.rnn_outputs)
    self.loss = self.add_loss_op(self.outputs)
    self.pred, self.groundtruth, self.dists = self.add_decoder(self.outputs)
    self.train_op = self.add_training_op(self.loss)

  def load_data(self, debug=False, test_only=False):
    filename_test = os.path.join(self.config.dataset_dir, 'test.hdf5')
    f_test = h5py.File(filename_test, 'r')
    self.imgs_test = np.array(f_test.get('imgs'), dtype=np.uint8)
    self.words_embed_test = f_test.get('words_embed')[()].tolist()
    self.time_test = np.array(f_test.get('time'), dtype=np.uint8)
    logger.info('loading test data (%d examples)', self.imgs_test.shape[0])
    f_test.close()

    if self.imgs_test.shape[0] > self.config.test_size:
      self.imgs_test = self.imgs_test[:self.config.test_size]
      self.words_embed_test = self.words_embed_test[:self.config.test_size]
      self.time_test = self.time_test[:self.config.test_size]

    if test_only:
      self.max_time = np.amax(self.time_test)
      self.imgs_test = self.imgs_test[:, :self.max_time]
      return

    filename_train = os.path.join(self.config.dataset_dir, 'train.hdf5')
    f_train = h5py.File(filename_train, 'r')
    self.imgs_train = np.array(f_train.get('imgs'), dtype=np.uint8)
    self.words_embed_train = f_train.get('words_embed')[()].tolist()
    self.time_train = np.array(f_train.get('time'), dtype=np.uint8)
    logger.info('loading training data (%d examples)', self.imgs_train.shape[0])
    f_train.close()

    if self.config.debug:
      self.imgs_train = self.imgs_train[:self.config.debug_size]
      self.words_embed_train = self.words_embed_train[:self.config.debug_size]
      self.time_train = self.time_train[:self.config.debug_size]

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

    # max_time x batch_size x embed_size
    self.outputs_mask_placeholder = tf.placeholder(tf.float32,
        shape=[self.max_time, self.config.batch_size, self.config.embed_size])

    # float
    self.keep_prob_placeholder = tf.placeholder(tf.float32)
    self.keep_prob_transformer_placeholder = tf.placeholder(tf.float32)

  def add_model(self):
    self.cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_size,
        state_is_tuple=True)

    with tf.variable_scope('TEXT') as scope:
      # inputs_placeholder: batch_size x max_time x height x window_size x depth
      # data_cnn: batch_size*max_time x height x window_size x depth
      data_cnn = tf.reshape(self.inputs_placeholder,
          [self.max_time*self.config.batch_size, self.config.height,
          self.config.window_size, self.config.depth])

      # img_features: batch_size*max_time x feature_size (128)
      img_features, self.variables_STN, self.variables_CNN, \
          self.variables_FC, self.saver_STN, self.saver_CNN, self.saver_FC, \
          x_trans = cnn.CNN(data_cnn, self.config.height,
          self.config.window_size, self.config.depth, self.config.embed_size,
          self.keep_prob_placeholder, self.keep_prob_transformer_placeholder)

      # data_encoder: batch_size x max_time x feature_size
      data_encoder = tf.reshape(img_features, (self.config.batch_size,
          self.max_time, self.config.embed_size))

      # rnn_outputs: max_time x batch_size x lstm_size
      rnn_outputs, _ = tf.nn.dynamic_rnn(self.cell,
          data_encoder, sequence_length=self.sequence_length_placeholder,
          dtype=tf.float32, time_major=False)

    return rnn_outputs

  def add_projection(self, rnn_outputs):
    with tf.variable_scope('Projection'):
      W = tf.get_variable('Weight', [self.config.lstm_size,
          self.config.embed_size],
          initializer=tf.contrib.layers.xavier_initializer())
      b = tf.get_variable('Bias', [self.config.embed_size],
          initializer=tf.constant_initializer(0.0))

      rnn_outputs_reshape = tf.reshape(rnn_outputs,
          (self.config.batch_size*self.max_time, self.config.lstm_size))
      outputs = tf.matmul(rnn_outputs_reshape, W)+b
      outputs = tf.nn.log_softmax(outputs)
      outputs = tf.reshape(outputs, (self.config.batch_size, self.max_time, -1))
      outputs = tf.transpose(outputs, perm=[1, 0, 2])
      outputs = tf.add(outputs, self.outputs_mask_placeholder)

      # outputs: max_time x batch_size x embed_size
    return outputs

  def add_loss_op(self, outputs):
    loss = tf.contrib.ctc.ctc_loss(outputs, self.labels_placeholder,
        self.sequence_length_placeholder)
    loss = tf.reduce_mean(loss)

    return loss

  def add_decoder(self, outputs):
    decoded, _ = tf.contrib.ctc.ctc_beam_search_decoder(outputs,
        self.sequence_length_placeholder, merge_repeated=False)
    decoded = tf.to_int32(decoded[0])

    pred = tf.sparse_tensor_to_dense(decoded,
        default_value=self.config.embed_size-1)
    groundtruth = tf.sparse_tensor_to_dense(self.labels_placeholder,
        default_value=self.config.embed_size-1)
    dists = tf.edit_distance(decoded, self.labels_placeholder, normalize=False)

    return (pred, groundtruth, dists)

  def add_training_op(self, loss):
    self.variables_LSTM_CTC = tf.trainable_variables()
    for var in self.variables_STN:
      self.variables_LSTM_CTC.remove(var)
    for var in self.variables_CNN:
      self.variables_LSTM_CTC.remove(var)

    train_op1 = tf.train.AdamOptimizer(self.config.lr*0.01).minimize(loss, var_list=self.variables_STN)
    train_op2 = tf.train.AdamOptimizer(self.config.lr*0.01).minimize(loss, var_list=self.variables_CNN)
    train_op3 = tf.train.AdamOptimizer(self.config.lr).minimize(loss, var_list=self.variables_LSTM_CTC)
    train_op = tf.group(train_op1, train_op2, train_op3)

    return train_op

def main():
  config = Config()
  model = TEXT_Model(config)
  init = tf.initialize_all_variables()

  if not os.path.exists(model.config.ckpt_dir):
    os.makedirs(model.config.ckpt_dir)

  config = tf.ConfigProto(allow_soft_placement=True)

  with tf.Session(config=config) as session:
    session.run(init)
    best_loss = float('inf')
    corresponding_accuracy = 0 # accuracy corresponding to the best loss
    best_accuracy = 0
    corresponding_loss = float('inf') # loss corresponding to the best accuracy
    model.saver = tf.train.Saver()

    # restore previous session
    if model.config.text_load_char_ckpt:
      if os.path.isfile(model.config.ckpt_dir+'model_best_accuracy_stn.ckpt'):
        model.saver_STN.restore(session, model.config.ckpt_dir+'model_best_accuracy_stn.ckpt')
        model.saver_CNN.restore(session, model.config.ckpt_dir+'model_best_accuracy_cnn.ckpt')
        logger.info('char model restored')
    elif model.config.load_text_ckpt or model.config.test_only:
      model.saver.restore(session, model.config.ckpt_dir+'model_best_loss_full.ckpt')
      logger.info('full model restored')
      if os.path.isfile(model.config.ckpt_dir+'text_best_loss.npy'):
        best_loss = np.load(model.config.ckpt_dir+'text_best_loss.npy')
        logger.info('best loss: '+str(best_loss))
      if os.path.isfile(model.config.ckpt_dir+'text_corr_distances.npy'):
        corresponding_distances = np.load(model.config.ckpt_dir+'text_corr_distances.npy')
        logger.info('corresponding distances: ')
        logger.info(corresponding_distances)
      if os.path.isfile(model.config.ckpt_dir+'text_best_distances.npy'):
        best_distances = np.load(model.config.ckpt_dir+'text_best_distances.npy')
        logger.info('best distances: ')
        logger.info(best_distances)
      if os.path.isfile(model.config.ckpt_dir+'text_corr_loss.npy'):
        corresponding_loss = np.load(model.config.ckpt_dir+'text_corr_loss.npy')
        logger.info('corresponding loss: '+str(corresponding_loss))

    iterator_train = utils.data_iterator(
        model.imgs_train, model.words_embed_train, model.time_train,
        model.config.num_epochs, model.config.batch_size, model.max_time,
        model.config.embed_size, model.config.jittering_size, False)

    num_examples = model.imgs_train.shape[0]
    num_steps = int(math.ceil(
        num_examples*model.config.num_epochs/model.config.batch_size))

    losses_train = []
    cur_epoch = 0
    step_epoch = 0
    for step, (inputs_train, labels_sparse_train, sequence_length_train,
        outputs_mask_train, epoch_train) in enumerate(iterator_train):

      # test
      if step%model.config.test_and_save_every_n_steps == 0:
        losses_test = []
        dists_test = np.zeros((model.config.batch_size))
        iterator_test = utils.data_iterator(
          model.imgs_test, model.words_embed_test, model.time_test,
          1, model.config.batch_size, model.max_time,
          model.config.embed_size, model.config.jittering_size, False)

        for step_test, (inputs_test, labels_sparse_test, sequence_length_test,
            outputs_mask_test, epoch_test) in enumerate(iterator_test):
          feed_test = {model.inputs_placeholder: inputs_test,
                       model.labels_placeholder: labels_sparse_test,
                       model.sequence_length_placeholder: sequence_length_test,
                       model.outputs_mask_placeholder: outputs_mask_test,
                       model.keep_prob_placeholder: 1.0,
                       model.keep_prob_transformer_placeholder: 1.0}

          ret_test = session.run([model.loss, model.dists],
              feed_dict=feed_test)
          losses_test.append(ret_test[0])
          dists_test = np.concatenate((dists_test, ret_test[1]))

        cur_loss = np.mean(losses_test)
        cur_dist = np.mean(dists_test)
        stats = np.bincount(dists_test.astype(int))

        logger.info('<-------------------->')
        logger.info('average test loss: %f (#batches = %d)',
            cur_loss, len(losses_test))
        logger.info('average edit distance: %f (#batches = %d)',
            cur_dist, len(dists_test))
        logger.info(stats)
        logger.info('<-------------------->')

        if model.config.test_only:
          return

        if cur_loss < best_loss:
          best_loss = cur_loss
          save_path = model.saver.save(session, model.config.ckpt_dir+'model_best_loss_full.ckpt')
          np.save(model.config.ckpt_dir+'text_best_loss.npy', np.array(best_loss))
          logger.info('model saved in file: %s', save_path)
        else:
          save_path = model.saver.save(session, model.config.ckpt_dir+'model_full.ckpt')
          logger.info('model saved in file: %s', save_path)

      # new epoch, calculate average loss from last epoch
      if epoch_train != cur_epoch:
        logger.info('average training loss in epoch %d: %f', cur_epoch,
            np.mean(losses_train[step_epoch:]))
        #logger.info('average loss overall: %f', np.mean(losses_train))
        step_epoch = step
        cur_epoch = epoch_train

      feed_train = {model.inputs_placeholder: inputs_train,
                    model.labels_placeholder: labels_sparse_train,
                    model.sequence_length_placeholder: sequence_length_train,
                    model.outputs_mask_placeholder: outputs_mask_train,
                    model.keep_prob_placeholder: model.config.keep_prob,
                    model.keep_prob_transformer_placeholder: \
                        model.config.keep_prob_transformer}

      ret_train = session.run([model.train_op, model.loss],
          feed_dict=feed_train)
      losses_train.append(ret_train[1])
      # logger.info('epoch %d, step %d: training loss = %f', epoch_train, step,
      # ret_train[1])

if __name__ == '__main__':
  main()
