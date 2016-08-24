import cnn
import h5py
import json
import math
import numpy as np
import os
import stn
import sys
import tensorflow as tf
import time
import utils
from utils import logger


class Config():
  def __init__(self):
    with open('config.json', 'r') as json_file:
      json_data = json.load(json_file)

      self.dataset_dir_iiit5k = json_data['dataset_dir_iiit5k']
      self.dataset_dir_vgg = json_data['dataset_dir_vgg']
      self.use_iiit5k = json_data['use_iiit5k']

      self.height = json_data['height']
      self.window_size = json_data['window_size']
      self.stride = json_data['stride']
      # iiitk5: 75 sliding windows, 586 pixels
      self.max_timestep = json_data['max_timestep']
      self.jittering_percent = json_data['jittering_percent']
      self.embed_size = json_data['embed_size']

      self.lr = json_data['lr']
      self.num_epochs = json_data['num_epochs']
      self.batch_size = json_data['batch_size']

      self.num_lstm_layer = json_data['num_lstm_layer']
      self.use_stn = json_data['use_stn']

      self.debug = json_data['debug']
      self.debug_size = json_data['debug_size']
      self.load_text_ckpt = json_data['load_text_ckpt']
      self.text_load_char_ckpt = json_data['text_load_char_ckpt']
      self.ckpt_dir = json_data['ckpt_dir']
      self.test_only = json_data['test_only']
      self.test_and_save_every_n_steps = json_data['test_and_save_every_n_steps']
      self.test_size = json_data['test_size']
      self.test_size = self.test_size-self.test_size%self.batch_size
      self.visualize = json_data['visualize']
      self.visualize_dir = json_data['visualize_dir']
      self.print_pred = json_data['print_pred']
      self.use_baseline = json_data['use_baseline']


class TEXT_Model():
  def __init__(self, config):
    self.config = config
    self.add_placeholders()
    self.rnn_outputs = self.add_model()
    self.outputs = self.add_projection(self.rnn_outputs)
    self.loss = self.add_loss_op(self.outputs)
    self.pred, self.groundtruth, self.dists = self.add_decoder(self.outputs)
    self.train_op = self.add_training_op(self.loss)

  def add_placeholders(self):
    if self.config.use_baseline:
      self.inputs_placeholder = tf.placeholder(tf.float32,
          shape=[None, self.config.height])
    else:
      self.inputs_placeholder = tf.placeholder(tf.float32,
          shape=[None, self.config.height, self.config.window_size, 1])
    self.labels_placeholder = tf.sparse_placeholder(tf.int32)
    self.timesteps_placeholder = tf.placeholder(tf.int32)
    self.partition_placeholder = tf.placeholder(tf.int32)
    self.dropout_placeholder = tf.placeholder(tf.float32)

  def add_model(self):
    with tf.variable_scope('TEXT') as scope:
      if self.config.use_baseline:
        # inputs_placeholder: sum(time) x height
        data_rnn = tf.dynamic_partition(self.inputs_placeholder, self.partition_placeholder, self.config.batch_size)
        for i in range(len(data_rnn)):
          data_rnn[i] = tf.concat(0, [data_rnn[i], tf.zeros([self.config.max_timestep-self.timesteps_placeholder[i], self.config.height])])
        data_rnn = tf.pack(data_rnn, 0)
        data_rnn = tf.transpose(data_rnn, [1, 0, 2])
        data_rnn = tf.reshape(data_rnn, [-1, self.config.height])
        data_rnn = tf.split(0, self.config.max_timestep, data_rnn)

        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.height, state_is_tuple=True)
        stacked_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw]*(self.config.num_lstm_layer+6), state_is_tuple=True)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.height, state_is_tuple=True)
        stacked_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw]*(self.config.num_lstm_layer+6), state_is_tuple=True)
        rnn_outputs, _, _ = tf.nn.bidirectional_rnn(stacked_lstm_cell_fw,
            stacked_lstm_cell_bw, data_rnn,
            sequence_length=self.timesteps_placeholder, dtype=tf.float32)
        rnn_outputs = tf.pack(rnn_outputs)
        rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
      else:
        # inputs_placeholder: sum(time) x height x window_size x 1
        # logits: sum(time) x cnn_feature_size (128)
        if self.config.use_stn:
          self.x_trans, variables_STN, self.saver_STN = stn.STN( \
              self.inputs_placeholder, self.dropout_placeholder, \
              self.config.height, self.config.window_size)
          logits, variables_CNN, self.saver_CNN = cnn.CNN( \
              self.x_trans, self.dropout_placeholder, \
              self.config.height, self.config.window_size)
          self.variables_CHAR = variables_STN+variables_CNN
        else:
          logits, variables_CNN, self.saver_CNN = cnn.CNN( \
              self.inputs_placeholder, self.dropout_placeholder, \
              self.config.height, self.config.window_size)
          self.variables_CHAR = variables_CNN

        # data_rnn: a length max_timestep list of shape batch_size x cnn_feature_size (128)
        data_rnn = tf.dynamic_partition(logits, self.partition_placeholder, self.config.batch_size)
        for i in range(len(data_rnn)):
          data_rnn[i] = tf.concat(0, [data_rnn[i], tf.zeros([self.config.max_timestep-self.timesteps_placeholder[i], 128])])
        data_rnn = tf.pack(data_rnn, 0)
        data_rnn = tf.transpose(data_rnn, [1, 0, 2])
        data_rnn = tf.reshape(data_rnn, [-1, 128])
        data_rnn = tf.split(0, self.config.max_timestep, data_rnn)

        # rnn_outputs: batch_size x max_timestep x 2*cnn_feature_size (128)
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
        stacked_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw]*self.config.num_lstm_layer, state_is_tuple=True)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
        stacked_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw]*self.config.num_lstm_layer, state_is_tuple=True)
        rnn_outputs, _, _ = tf.nn.bidirectional_rnn(stacked_lstm_cell_fw,
            stacked_lstm_cell_bw, data_rnn,
            sequence_length=self.timesteps_placeholder, dtype=tf.float32)
        rnn_outputs = tf.pack(rnn_outputs)
        rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])

    return rnn_outputs

  def add_projection(self, rnn_outputs):
    if self.config.use_baseline:
      lstm_size = self.config.height
    else:
      lstm_size = 128

    with tf.variable_scope('Projection'):
      W = tf.get_variable('Weight', [2*lstm_size,
          self.config.embed_size],
          initializer=tf.contrib.layers.xavier_initializer())
      b = tf.get_variable('Bias', [self.config.embed_size],
          initializer=tf.constant_initializer(0.0))

      rnn_outputs_reshape = tf.reshape(rnn_outputs,
          (self.config.batch_size*self.config.max_timestep, 2*lstm_size))
      outputs = tf.matmul(rnn_outputs_reshape, W)+b
      outputs = tf.nn.log_softmax(outputs)
      outputs = tf.reshape(outputs, (self.config.batch_size, self.config.max_timestep, -1))
      outputs = tf.transpose(outputs, perm=[1, 0, 2])

      # outputs: max_timestep x batch_size x embed_size
    return outputs

  def add_loss_op(self, outputs):
    loss = tf.nn.ctc_loss(outputs, self.labels_placeholder,
        self.timesteps_placeholder)
    loss = tf.reduce_mean(loss)

    return loss

  def add_decoder(self, outputs):
    decoded, _ = tf.nn.ctc_beam_search_decoder(outputs,
        self.timesteps_placeholder, merge_repeated=False)
    decoded = tf.to_int32(decoded[0])

    pred = tf.sparse_tensor_to_dense(decoded,
        default_value=self.config.embed_size-1)
    groundtruth = tf.sparse_tensor_to_dense(self.labels_placeholder,
        default_value=self.config.embed_size-1)
    dists = tf.edit_distance(decoded, self.labels_placeholder, normalize=False)

    return (pred, groundtruth, dists)

  def add_training_op(self, loss):
    self.variables_LSTM_CTC = tf.trainable_variables()

    if self.config.use_baseline:
      train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss, var_list=self.variables_LSTM_CTC)
    else:
      for var in self.variables_CHAR:
        self.variables_LSTM_CTC.remove(var)

      train_op1 = tf.train.AdamOptimizer(0.1*self.config.lr).minimize(loss, var_list=self.variables_CHAR)
      train_op2 = tf.train.AdamOptimizer(self.config.lr).minimize(loss, var_list=self.variables_LSTM_CTC)
      train_op = tf.group(train_op1, train_op2)

    return train_op

def main():
  config = Config()
  model = TEXT_Model(config)
  init = tf.initialize_all_variables()

  if not os.path.exists(model.config.ckpt_dir):
    os.makedirs(model.config.ckpt_dir)

  config = tf.ConfigProto(allow_soft_placement=True)

  # utils.get_info(model.config.dataset_dir_iiit5k, model.config.dataset_dir_vgg, \
  #     model.config.use_iiit5k, model.config.height, model.config.window_size, \
  #     model.config.stride, model.config.num_epochs, model.config.batch_size)

  with tf.Session(config=config) as session:
    session.run(init)
    best_loss = float('inf')
    corresponding_accuracy = 0 # accuracy corresponding to the best loss
    best_accuracy = 0
    corresponding_loss = float('inf') # loss corresponding to the best accuracy
    model.saver = tf.train.Saver()

    # restore previous session
    if model.config.text_load_char_ckpt:
      if os.path.isfile(model.config.ckpt_dir+'model_best_accuracy_cnn.ckpt'):
        if model.config.use_stn:
          model.saver_STN.restore(session, model.config.ckpt_dir+'model_best_accuracy_stn.ckpt')
        model.saver_CNN.restore(session, model.config.ckpt_dir+'model_best_accuracy_cnn.ckpt')
        logger.info('char model restored')
    elif model.config.load_text_ckpt:
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

    if model.config.use_baseline:
      iterator_train = utils.data_iterator_baseline( \
          model.config.dataset_dir_iiit5k, model.config.dataset_dir_vgg, \
          model.config.use_iiit5k, model.config.height, model.config.window_size, \
          model.config.stride, model.config.max_timestep, model.config.jittering_percent, \
          model.config.num_epochs, model.config.batch_size, \
          True, model.config.debug, model.config.debug_size, \
          model.config.visualize, model.config.visualize_dir)
    else:
      iterator_train = utils.data_iterator( \
          model.config.dataset_dir_iiit5k, model.config.dataset_dir_vgg, \
          model.config.use_iiit5k, model.config.height, model.config.window_size, \
          model.config.stride, model.config.max_timestep, model.config.jittering_percent, \
          model.config.num_epochs, model.config.batch_size, \
          True, model.config.debug, model.config.debug_size, \
          model.config.visualize, model.config.visualize_dir)

    losses_train = []
    cur_epoch = 0
    step_epoch = 0

    for step_train, (inputs_train, labels_sparse_train, timesteps_train,
        partition_train, epoch_train) in enumerate(iterator_train):

      # test
      if step_train%model.config.test_and_save_every_n_steps == 0:
        losses_test = []
        dists_test = np.zeros((model.config.batch_size))

        if model.config.use_baseline:
          iterator_test = utils.data_iterator_baseline( \
              model.config.dataset_dir_iiit5k, model.config.dataset_dir_vgg,
              model.config.use_iiit5k, model.config.height, model.config.window_size, \
              model.config.stride, model.config.max_timestep, model.config.jittering_percent, \
              1, model.config.batch_size, \
              False, True, model.config.test_size, \
              model.config.visualize, model.config.visualize_dir)
        else:
          iterator_test = utils.data_iterator( \
              model.config.dataset_dir_iiit5k, model.config.dataset_dir_vgg,
              model.config.use_iiit5k, model.config.height, model.config.window_size, \
              model.config.stride, model.config.max_timestep, model.config.jittering_percent, \
              1, model.config.batch_size, \
              False, True, model.config.test_size, \
              model.config.visualize, model.config.visualize_dir)

        for step_test, (inputs_test, labels_sparse_test, timesteps_test,
            partition_test, epoch_test) in enumerate(iterator_test):

          feed_test = {model.inputs_placeholder: inputs_test,
                       model.labels_placeholder: labels_sparse_test,
                       model.timesteps_placeholder: timesteps_test,
                       model.dropout_placeholder: 0,
                       model.partition_placeholder: partition_test}

          ret_test = session.run([model.loss, model.dists, model.pred, model.groundtruth],
              feed_dict=feed_test)
          losses_test.append(ret_test[0])
          dists_test = np.concatenate((dists_test, ret_test[1]))

          if model.config.print_pred:
            pred = utils.indices2d2words(ret_test[2])
            groundtruth = utils.indices2d2words(ret_test[3])
            for i in range(len(pred)):
              print pred[i], groundtruth[i]

          # visualize the STN results
        #   if model.config.visualize and step_test < 1:
        #     utils.save_imgs(inputs_test.reshape(-1, model.config.height,
        #         model.config.window_size),
        #         model.config.visualize_dir, 'original'+str(step_test)+'-')
        #     utils.save_imgs(ret_test[4], model.config.visualize_dir,
        #         'trans'+str(step_test)+'-')

        losses_test = np.array(losses_test)
        cur_loss = np.mean(losses_test[np.isfinite(losses_test)])
        cur_dist = np.mean(dists_test)
        cur_char_accuracy = utils.get_char_accuracy(ret_test[3], ret_test[1])
        stats = np.bincount(dists_test.astype(int))

        logger.info('<-------------------->')
        logger.info('average test loss: %f (#batches = %d)',
            cur_loss, len(losses_test))
        logger.info('average edit distance: %f (#batches = %d)',
            cur_dist, len(losses_test))
        logger.info('character error: %f', cur_char_accuracy)
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
        logger.info('training loss in epoch %d, step %d: %f', cur_epoch, step_train,
            np.mean(losses_train[step_epoch:]))
        step_epoch = step_train
        cur_epoch = epoch_train

      feed_train = {model.inputs_placeholder: inputs_train,
                    model.labels_placeholder: labels_sparse_train,
                    model.timesteps_placeholder: timesteps_train,
                    model.dropout_placeholder: 1,
                    model.partition_placeholder: partition_train}

      ret_train = session.run([model.train_op, model.loss],
          feed_dict=feed_train)
      losses_train.append(ret_train[1])
      logger.info('epoch %d, step %d: training loss = %f', epoch_train, step_train,
        ret_train[1])

if __name__ == '__main__':
  main()
