import numpy as np
import tensorflow as tf
import math
import cv2
import logging

np.set_printoptions(threshold=np.nan)

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

def char2index(char):
  """Convert a character into an index
    index 0 - 9: '0' - '9'
    index 10 - 35: 'A' - 'Z'
    index 36 - 61: 'a' - 'z'

    ord('0') = 48, ord('9') = 57,
    ord('A') = 65, ord('Z') = 90, ord('a') = 97, ord('z') = 122
  """
  if ord(char) >= ord('0') and ord(char) <= ord('9'):
    return ord(char)-ord('0')
  elif ord(char) >= ord('A') and ord(char) <= ord('Z'):
    return ord(char)-ord('A')+10
  elif ord(char) >= ord('a') and ord(char) <= ord('z'):
    return ord(char)-ord('a')+36
  else:
    print 'char2index: invalid input'
    return -1

def index2char(index):
  if index >= 0 and index <= 9:
    return chr(ord('0')+index)
  elif index >= 10 and index <= 35:
    return chr(ord('A')+index-10)
  elif index >= 36 and index <= 61:
    return chr(ord('a')+index-36)
  elif index == 62:
    return '_'
  else:
    print 'index2char: invalid input'
    return '?'

def dense2sparse(labels):
  x_ix = []
  x_val = []

  for b, label in enumerate(labels):
    for t, val in enumerate(label):
      if t < label.shape[0]:
        x_ix.append([b, t])
        x_val.append(val)

  x_shape = [len(labels), np.asarray(x_ix).max(0)[1]+1]

  return (x_ix, x_val, x_shape)

def indices2word(indices):
  word = ''
  for index in indices:
    word += index2char(index)
  return word

def data_iterator(imgs, words_embed, time, num_epochs, batch_size, max_time,
    embed_size):
  num_examples = imgs.shape[0]
  max_time = imgs.shape[1]
  height = imgs.shape[2]
  window_size = imgs.shape[3]
  depth = imgs.shape[4]
  num_steps = int(math.ceil(num_examples*num_epochs/batch_size))

  max_words_length = 0
  for word_embed in words_embed:
    word_length = word_embed.shape[0]
    if max_words_length < word_length:
      max_words_length = word_length

  logger.info('max time: %d', max_time)
  logger.info('max words length: %d', max_words_length)

  inputs = np.zeros((batch_size, max_time, height, window_size, depth))
  sequence_length = np.zeros(batch_size, dtype='int32') # number of windows
  labels = []

  for i in range(num_steps):
    labels = []

    startIdx = i*batch_size%num_examples
    endIdx = (i+1)*batch_size%num_examples

    if startIdx < endIdx:
      inputs = imgs[startIdx:endIdx]
      sequence_length = time[startIdx:endIdx]
      labels = words_embed[startIdx:endIdx]
    elif endIdx == 0:
      inputs = imgs[startIdx:]
      sequence_length = time[startIdx:]
      labels = words_embed[startIdx:]
    else:
      inputs = np.concatenate((imgs[startIdx:], imgs[:endIdx]))
      sequence_length = np.concatenate((time[startIdx:], time[:endIdx]))
      labels = words_embed[startIdx:]+words_embed[:endIdx]

    labels_sparse = dense2sparse(labels)

    epoch = i*batch_size/num_examples

    outputs_mask = np.zeros((max_time, batch_size, embed_size))
    for j, length in enumerate(sequence_length):
      outputs_mask[length:, j, :] = np.nan

    # print '\n\n'
    # print i
    # for j in range(inputs.shape[0]):
    #   print inputs[j, :sequence_length[j]].shape
    #   print sequence_length[j], indices2word(labels[j])
    #   for k, img in enumerate(inputs[j, :sequence_length[j]]):
    #     cv2.imwrite('/home/local/ANT/zelunluo/SceneTextRecognition/imgs/'
    #         +str(i)+'_'+str(j)+'_'+str(k)+'.jpg', img)

    yield (inputs, labels_sparse, sequence_length, outputs_mask, epoch)

def variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape,
      initializer=tf.truncated_normal_initializer(stddev=stddev))

  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

  return var
