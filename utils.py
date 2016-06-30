import numpy as np
import tensorflow as tf
import math

np.set_printoptions(threshold=np.nan)

'''
    convert a character into an index
    index 1 - 10: '0' - '9'
    index 11 - 36: 'A' - 'Z'
    index 37 - 62: 'a' - 'z'
    ord('0') = 48, ord('9') = 57,
    ord('A') = 65, ord('Z') = 90, ord('a') = 97, ord('z') = 122
'''
def char2index(char):
  if ord(char) >= ord('0') and ord(char) <= ord('9'):
    return ord(char)-ord('0')+1
  elif ord(char) >= ord('A') and ord(char) <= ord('Z'):
    return ord(char)-ord('A')+11
  elif ord(char) >= ord('a') and ord(char) <= ord('z'):
    return ord(char)-ord('a')+37
  else:
    print 'char2index: invalid input'
    return -1

def index2char(index):
  index -= 1
  if index >= 0 and index <= 9:
    return chr(ord('0')+index)
  elif index >= 10 and index <= 35:
    return chr(ord('A')+index-10)
  elif index >= 36 and index <= 61:
    return chr(ord('a')+index-36)
  else:
    print 'index2char: invalid input'
    return '?'

def data_iterator(imgs, words_embed, time, words_length, \
    num_epochs, batch_size, max_time, max_words_length):

  num_examples = imgs.shape[0]
  max_time = imgs.shape[1]
  height = imgs.shape[2]
  window_size = imgs.shape[3]
  depth = imgs.shape[4]
  num_steps = int(math.ceil(num_examples*num_epochs/batch_size))

  inputs = np.zeros((max_time, batch_size, height, window_size, depth))
  labels_indices = np.zeros([batch_size*max_words_length, 2], dtype='int32')
  labels_values = np.zeros(batch_size*max_words_length, dtype='int32')
  labels_shape = np.array([batch_size, max_words_length], dtype='int32')
  sequence_length = np.zeros(batch_size, dtype='int32')

  for i in range(num_steps):
    startIdx = i*batch_size%num_examples
    inputs.fill(0)
    labels_indices.fill(0)
    labels_values.fill(0)
    sequence_length.fill(0)
    N = 0

    for j in range(batch_size):
      idx = (startIdx+j)%num_examples

      inputs[:, j, :, :, :] = imgs[idx]
      sequence_length[j] = time[idx]

      for k in range(words_length[idx]):
        labels_indices[N, 0] = j
        labels_indices[N, 1] = k
        labels_values[N] = words_embed[idx, k]
        N += 1

      # sanity check
    #   if words_length[idx] > time[idx]:
    #     print 'Not enough time for target transition sequence!!!'

    yield (inputs, labels_indices[:N], labels_values[:N], labels_shape, \
        sequence_length)

def variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, \
        dtype=tf.float32)
  return var

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
  var = variable_on_cpu(name, shape, \
      tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))

  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

  return var
