import numpy as np
import tensorflow as tf
import math

np.set_printoptions(threshold=np.nan)

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
  """
  From index to actual character
  Args:
      index:

  Returns:

  """
  index -= 1
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

def dense2sparse(x, max_words_length):
  """
  Purposes: ?
  Args:
      x:
      max_words_length:

  Returns:
      x_ix: ?
      x_val: ?
      x_shape: ?
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    for time, val in enumerate(batch):
      x_ix.append([batch_i, time])
      x_val.append(val)
  x_shape = [len(x), max_words_length]

  return (x_ix, x_val, x_shape)

# def outputs2words(outputs):
#   """
#   Args:
#     outputs: max_time x batch_size x embed_size
#   """
#   words = []
#   for i in range(outputs.shape[1]):
#     indices = np.argmax(outputs[:, i, :], axis=1)


def data_iterator(imgs, words_embed, time, num_epochs, batch_size, max_time):
  """
  Purposes: ?
  Args:
      imgs:
      words_embed:
      time:
      num_epochs:
      batch_size:
      max_time: not used?

  Returns:
    inputs:
    labels_sparse:
     sequence_length:
  """
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

  #-- not needed?
  inputs = np.zeros((max_time, batch_size, height, window_size, depth))
  sequence_length = np.zeros(batch_size, dtype='int32')
  labels = [] # a list of numpy arrays
  #---

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

    labels_sparse = dense2sparse(labels, max_words_length)

    epoch = i*batch_size/num_examples
    yield (inputs, labels_sparse, sequence_length, epoch)

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
    var = tf.get_variable(name, shape, initializer=initializer,
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
  var = variable_on_cpu(name, shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))

  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

  return var
