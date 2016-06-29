import numpy as np
import tensorflow as tf
import math

np.set_printoptions(threshold=np.nan)

'''
    convert a character into an index
    index 0 - 9: '0' - '9'
    index 10 - 35: 'A' - 'Z'
    index 36 - 61: 'a' - 'z'
    ord('0') = 48, ord('9') = 57,
    ord('A') = 65, ord('Z') = 90, ord('a') = 97, ord('z') = 122
'''
def char2index(char):
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
  else:
    print 'index2char: invalid input'
    return '?'

def data_iterator(imgs, words, imgs_length, words_length, batch_size,
    window_size, num_epochs, max_time_encoder, max_time_decoder):

  print words.shape

  num_examples = words.shape[0]
  num_steps = int(math.ceil(num_examples*num_epochs/batch_size))
  height = imgs.shape[1]
  depth = imgs.shape[2]
  print num_examples, num_steps, imgs.shape, batch_size, num_epochs, depth

  for i in range(num_steps):
    startIdx = i*batch_size%num_examples
    #endIdx = (i+1)*batch_size%num_examples

    inputs_encoder = np.zeros((batch_size, max_time_encoder, height, \
        window_size, depth))
    inputs_decoder = np.zeros((batch_size, max_time_decoder, 62))
    labels = np.zeros((batch_size, max_time_decoder, 62))
    inputs_length = np.zeros(batch_size)
    labels_mask = np.zeros((batch_size, max_time_decoder))

    for j in range(batch_size):
      idx = (startIdx+j)%num_examples
      startColIdx = np.sum(imgs_length[:idx])
      endColIdx = startColIdx+imgs_length[idx]

      img = imgs[startColIdx:endColIdx, :, :]

      time_encoder = imgs_length[idx]/window_size

      for k in range(time_encoder):
        startWindowIdx = k*window_size
        endWindowIdx = min((k+1)*window_size, img.shape[0])
        window = img[startWindowIdx:endWindowIdx, :, :]
        inputs_encoder[j, k, :] = window.reshape((height, window_size, depth))

      inputs_length[j] = time_encoder

      word = words[idx]
      #print word
      for l, char in enumerate(word):
        index = char2index(char)
        if l != len(word)-1:
          inputs_decoder[j, l, index] += 1
        if l != 0:
          labels[j, l-1, index] += 1

      labels_mask[j, :words_length[idx]] = 1

    yield (inputs_encoder, inputs_decoder, labels, inputs_length, labels_mask)

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
      tf.truncated_normal_initializer(stddev=stddev))

  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

  return var
