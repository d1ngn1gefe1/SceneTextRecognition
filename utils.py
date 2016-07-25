import numpy as np
import tensorflow as tf
import math
import cv2
import logging
import os
from random import randint

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
  """Convert a character into an index (case insentitive)
    index 0 - 9: '0' - '9'
    index 10 - 35: 'A' & 'a' - 'Z' & 'z'

    ord('0') = 48, ord('9') = 57,
    ord('A') = 65, ord('Z') = 90, ord('a') = 97, ord('z') = 122
  """
  if ord(char) >= ord('0') and ord(char) <= ord('9'):
    return ord(char)-ord('0')
  elif ord(char) >= ord('A') and ord(char) <= ord('Z'):
    return ord(char)-ord('A')+10
  elif ord(char) >= ord('a') and ord(char) <= ord('z'):
    return ord(char)-ord('a')+10
  else:
    print 'char2index: invalid input'
    return -1

def index2char(index):
  if index >= 0 and index <= 9:
    return chr(ord('0')+index)
  elif index >= 10 and index <= 35:
    return chr(ord('A')+index-10)
  elif index == 36:
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
    embed_size, jittering):
  num_examples = imgs.shape[0]
  max_time = imgs.shape[1]
  height = imgs.shape[2]-10
  window_size = imgs.shape[3]-10
  depth = imgs.shape[4]
  num_steps = int(math.ceil(num_examples*num_epochs/batch_size))

  max_words_length = 0
  for word_embed in words_embed:
    word_length = word_embed.shape[0]
    if max_words_length < word_length:
      max_words_length = word_length

  #logger.info('max time: %d', max_time)
  #logger.info('max words length: %d', max_words_length)

  inputs = np.zeros((batch_size, max_time, height, window_size, depth))
  sequence_length = np.zeros(batch_size, dtype='int32') # number of windows
  outputs_mask = np.zeros((max_time, batch_size, embed_size))

  for i in range(num_steps):
    startIdx = i*batch_size%num_examples
    endIdx = (i+1)*batch_size%num_examples

    rand1 = randint(0, jittering)
    rand2 = randint(0, jittering)

    if jittering == 0:
      rand1 = 5
      rand2 = 5

    if startIdx < endIdx:
      inputs = imgs[startIdx:endIdx, :, rand1:rand1+height,
          rand2:rand2+window_size, :]
      sequence_length = time[startIdx:endIdx] # number of windows
      labels = words_embed[startIdx:endIdx]
    elif endIdx == 0:
      inputs = imgs[startIdx:, :, rand1:rand1+height,
          rand2:rand2+window_size, :]
      sequence_length = time[startIdx:]
      labels = words_embed[startIdx:]
    else:
      inputs = np.concatenate((imgs[startIdx:, :, rand1:rand1+height,
          rand2:rand2+window_size, :], imgs[:endIdx, :, rand1:rand1+height,
          rand2:rand2+window_size, :]))
      sequence_length = np.concatenate((time[startIdx:], time[:endIdx]))
      labels = words_embed[startIdx:]+words_embed[:endIdx]

    labels_sparse = dense2sparse(labels)

    epoch = i*batch_size/num_examples

    outputs_mask.fill(0)
    for j, length in enumerate(sequence_length):
      outputs_mask[length:, j, :] = np.nan

    yield (inputs, labels_sparse, sequence_length, outputs_mask, epoch)

def data_iterator_char(char_imgs, chars_embed, num_epochs, batch_size,
    embed_size, jittering_size, is_test):
  num_chars = char_imgs.shape[0]
  height = char_imgs.shape[1]-jittering_size
  window_size = char_imgs.shape[2]-jittering_size
  depth = char_imgs.shape[3]
  num_steps = int(math.ceil(num_chars*num_epochs/batch_size))

  for i in range(num_steps):
    startIdx = i*batch_size%num_chars
    endIdx = (i+1)*batch_size%num_chars

    inputs = np.zeros((batch_size, height, window_size, depth), dtype=np.uint8)
    labels = np.zeros((batch_size, embed_size), dtype=np.float32)

    if is_test:
      rand1 = int(jittering_size/2)
      rand2 = int(jittering_size/2)
    else:
      rand1 = randint(0, jittering_size)
      rand2 = randint(0, jittering_size)


    if startIdx < endIdx:
      inputs = char_imgs[startIdx:endIdx, rand1:rand1+height,
          rand2:rand2+window_size, :]
      labels[np.arange(0, batch_size), chars_embed[startIdx:endIdx]] = 1
    elif endIdx == 0:
      inputs = char_imgs[startIdx:, rand1:rand1+height,
          rand2:rand2+window_size, :]
      labels[np.arange(0, batch_size), chars_embed[startIdx:]] = 1
    else:
      inputs = np.concatenate((char_imgs[startIdx:, rand1:rand1+height,
          rand2:rand2+window_size, :], char_imgs[:endIdx, rand1:rand1+height,
          rand2:rand2+window_size, :]))
      labels[np.arange(0, batch_size), np.concatenate((chars_embed[startIdx:],
          chars_embed[:endIdx]))] = 1

    epoch = i*batch_size/num_chars

    yield (inputs, labels, epoch)

def save_imgs(imgs, dir, name):
  if not os.path.exists(dir):
    os.makedirs(dir)

  for i, img in enumerate(imgs):
    cv2.imwrite(dir+name+str(i)+'.jpg', img)
