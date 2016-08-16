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

def indices2d2words(indices2d):
  words = []
  for indices in indices2d:
    words.append(indices2word(indices))
  return words

def data_iterator_vgg(dataset_dir_vgg, height, window_size, depth, embed_size, \
    stride, max_time, num_epochs, batch_size, is_train, debug, debug_size, \
    jittering_size):
  count = 0
  dataset = 'annotation_train.txt' if is_train else 'annotation_val.txt'

  with open(dataset_dir_vgg+dataset) as f:
    lines = f.readlines()
    num_examples = len(lines)
    if debug and num_examples > debug_size:
      lines = lines[:debug_size]
      num_examples = len(lines)
    num_steps = int(math.ceil(num_examples*num_epochs/batch_size))
    print 'reading '+dataset, num_examples, num_steps

    for i in range(num_steps):
      imgs = np.zeros((batch_size, max_time, height, window_size, depth),
          dtype=np.uint8)
      time = np.zeros(batch_size, dtype=np.uint8)
      words_embed = []

      j = 0
      while True:
        line_index = count%num_examples
        line = lines[line_index]
        strings = line.split(' ')
        filename = strings[0][1:]
        word = filename.split('_')[1]
        count += 1

        img = cv2.imread(dataset_dir_vgg+filename)
        h = height
        scale = height/float(img.shape[0])
        w = int(round(scale*img.shape[1]))
        img = cv2.resize(img, (w, h))

        cur_time = int(math.ceil((w+window_size)/float(stride)-1))
        if cur_time > max_time or cur_time <= len(word):
          continue
        time[j] = cur_time

        word_embed = np.zeros(len(word), dtype=np.uint8)
        for k, char in enumerate(word):
          word_embed[k] = char2index(char)
        words_embed.append(word_embed)

        for l in range(cur_time):
          start1 = max((l+1)*stride-window_size, 0)
          end1 = min((l+1)*stride, w)
          start2 = max(-((l+1)*stride-window_size), 0)
          end2 = min(start2+end1-start1, window_size)

          imgs[j, l, :, start2:end2, :] = img[:, start1:end1, :]
          if start2 != 0:
            imgs[j, l, :, :start2, :] = imgs[j, l, :, start2][:, np.newaxis, :]
          if end2 != window_size:
            imgs[j, l, :, end2:, :] = imgs[j, l, :, end2-1][:, np.newaxis, :]

        j += 1
        if j == batch_size:
          break

      inputs = np.swapaxes(imgs, 0, 1)
      inputs = [inputs[:time[m], m] for m in range(batch_size)]
      inputs = np.concatenate(inputs, 0)

      labels_sparse = dense2sparse(words_embed)

      partition = np.arange(0, batch_size)
      partition = np.repeat(partition, time)

      epoch = i*batch_size/num_examples

      yield (inputs, labels_sparse, time, partition, epoch)

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
    labels = np.zeros(batch_size, dtype=np.float32)

    if is_test:
      # crop window at the center
      rand1 = int(jittering_size/2)
      rand2 = int(jittering_size/2)
    else:
      rand1 = randint(0, jittering_size)
      rand2 = randint(0, jittering_size)

    if startIdx < endIdx:
      inputs = char_imgs[startIdx:endIdx, rand1:rand1+height,
          rand2:rand2+window_size, :]
      labels = chars_embed[startIdx:endIdx]
    elif endIdx == 0:
      inputs = char_imgs[startIdx:, rand1:rand1+height,
          rand2:rand2+window_size, :]
      labels = chars_embed[startIdx:endIdx]
    else:
      inputs = np.concatenate((char_imgs[startIdx:, rand1:rand1+height,
          rand2:rand2+window_size, :], char_imgs[:endIdx, rand1:rand1+height,
          rand2:rand2+window_size, :]))
      labels = np.concatenate((chars_embed[startIdx:], chars_embed[:endIdx]))

    epoch = i*batch_size/num_chars

    yield (inputs, labels, epoch)

def save_imgs(imgs, dir, name):
  if not os.path.exists(dir):
    os.makedirs(dir)

  for i, img in enumerate(imgs):
    cv2.imwrite(dir+name+str(i)+'.jpg', img)
