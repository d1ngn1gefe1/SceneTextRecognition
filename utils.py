import cv2
import h5py
import logging
import math
import numpy as np
import os
from random import randint
import scipy.io


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


# return: an integer
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
    logger.warning('char2index: invalid input')
    return -1


# return: a string
def index2char(index):
  if index >= 0 and index <= 9:
    return chr(ord('0')+index)
  elif index >= 10 and index <= 35:
    return chr(ord('A')+index-10)
  elif index == 36:
    return '_'
  else:
    logger.warning('index2char: invalid input')
    return '?'


# return: an numpy array
def word2indices(word):
  word_embed = np.zeros(len(word), dtype=np.uint8)
  for i, char in enumerate(word):
    word_embed[i] = char2index(char)
  return word_embed


# return: a string
def indices2word(indices):
  word = ''
  for index in indices:
    word += index2char(index)
  return word


# return: a list of strings
def indices2d2words(indices2d):
  words = []
  for indices in indices2d:
    words.append(indices2word(indices))
  return words


def get_char_accuracy(groundtruth, dists):
  return np.sum(dists)/float(np.sum(groundtruth != 36))


def save_imgs(imgs, dir, name):
  if not os.path.exists(dir):
    os.makedirs(dir)

  for i, img in enumerate(imgs):
    cv2.imwrite(dir+name+str(i)+'.jpg', img)


def data_iterator(dataset_dir_iiit5k, dataset_dir_vgg, use_iiit5k,
    height, window_size, stride, max_timestep, jittering_percent, \
    num_epochs, batch_size, \
    is_train, debug, debug_size, visualize, visualize_dir):

  if use_iiit5k:
    dataset_dir = dataset_dir_iiit5k
    train_dict = scipy.io.loadmat(dataset_dir+'trainCharBound.mat')
    train_data = np.squeeze(train_dict['trainCharBound'])
    test_dict = scipy.io.loadmat(dataset_dir+'testCharBound.mat')
    test_data = np.squeeze(test_dict['testCharBound'])
    if is_train:
      data = np.concatenate((train_data, test_data[:2000]))
      dataset = dataset_dir+'trainCharBound.mat'
    else:
      data = test_data[2000:]
      dataset = dataset_dir+'testCharBound.mat'
  else:
    dataset_dir = dataset_dir_vgg
    dataset_name = 'annotation_train.txt' if is_train else 'annotation_val.txt'
    with open(dataset_dir+dataset_name) as f:
      data = f.readlines()
      dataset = dataset_dir+dataset_name

  num_examples = min(debug_size, len(data)) if debug else len(data)
  num_steps = int(math.ceil(num_examples*num_epochs/batch_size))
  print 'reading '+dataset, num_examples, num_steps

  if visualize and not os.path.exists(visualize_dir):
    os.makedirs(visualize_dir)

  for i in range(num_steps):
    imgs = []
    timesteps = []
    word_embeds = []

    num_bad_examples = 0
    j = 0
    while True:
      index = (i*batch_size+j+num_bad_examples)%num_examples
      if use_iiit5k:
        img_path = data[index][0][0]
        word = str(data[index][1][0])
      else:
        img_path = data[index].split(' ')[0][1:]
        word = img_path.split('_')[1]

      # load image
      img = cv2.imread(dataset_dir+img_path, cv2.IMREAD_GRAYSCALE)
      if img is None:
        logger.warning('image does not exist: %s', dataset_dir+filename)
        num_bad_examples += 1
        continue
      if is_train and visualize and i < 5 and j < 5:
        cv2.imwrite(visualize_dir+str(i)+'_'+str(j)+'.jpg', img)
      h_jittering = int(round(height*(1+jittering_percent)))
      scale = h_jittering/float(img.shape[0])
      w_jittering = int(round(scale*img.shape[1]))
      img = cv2.resize(img, (w_jittering, h_jittering))
      jittering_size = h_jittering-height
      window_size_jittering = window_size+jittering_size
      # print 'img', h_jittering, w_jittering, jittering_size

      # timestep
      # first window: [(0-2)*stride, (0-2)*stride+w)
      # last window: [(i-2)*stride, (i-2)*stride+w) such that
      #   (i-2)*stride >= w_jittering-2*stride and
      #   (i-3)*stride < w_jittering-2*stride
      # cur_timestep = i+1
      cur_timestep = int(math.ceil(w_jittering/float(stride)))+1
      if cur_timestep > max_timestep or cur_timestep <= len(word):
        #logger.warning('timestep not valid: %d, %d, %d', cur_timestep, max_timestep, len(word))
        num_bad_examples += 1
        continue
      timesteps.append(cur_timestep)
      # print 'timestep', cur_timestep, max_timestep

      # word_embed
      word_embed = word2indices(word)
      word_embeds.append(word_embed)
      # print 'word_embed', word, word_embed

      # img
      for k in range(cur_timestep):
        # crop an image for current timestep
        start_src = max((k-2)*stride, 0)
        end_src = min((k-2)*stride+window_size_jittering, w_jittering)
        start_des = max(-(k-2)*stride, 0)
        end_des = min(start_des+end_src-start_src, window_size_jittering)
        #print start_src, end_src, start_des, end_des
        img_crop = np.zeros((h_jittering, window_size_jittering), dtype=np.uint8)
        img_crop[:, start_des:end_des] = img[:, start_src:end_src]

        # fill empty space at start and end
        if start_des != 0:
          img_crop[:, :start_des] = img_crop[:, start_des][:, np.newaxis]
        if end_des != window_size:
          img_crop[:, end_des:] = img_crop[:, end_des-1][:, np.newaxis]

        # jittering
        if is_train:
          rand_x = randint(0, jittering_size)
          rand_y = randint(0, jittering_size)
        else:
          # crop window at the center
          rand_x = int(jittering_size/2)
          rand_y = int(jittering_size/2)
        img_crop = img_crop[rand_y:rand_y+height, rand_x:rand_x+window_size]
        imgs.append(img_crop[:, :, np.newaxis])
        # print img_crop.shape

        if is_train and visualize and i < 5 and j < 5:
          cv2.imwrite(visualize_dir+str(i)+'_'+str(j)+'_'+str(k)+'.jpg', img_crop)

      j += 1
      if j == batch_size:
        break

    inputs = np.stack(imgs)
    labels_sparse = dense2sparse(word_embeds)
    partition = np.arange(0, batch_size)
    partition = np.repeat(partition, timesteps)
    epoch = i*batch_size/num_examples

    # if num_bad_examples > 0:
    #   print 'num_bad_examples: ', num_bad_examples, '/', batch_size
    yield (inputs, labels_sparse, timesteps, partition, epoch)


def data_iterator_baseline(dataset_dir_iiit5k, dataset_dir_vgg, use_iiit5k,
    height, window_size, stride, max_timestep, jittering_percent, \
    num_epochs, batch_size, \
    is_train, debug, debug_size, visualize, visualize_dir):

  if use_iiit5k:
    dataset_dir = dataset_dir_iiit5k
    train_dict = scipy.io.loadmat(dataset_dir+'trainCharBound.mat')
    train_data = np.squeeze(train_dict['trainCharBound'])
    test_dict = scipy.io.loadmat(dataset_dir+'testCharBound.mat')
    test_data = np.squeeze(test_dict['testCharBound'])
    if is_train:
      data = np.concatenate((train_data, test_data[:2000]))
      dataset = dataset_dir+'trainCharBound.mat'
    else:
      data = test_data[2000:]
      dataset = dataset_dir+'testCharBound.mat'
  else:
    dataset_dir = dataset_dir_vgg
    dataset_name = 'annotation_train.txt' if is_train else 'annotation_val.txt'
    with open(dataset_dir+dataset_name) as f:
      data = f.readlines()
      dataset = dataset_dir+dataset_name

  num_examples = min(debug_size, len(data)) if debug else len(data)
  num_steps = int(math.ceil(num_examples*num_epochs/batch_size))
  print 'reading '+dataset, num_examples, num_steps

  for i in range(num_steps):
    imgs = []
    timesteps = []
    word_embeds = []

    num_bad_examples = 0
    j = 0
    while True:
      index = (i*batch_size+j+num_bad_examples)%num_examples
      if use_iiit5k:
        img_path = data[index][0][0]
        word = str(data[index][1][0])
      else:
        img_path = data[index].split(' ')[0][1:]
        word = img_path.split('_')[1]

      # load image
      img = cv2.imread(dataset_dir+img_path, cv2.IMREAD_GRAYSCALE)
      if img is None:
        logger.warning('image does not exist: %s', dataset_dir+filename)
        num_bad_examples += 1
        continue
      h = height
      scale = h/float(img.shape[0])
      w = int(round(scale*img.shape[1]))
      img = cv2.resize(img, (w, h))
      # print 'img', h, w

      # timestep
      cur_timestep = w
      if cur_timestep > max_timestep or cur_timestep <= len(word):
        #logger.warning('timestep not valid: %d, %d, %d', cur_timestep, max_timestep, len(word))
        num_bad_examples += 1
        continue
      timesteps.append(cur_timestep)
      # print 'timestep', cur_timestep, max_timestep

      # word_embed
      word_embed = word2indices(word)
      word_embeds.append(word_embed)
      # print 'word_embed', word, word_embed

      # img
      imgs.append(img)

      j += 1
      if j == batch_size:
        break

    inputs = np.concatenate(imgs, axis=1)
    inputs = np.swapaxes(inputs, 0, 1)
    labels_sparse = dense2sparse(word_embeds)
    partition = np.arange(0, batch_size)
    partition = np.repeat(partition, timesteps)
    epoch = i*batch_size/num_examples

    # if num_bad_examples > 0:
    #   print 'num_bad_examples: ', num_bad_examples, '/', batch_size
    yield (inputs, labels_sparse, timesteps, partition, epoch)


def data_iterator_char(dataset_dir_iiit5k, dataset_dir_vgg, use_iiit5k, \
    height, window_size, num_epochs, batch_size, embed_size, \
    jittering_percent, is_train, visualize, visualize_dir):

  if use_iiit5k:
    train_dict = scipy.io.loadmat(dataset_dir_iiit5k+'trainCharBound.mat')
    train_data = np.squeeze(train_dict['trainCharBound'])
    test_dict = scipy.io.loadmat(dataset_dir_iiit5k+'testCharBound.mat')
    test_data = np.squeeze(test_dict['testCharBound'])
    if is_train:
      data = np.concatenate((train_data, test_data[:2000]))
      dataset = dataset_dir_iiit5k+'trainCharBound.mat'
    else:
      data = test_data[2000:]
      dataset = dataset_dir_iiit5k+'testCharBound.mat'
  else:
    # not working
    f = h5py.File(dataset_dir_vgg+'case-insensitive-train.mat')
    data = f.get('#refs#')
    for i, key in enumerate(data.iterkeys()):
      img = np.array(data.get(key), dtype=np.uint8)
      cv2.imshow('image', img)
      cv2.waitKey(0)
    return

  num_examples = len(data)
  num_steps = int(math.ceil(num_examples*num_epochs/batch_size))
  print 'reading '+dataset, num_examples, num_steps

  if visualize and not os.path.exists(visualize_dir):
    os.makedirs(visualize_dir)

  for i in range(num_steps):
    char_imgs = []
    char_embeds = []

    num_bad_examples = 0
    for j in range(batch_size):
      index = (i*batch_size+j+num_bad_examples)%num_examples
      img_path = data[index][0][0]
      word = str(data[index][1][0])
      bboxes = data[index][2]

      img = cv2.imread(dataset_dir_iiit5k+img_path, cv2.IMREAD_GRAYSCALE)
      if img is None:
        logger.warning('image does not exist: %s', dataset_dir_iiit5k+filename)
        num_bad_examples += 1
        continue
      h_jittering = int(round(height*(1+jittering_percent)))
      scale = h_jittering/float(img.shape[0])
      w_jittering = int(round(scale*img.shape[1]))
      img = cv2.resize(img, (w_jittering, h_jittering))
      jittering_size = h_jittering-height
      window_size_jittering = window_size+jittering_size

      for k, bbox in enumerate(bboxes):
        # crop a character
        x_c = int(round(scale*(bbox[0]+bbox[2]/2)))
        start_src = max(x_c-window_size_jittering/2, 0)
        end_src = min(x_c+(window_size_jittering+1)/2, w_jittering)
        start_des = max(-(x_c-window_size_jittering/2), 0)
        end_des = min(start_des+end_src-start_src, window_size_jittering)
        char_img = np.zeros((h_jittering, window_size_jittering), dtype=np.uint8)
        char_img[:, start_des:end_des] = img[:, start_src:end_src]

        # fill empty space at start and end
        if start_des != 0:
          char_img[:, :start_des] = char_img[:, start_des][:, np.newaxis]
        if end_des != window_size_jittering:
          char_img[:, end_des:] = char_img[:, end_des-1][:, np.newaxis]

        # jittering
        if is_train:
          rand_x = randint(0, jittering_size)
          rand_y = randint(0, jittering_size)
        else:
          # crop window at the center
          rand_x = int(jittering_size/2)
          rand_y = int(jittering_size/2)
        char_img = char_img[rand_y:rand_y+height, rand_x:rand_x+window_size]

        # char_img
        char_imgs.append(char_img[:, :, np.newaxis])

        # char_embed
        char_embed = char2index(word[k])
        char_embeds.append(char_embed)

        if is_train and visualize and i < 5 and j < 5:
          cv2.imwrite(visualize_dir+str(i)+'_'+str(j)+'_'+str(k)+'_'+word[k]+'.jpg', char_img)

    inputs = np.stack(char_imgs)
    labels = np.array(char_embeds)
    epoch = i*batch_size/num_examples

    yield (inputs, labels, epoch)


def get_info(dataset_dir_iiit5k, dataset_dir_vgg, use_iiit5k, \
    height, window_size, stride, num_epochs, batch_size):

  if use_iiit5k:
    dataset_dir = dataset_dir_iiit5k
    train_dict = scipy.io.loadmat(dataset_dir+'trainCharBound.mat')
    train_data = np.squeeze(train_dict['trainCharBound'])
    test_dict = scipy.io.loadmat(dataset_dir+'testCharBound.mat')
    test_data = np.squeeze(test_dict['testCharBound'])
    data = np.concatenate((train_data, test_data))
  else:
    dataset_dir = dataset_dir_vgg
    with open(dataset_dir+'annotation_train.txt') as f:
      data1 = f.readlines()
    with open(dataset_dir+'annotation_val.txt') as f:
      data2 = f.readlines()
    data = data1+data2

  num_examples = len(data)
  print num_examples

  timesteps = []
  widths = []

  for i in range(num_examples):
    if use_iiit5k:
      img_path = data[i][0][0]
      word = str(data[i][1][0])
    else:
      img_path = data[i].split(' ')[0][1:]
      word = img_path.split('_')[1]

    img = cv2.imread(dataset_dir+img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
      logger.warning('image does not exist: %s', dataset_dir+filename)
      continue
    h = height
    scale = h/float(img.shape[0])
    w = int(round(scale*img.shape[1]))
    widths.append(w)

    cur_timestep = int(math.ceil(w/float(stride)))+1
    timesteps.append(cur_timestep)

  print max(timesteps), max(widths)
  print np.bincount(np.array(widths))
