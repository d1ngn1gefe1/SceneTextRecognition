import numpy as np
import scipy.io
import cv2
import glob
import os
import h5py
import json
import utils
import math

def main():
  """
  Read data in (VGG format), convert/process and save it to hdf5 format
  Returns:

  """
  with open('config.json', 'r') as json_file:
    json_data = json.load(json_file)
    dataset_dir_vgg = json_data['dataset_dir_vgg']
    height = json_data['height']+ \
        int(json_data['height']*json_data['jittering_percent'])
    window_size = json_data['window_size']+ \
        int(json_data['height']*json_data['jittering_percent'])
    depth = json_data['depth']
    embed_size = json_data['embed_size']
    stride = json_data['stride']
    visualize = json_data['visualize']
    visualize_dir = json_data['visualize_dir']

  max_time = 0
  imgs_train = []
  words_embed_train = []
  imgs_test = []
  words_embed_test = []

  if visualize and not os.path.exists(visualize_dir):
    os.makedirs(visualize_dir)

  with open(dataset_dir_vgg+'annotation_train.txt') as f:
    print 'reading annotation_train.txt'
    lines = f.readlines()
    num_examples_train = len(lines)
    time_train = np.zeros(num_examples_train, dtype=np.uint8)

    for i, line in enumerate(lines):
      strings = line.split(' ')
      filename = strings[0][1:]
      word = filename.split('_')[1]

      word_embed = np.zeros(len(word), dtype=np.uint8)
      for j, char in enumerate(word):
        word_embed[j] = utils.char2index(char)
      words_embed_train.append(word_embed)

      img = cv2.imread(dataset_dir_vgg+filename)
      h = height
      scale = height/float(img.shape[0])
      w = int(round(scale*img.shape[1]))
      img = cv2.resize(img, (w, h))

      if depth == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img[:, :, None]

      cur_time = int(math.ceil((w+window_size)/float(stride)-1))
      img_windows = np.zeros((cur_time, height, window_size, depth),
          dtype=np.uint8)

      for j in range(cur_time):
        start1 = max((j+1)*stride-window_size, 0)
        end1 = min((j+1)*stride, w)
        start2 = max(-((j+1)*stride-window_size), 0)
        end2 = min(start2+end1-start1, window_size)

        img_windows[j, :, start2:end2, :] = img[:, start1:end1, :]
        if start2 != 0:
          img_windows[j, :, :start2] = img_windows[j, :, start2][:, np.newaxis, :]
        if end2 != window_size:
          img_windows[j, :, end2:] = img_windows[j, :, end2-1][:, np.newaxis, :]

        if visualize and i < 50:
          cv2.imwrite(visualize_dir+str(i)+'_'+str(j)+'_train.jpg', img_windows[j])

      imgs_train.append(img_windows)
      time_train[i] = cur_time
      if max_time < cur_time:
        max_time = cur_time

  with open(dataset_dir_vgg+'annotation_val.txt') as f:
    print 'reading annotation_val.txt'
    lines = f.readlines()
    num_examples_test = len(lines)
    time_test = np.zeros(num_examples_test, dtype=np.uint8)

    for i, line in enumerate(lines):
      strings = line.split(' ')
      filename = strings[0][1:]
      word = filename.split('_')[1]

      word_embed = np.zeros(len(word), dtype=np.uint8)
      for j, char in enumerate(word):
        word_embed[j] = utils.char2index(char)
      words_embed_test.append(word_embed)

      img = cv2.imread(dataset_dir_vgg+filename)
      h = height
      scale = height/float(img.shape[0])
      w = int(round(scale*img.shape[1]))
      img = cv2.resize(img, (w, h))

      if depth == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img[:, :, None]

      cur_time = int(math.ceil((w+window_size)/float(stride)-1))
      img_windows = np.zeros((cur_time, height, window_size, depth),
          dtype=np.uint8)

      for j in range(cur_time):
        start1 = max((j+1)*stride-window_size, 0)
        end1 = min((j+1)*stride, w)
        start2 = max(-((j+1)*stride-window_size), 0)
        end2 = min(start2+end1-start1, window_size)

        img_windows[j, :, start2:end2, :] = img[:, start1:end1, :]
        if start2 != 0:
          img_windows[j, :, :start2] = img_windows[j, :, start2][:, np.newaxis, :]
        if end2 != window_size:
          img_windows[j, :, end2:] = img_windows[j, :, end2-1][:, np.newaxis, :]

        if visualize and i < 50:
          cv2.imwrite(visualize_dir+str(i)+'_'+str(j)+'_test.jpg', img_windows[j])

      imgs_test.append(img_windows)
      time_test[i] = cur_time
      if max_time < cur_time:
        max_time = cur_time

    print 'max_time', max_time
    print 'num_examples_train', num_examples_train
    print 'num_examples_test', num_examples_test

    imgs_train_np = np.zeros((num_examples_train, max_time, height, window_size, depth),
        dtype=np.uint8)
    for i in range(num_examples_train):
      imgs_train_np[i, :time_train[i], :, :, :] = imgs_train[i]
    imgs_test_np = np.zeros((num_examples_test, max_time, height, window_size, depth),
        dtype=np.uint8)
    for i in range(num_examples_test):
      imgs_test_np[i, :time_test[i], :, :, :] = imgs_test[i]

    filename = os.path.join(dataset_dir_vgg, 'train.hdf5')
    print 'Writing ' + filename
    with h5py.File(filename, 'w') as hf:
      dt = h5py.special_dtype(vlen=np.dtype('uint8'))
      hf.create_dataset('imgs_test', data=imgs_train_np)
      hf.create_dataset('words_embed_test', data=words_embed_train, dtype=dt)
      hf.create_dataset('time_test', data=time_train)
    filename = os.path.join(dataset_dir_vgg, 'test.hdf5')
    print 'Writing ' + filename
    with h5py.File(filename, 'w') as hf:
      dt = h5py.special_dtype(vlen=np.dtype('uint8'))
      hf.create_dataset('imgs_test', data=imgs_test_np)
      hf.create_dataset('words_embed_test', data=words_embed_test, dtype=dt)
      hf.create_dataset('time_test', data=time_test)

if __name__ == '__main__':
  main()
