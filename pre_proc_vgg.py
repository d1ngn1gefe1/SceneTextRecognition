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
  time_train = []
  imgs_test = []
  words_embed_test = []
  time_test = []

  if visualize and not os.path.exists(visualize_dir):
    os.makedirs(visualize_dir)

  with open(dataset_dir_vgg+'annotation_train.txt') as f:
    lines = f.readlines()
    num_examples_train = len(lines)
    print 'reading annotation_train.txt', num_examples_train

    for i, line in enumerate(lines):
      if (i % 1000 == 0):
        print 'image', i
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
      if cur_time > 20 or cur_time <= len(word):
        continue
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

        if visualize and i < 15:
          cv2.imwrite(visualize_dir+str(i)+'_'+str(j)+'_train.jpg', img_windows[j])

      imgs_train.append(img_windows)
      time_train.append(cur_time)
      if max_time < cur_time:
        max_time = cur_time
      if len(time_train) >= 200000:
        break

  with open(dataset_dir_vgg+'annotation_val.txt') as f:
    lines = f.readlines()
    num_examples_test = len(lines)
    print 'reading annotation_val.txt', num_examples_test

    for i, line in enumerate(lines):
      if (i % 1000 == 0):
        print 'image', i
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
      if cur_time > 20 or cur_time <= len(word):
        continue
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

        if visualize and i < 15:
          cv2.imwrite(visualize_dir+str(i)+'_'+str(j)+'_test.jpg', img_windows[j])

      imgs_test.append(img_windows)
      time_test.append(cur_time)
      if max_time < cur_time:
        max_time = cur_time
      if len(time_test) >= 20000:
        break

    num_examples_train = len(imgs_train)
    num_examples_test = len(imgs_test)
    print 'max_time', max_time
    print 'num_examples_train', num_examples_train
    print 'num_examples_test', num_examples_test

    imgs_train_np = np.zeros((num_examples_train, max_time, height, window_size, depth),
        dtype=np.uint8)
    time_train_np = np.zeros(num_examples_train, dtype=np.uint8)
    for i in range(num_examples_train):
      imgs_train_np[i, :time_train[i], :, :, :] = imgs_train[i]
      time_train_np[i] = time_train[i]
    imgs_test_np = np.zeros((num_examples_test, max_time, height, window_size, depth),
        dtype=np.uint8)
    time_test_np = np.zeros(num_examples_test, dtype=np.uint8)
    for i in range(num_examples_test):
      imgs_test_np[i, :time_test[i], :, :, :] = imgs_test[i]
      time_test_np[i] = time_test[i]

    filename = os.path.join(dataset_dir_vgg, 'train.hdf5')
    print 'Writing ' + filename
    with h5py.File(filename, 'w') as hf:
      dt = h5py.special_dtype(vlen=np.dtype('uint8'))
      hf.create_dataset('imgs', data=imgs_train_np)
      hf.create_dataset('words_embed', data=words_embed_train, dtype=dt)
      hf.create_dataset('time', data=time_train_np)
    filename = os.path.join(dataset_dir_vgg, 'test.hdf5')
    print 'Writing ' + filename
    with h5py.File(filename, 'w') as hf:
      dt = h5py.special_dtype(vlen=np.dtype('uint8'))
      hf.create_dataset('imgs', data=imgs_test_np)
      hf.create_dataset('words_embed', data=words_embed_test, dtype=dt)
      hf.create_dataset('time', data=time_test_np)

if __name__ == '__main__':
  main()
