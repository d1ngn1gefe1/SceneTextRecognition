import numpy as np
import scipy.io
import cv2
import glob
import os
import h5py
import json
import utils
import math

def load_and_process(dataset_dir, data, height, window_size, depth, stride):
  """
  Args:
      dataset_dir:
      data:
      height:
      window_size:
      depth:
      stride:

  Returns:
      imgs
      words_embeded
      time
  """
  num_examples = data.shape[0]

  imgs = []
  words_embed = []
  time = np.zeros(num_examples, dtype=np.uint8)

  num_rescale = 0
  scales = []
  scale_factor = 1.1

  for i in range(num_examples):
    img = cv2.imread(dataset_dir + data[i][0][0])
    h = height
    w = int(round(height*img.shape[1]/float(img.shape[0])))
    img = cv2.resize(img, (w, h))
    word = str(data[i][1][0])

    cur_time = int(math.ceil((w-window_size)/float(stride)+1))
    cur_time = max(cur_time, 1)
    word_length = len(word)

    # Not enough time for target transition sequence
    if word_length >= cur_time:
      num_rescale += 1
      scale = 1
      while word_length >= cur_time:
        w *= scale_factor
        w = int(w)
        scale *= scale_factor
        img = cv2.resize(img, (w, h))
        cur_time = int(math.ceil((w-window_size)/float(stride)+1))
        cur_time = max(cur_time, 1)
      scales.append(scale)

    img_windows = np.zeros((cur_time, height, window_size, depth))
    for j in range(cur_time):
      start = j*stride
      end = min(j*stride+window_size, w)
      img_windows[j, :, :(end-start), :] = img[:, start:end, :]
    imgs.append(img_windows)
    time[i] = cur_time

    word_embed = np.zeros(word_length, dtype=np.uint8)
    for j, char in enumerate(word):
      word_embed[j] = utils.char2index(char)
    words_embed.append(word_embed)

  print 'rescale ', num_rescale, '/', num_examples, ' images'
  print 'average scale factor: ', np.array(scales).mean()

  return (imgs, words_embed, time)

def process_and_save(dataset_dir, name, height, window_size, depth,
    imgs, words_embed, time, max_time):
  """
  Args:
      dataset_dir:
      name:
      height:
      window_size:
      depth:
      imgs:
      words_embed:
      time:
      max_time:

  Returns:
    image data in hdf5 file
  """
  num_examples = len(imgs)

  imgs_np = np.zeros((num_examples, max_time, height, window_size, depth),
      dtype=np.uint8)
  for i in range(num_examples):
    imgs_np[i, :time[i], :, :, :] = imgs[i]

  filename = os.path.join(dataset_dir, name+'.hdf5')
  print 'Writing ' + filename
  with h5py.File(filename, 'w') as hf:
    hf.create_dataset('imgs', data=imgs_np)
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    hf.create_dataset('words_embed', data=words_embed, dtype=dt)
    hf.create_dataset('time', data=time)

def main():
  """
  Read data in (IIT5K format), convert/process and save it to hdf5 format
  Returns:

  """
  with open('config.json', 'r') as json_file:
    json_data = json.load(json_file)
    dataset_dir = json_data['dataset_dir']
    height = json_data['height']
    window_size = json_data['window_size']
    depth = json_data['depth']
    embed_size = json_data['embed_size']
    stride = json_data['stride']

  train_dict = scipy.io.loadmat(dataset_dir + 'trainCharBound.mat')
  train_data = np.squeeze(train_dict['trainCharBound'])
  test_dict = scipy.io.loadmat(dataset_dir + 'testCharBound.mat')
  test_data = np.squeeze(test_dict['testCharBound'])

  imgs_train, words_embed_train, time_train = load_and_process(dataset_dir,
      train_data, height, window_size, depth, stride)
  imgs_test, words_embed_test, time_test = load_and_process(dataset_dir,
      test_data, height, window_size, depth, stride)

  max_time = int(max(max(time_train), max(time_test)))

  process_and_save(dataset_dir, 'train', height, window_size, depth,
      imgs_train, words_embed_train, time_train, max_time)

  process_and_save(dataset_dir, 'test', height, window_size, depth,
      imgs_test, words_embed_test, time_test, max_time)

if __name__ == '__main__':
  main()
