import numpy as np
import scipy.io
import cv2
import glob
import os
import h5py
import json
import utils
import math

def load_and_process(dataset_dir, data, height, window_size, depth, stride,
    visualize, visualize_dir):
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
  drop = 3 # drop frame when too much padding

  char_imgs = []
  chars_embed = []

  if visualize and not os.path.exists(visualize_dir):
    os.makedirs(visualize_dir)

  for i in range(num_examples):
    img = cv2.imread(dataset_dir + data[i][0][0])
    h = height
    scale = height/float(img.shape[0])
    w = int(round(scale*img.shape[1]))
    img = cv2.resize(img, (w, h))
    word = str(data[i][1][0])
    bboxes = data[i][2]

    if depth == 1:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      img = img[:, :, None]

    cur_time = int(math.ceil((w+window_size)/float(stride)-1))
    word_length = len(word)

    # Not enough time for target transition sequence
    assert cur_time-drop*2 > word_length

    img_windows = np.zeros((cur_time, height, window_size, depth), dtype=np.uint8)
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

      if visualize and i < 50 and j >= drop and j < cur_time-drop:
        #print i, j, cur_time, start1, end1, start2, end2, w
        cv2.imwrite(visualize_dir+str(i)+'_'+str(j)+'.jpg', img_windows[j])

    img_windows = img_windows[drop:cur_time-drop]
    cur_time -= drop*2

    imgs.append(img_windows)
    time[i] = cur_time

    word_embed = np.zeros(word_length, dtype=np.uint8)
    for j, char in enumerate(word):
      word_embed[j] = utils.char2index(char)
    words_embed.append(word_embed)

    for j, bbox in enumerate(bboxes):
      x = int(round(scale*(bbox[0]+bbox[2]/2)))
      start1 = max(x-window_size/2, 0)
      end1 = min(x+(window_size+1)/2, w)
      start2 = max(-(x-window_size/2), 0)
      end2 = min(start2+end1-start1, window_size)
      char_img = np.zeros((height, window_size, depth), dtype=np.uint8)
      char_img[:, start2:end2, :] = img[:, start1:end1, :]
      if start2 != 0:
        char_img[:, :start2, :] = char_img[:, start2, :][:, np.newaxis, :]
      if end2 != window_size:
        char_img[:, end2:, :] = char_img[:, end2-1, :][:, np.newaxis, :]
      char_imgs.append(char_img)
      chars_embed.append(word_embed[j])

      if visualize and i < 50:
        cv2.imwrite(visualize_dir+'char_'+str(i)+'_'+str(j)+'.jpg', char_img)

  return (imgs, words_embed, time, char_imgs, chars_embed)

def process_and_save(dataset_dir, name, height, window_size, depth,
    imgs, words_embed, time, max_time, char_imgs, chars_embed):
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

  num_chars = len(char_imgs)
  char_imgs_np = np.zeros((num_chars, height, window_size, depth),
      dtype=np.uint8)
  chars_embed_np = np.zeros(num_chars, dtype=np.uint8)
  for j in range(num_chars):
    char_imgs_np[j, :, :, :] = char_imgs[j]
    chars_embed_np[j] = chars_embed[j]

  filename = os.path.join(dataset_dir, name+'.hdf5')
  print 'Writing ' + filename
  with h5py.File(filename, 'w') as hf:
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    hf.create_dataset('imgs', data=imgs_np)
    hf.create_dataset('words_embed', data=words_embed, dtype=dt)
    hf.create_dataset('time', data=time)
    hf.create_dataset('char_imgs', data=char_imgs_np)
    hf.create_dataset('chars_embed', data=chars_embed_np)

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
    visualize = json_data['visualize']
    visualize_dir = json_data['visualize_dir']

  train_dict = scipy.io.loadmat(dataset_dir + 'trainCharBound.mat')
  train_data = np.squeeze(train_dict['trainCharBound'])
  test_dict = scipy.io.loadmat(dataset_dir + 'testCharBound.mat')
  test_data = np.squeeze(test_dict['testCharBound'])

  imgs_train, words_embed_train, time_train, char_imgs_train, \
      chars_embed_train = load_and_process(dataset_dir, train_data, height, \
      window_size, depth, stride, visualize, visualize_dir)
  imgs_test, words_embed_test, time_test, char_imgs_test, \
      chars_embed_test = load_and_process(dataset_dir, test_data, height, \
      window_size, depth, stride, False, visualize_dir)

  max_time = int(max(max(time_train), max(time_test)))

  process_and_save(dataset_dir, 'train', height, window_size, depth,
      imgs_train, words_embed_train, time_train, max_time, char_imgs_train, \
      chars_embed_train)

  process_and_save(dataset_dir, 'test', height, window_size, depth,
      imgs_test, words_embed_test, time_test, max_time, char_imgs_test, \
      chars_embed_test)

if __name__ == '__main__':
  main()
