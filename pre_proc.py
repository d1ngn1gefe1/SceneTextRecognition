import numpy as np
import scipy.io
import cv2
import glob
import os
import h5py

dataset_dir = '/home/local/ANT/zelunluo/Documents/IIIT5K/'
height = 32
depth = 3
buf_size = 500000

def convert_and_save(data, name):
  num_examples = data.shape[0]
  imgs = np.empty((buf_size, height, depth))
  words = []
  imgs_length = []
  words_length = []
  curIdx = 0

  for i in range(num_examples):
    img = cv2.imread(dataset_dir + data[i][0][0])
    h = height
    w = height*img.shape[1]/img.shape[0]
    img = cv2.resize(img, (h, w))
    word = str(data[i][1][0])

    imgs[curIdx:(curIdx+w), :, :] = img
    curIdx += w
    words.append(word)
    imgs_length.append(w)
    words_length.append(len(word))

  imgs = imgs[:curIdx, :, :]

  filename = os.path.join(dataset_dir, name+'.hdf5')
  print 'Writing ' + filename
  with h5py.File(filename, 'w') as hf:
    hf.create_dataset('imgs', data=imgs)
    hf.create_dataset('words', data=words)
    hf.create_dataset('imgs_length', data=imgs_length)
    hf.create_dataset('words_length', data=words_length)

def main():
  train_dict = scipy.io.loadmat(dataset_dir + 'trainCharBound.mat')
  train_data = np.squeeze(train_dict['trainCharBound'])
  test_dict = scipy.io.loadmat(dataset_dir + 'testCharBound.mat')
  test_data = np.squeeze(test_dict['testCharBound'])

  convert_and_save(train_data, 'train')
  convert_and_save(test_data, 'test')

if __name__ == '__main__':
  main()
