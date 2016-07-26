import matplotlib.pyplot as plt
import numpy as np
from math import log
import re

start = 0
losses_train = []
accuracies_train = []
losses_test = []
accuracies_test = []
ratio = 13
num_epoch = 3000

for i, line in enumerate(open('debug.log', 'r')):
  if i < start:
    continue
  if re.search(r'training loss in epoch (\d+): ([-+]?\d*\.\d+|\d+)', line):
    losses_train.append(float(re.search(r'average training loss in epoch (\d+): ([-+]?\d*\.\d+|\d+)', line).group(2)))
  elif re.search(r'training accuracy in epoch (\d+): ([-+]?\d*\.\d+|\d+)', line):
    accuracies_train.append(float(re.search(r'average training accuracy in epoch (\d+): ([-+]?\d*\.\d+|\d+)', line).group(2)))
  elif re.search(r'test loss: ([-+]?\d*\.\d+|\d+)', line) and not re.search(r'best test loss: ([-+]?\d*\.\d+|\d+)', line):
    losses_test.append(float(re.search(r'test loss: ([-+]?\d*\.\d+|\d+)', line).group(1)))
  elif re.search(r'test accuracy: ([-+]?\d*\.\d+|\d+)', line) and not re.search(r'best test accuracy: ([-+]?\d*\.\d+|\d+)', line):
    accuracies_test.append(float(re.search(r'test accuracy: ([-+]?\d*\.\d+|\d+)', line).group(1)))

print len(losses_train), len(accuracies_train), len(losses_test), len(accuracies_test)

losses_test = np.repeat(losses_test, ratio)
accuracies_test = np.repeat(accuracies_test, ratio)

plt.plot(losses_train[:num_epoch], label='training loss')
plt.plot(losses_test[:num_epoch],  label='test loss')
plt.yscale('log', nonposy='clip')
plt.ylabel('loss (log)')
plt.legend(loc='center', shadow=True)
plt.show()

plt.plot(accuracies_train[:num_epoch], label='training accuracy')
plt.plot(accuracies_test[:num_epoch],  label='test accuracy')
plt.ylabel('accuracy')
plt.legend(loc='center', shadow=True)
plt.show()
