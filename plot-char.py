import matplotlib.pyplot as plt
import numpy as np
from math import log
import re

start = 0
ratio = 4
num_epoch = 50000

losses_train = []
accuracies_train = []
losses_test = []
accuracies_test = []
for i, line in enumerate(open('./checkpoint-char-with-stn/debug.log', 'r')):
  if i < start:
    continue
  if re.search(r'training loss in epoch (\d+), step (\d+): ([-+]?\d*\.\d+|\d+)', line):
    losses_train.append(float(re.search(r'training loss in epoch (\d+), step (\d+): ([-+]?\d*\.\d+|\d+)', line).group(3)))
  elif re.search(r'training accuracy in epoch (\d+), step (\d+): ([-+]?\d*\.\d+|\d+)', line):
    accuracies_train.append(float(re.search(r'training accuracy in epoch (\d+), step (\d+): ([-+]?\d*\.\d+|\d+)', line).group(3)))
  elif re.search(r'test loss: ([-+]?\d*\.\d+|\d+)', line) and not re.search(r'best test loss: ([-+]?\d*\.\d+|\d+)', line):
    losses_test.append(float(re.search(r'test loss: ([-+]?\d*\.\d+|\d+)', line).group(1)))
  elif re.search(r'test accuracy: ([-+]?\d*\.\d+|\d+)', line) and not re.search(r'best test accuracy: ([-+]?\d*\.\d+|\d+)', line):
    accuracies_test.append(float(re.search(r'test accuracy: ([-+]?\d*\.\d+|\d+)', line).group(1)))

print len(losses_train), len(accuracies_train), len(losses_test), len(accuracies_test)
losses_test = np.repeat(losses_test, ratio)
accuracies_test = np.repeat(accuracies_test, ratio)
print len(losses_train), len(accuracies_train), len(losses_test), len(accuracies_test)

losses_train_2 = []
accuracies_train_2 = []
losses_test_2 = []
accuracies_test_2 = []
for i, line in enumerate(open('./checkpoint-char-no-stn/debug.log', 'r')):
  if i < start:
    continue
  if re.search(r'training loss in epoch (\d+), step (\d+): ([-+]?\d*\.\d+|\d+)', line):
    losses_train_2.append(float(re.search(r'training loss in epoch (\d+), step (\d+): ([-+]?\d*\.\d+|\d+)', line).group(3)))
  elif re.search(r'training accuracy in epoch (\d+), step (\d+): ([-+]?\d*\.\d+|\d+)', line):
    accuracies_train_2.append(float(re.search(r'training accuracy in epoch (\d+), step (\d+): ([-+]?\d*\.\d+|\d+)', line).group(3)))
  elif re.search(r'test loss: ([-+]?\d*\.\d+|\d+)', line) and not re.search(r'best test loss: ([-+]?\d*\.\d+|\d+)', line):
    losses_test_2.append(float(re.search(r'test loss: ([-+]?\d*\.\d+|\d+)', line).group(1)))
  elif re.search(r'test accuracy: ([-+]?\d*\.\d+|\d+)', line) and not re.search(r'best test accuracy: ([-+]?\d*\.\d+|\d+)', line):
    accuracies_test_2.append(float(re.search(r'test accuracy: ([-+]?\d*\.\d+|\d+)', line).group(1)))

print len(losses_train_2), len(accuracies_train_2), len(losses_test_2), len(accuracies_test_2)
losses_test_2 = np.repeat(losses_test_2, ratio)
accuracies_test_2 = np.repeat(accuracies_test_2, ratio)
print len(losses_train_2), len(accuracies_train_2), len(losses_test_2), len(accuracies_test_2)


plt.plot(losses_train[:num_epoch], label='training loss stn')
plt.plot(losses_test[:num_epoch],  label='test loss stn')
plt.plot(losses_train_2[:num_epoch], label='training loss')
plt.plot(losses_test_2[:num_epoch], label='test loss')
plt.yscale('log', nonposy='clip')
plt.ylabel('loss (log)')
plt.legend(loc='upper center', shadow=True)
plt.show()

plt.plot([a_i - b_i for a_i, b_i in zip(losses_train_2[:num_epoch], losses_train[:num_epoch])], label='training loss diff')
plt.plot([a_i - b_i for a_i, b_i in zip(losses_test_2[:num_epoch], losses_test[:num_epoch])],  label='test loss diff')
plt.ylabel('loss diff')
plt.legend(loc='upper center', shadow=True)
plt.show()

plt.plot(accuracies_train[:num_epoch], label='training accuracy stn')
plt.plot(accuracies_test[:num_epoch],  label='test accuracy stn')
plt.plot(accuracies_train_2[:num_epoch], label='training accuracy')
plt.plot(accuracies_test_2[:num_epoch], label='test accuracy')
plt.ylabel('accuracy')
plt.legend(loc='center', shadow=True)
plt.show()

plt.plot([a_i - b_i for a_i, b_i in zip(accuracies_train_2[:num_epoch], accuracies_train[:num_epoch])], label='training accuracy diff')
plt.plot([a_i - b_i for a_i, b_i in zip(accuracies_test_2[:num_epoch], accuracies_test[:num_epoch])],  label='test accuracy diff')
plt.ylabel('accuracy diff')
plt.legend(loc='upper center', shadow=True)
plt.show()
