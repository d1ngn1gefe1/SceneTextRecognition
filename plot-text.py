import matplotlib.pyplot as plt
import numpy as np
from math import log
import re

start = 0
ratio = 4
num_epoch = 50000

losses_train = []
losses_test = []
edit_distance_test = []
character_accuracy_test = []

bins1 = [723, 155, 77, 32, 17, 13, 5, 1, 1]
bins2 = [769, 169, 73, 43, 22, 15, 5, 3, 1]
n_groups = max(len(bins1), len(bins2))
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
rects1 = plt.bar(index, bins1, bar_width, color='b', label='IIIT5K')
rects2 = plt.bar(index+bar_width, bins2, bar_width, color='r', label='MJSynth+IIIT5K')
plt.xlabel('Edit distance')
plt.ylabel('Counts')
plt.xticks(index + bar_width, ['0', '1', '2', '3', '4', '5', '6', '7', '8'])
plt.legend()
plt.tight_layout()
plt.show()


for i, line in enumerate(open('./checkpoint-text-iiit5k/debug.log', 'r')):
  if i < start:
    continue
  if re.search(r'training loss in epoch (\d+), step (\d+): ([-+]?\d*\.\d+|\d+)', line):
    losses_train.append(float(re.search(r'training loss in epoch (\d+), step (\d+): ([-+]?\d*\.\d+|\d+)', line).group(3)))
  elif re.search(r'test loss: ([-+]?\d*\.\d+|\d+)', line):
    losses_test.append(float(re.search(r'test loss: ([-+]?\d*\.\d+|\d+)', line).group(1)))
  elif re.search(r'edit distance: ([-+]?\d*\.\d+|\d+)', line):
    edit_distance_test.append(float(re.search(r'edit distance: ([-+]?\d*\.\d+|\d+)', line).group(1)))
  elif re.search(r'character accuracy: ([-+]?\d*\.\d+|\d+)', line):
    character_accuracy_test.append(float(re.search(r'character accuracy: ([-+]?\d*\.\d+|\d+)', line).group(1)))

print len(losses_train), len(losses_test), len(edit_distance_test), len(character_accuracy_test)
losses_test = np.repeat(losses_test, ratio)
edit_distance_test = np.repeat(edit_distance_test, ratio)
character_accuracy_test = np.repeat(character_accuracy_test, ratio)
print len(losses_train), len(losses_test), len(edit_distance_test), len(character_accuracy_test)


plt.plot(losses_train[:num_epoch], label='training loss')
plt.plot(losses_test[:num_epoch],  label='test loss')
plt.yscale('log', nonposy='clip')
plt.ylabel('loss (log)')
plt.legend(loc='upper center', shadow=True)
plt.show()

plt.plot(edit_distance_test[:num_epoch], label='edit distance')
plt.ylabel('edit distance')
plt.legend(loc='center', shadow=True)
plt.show()

plt.plot(character_accuracy_test[:num_epoch], label='character error')
plt.ylabel('character error')
plt.legend(loc='center', shadow=True)
plt.show()
