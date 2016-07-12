import matplotlib.pyplot as plt
from math import log

find = 'loss = '
start = 449770
losses = []
count = 0
for i, line in enumerate(open('debug.log', 'r')):
  if i < start:
    continue
  idx = line.find(find)
  if idx != -1:
    loss = float(line[idx+len(find):])+1
    losses.append(loss)
    count += 1
    if count >= 20000 and loss > 3.0:
      print count, loss

print len(losses)
plt.plot(losses[:30000])
plt.ylabel('loss (log)')
plt.show()
