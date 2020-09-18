import matplotlib.pyplot as plt
import os
from collections import defaultdict
import glob
import numpy as np


def get_exp_train_test_loss(exp, epoch=20):
    log_file = os.path.join(exp, 'log.txt')
    with open(log_file) as fin:
        for line in fin:
            if 'epoch: {}'.format(epoch) in line:
                records = line.split(',')
                train_loss = float(records[1].split(':')[1])
                test_loss = float(records[2].split(':')[1])

                return train_loss, test_loss

    return None, None


exps = sorted(glob.glob('outs/model*'))

keys = set()
exp2data = defaultdict(list)
for exp in exps:
    key = '_'.join(os.path.basename(exp).split('_')[:2])
    keys.add(key)
    res = get_exp_train_test_loss(exp, epoch=19)
    if res[0] is not None:
        exp2data[key].append(res)

x = np.array([1, 2, 3, 4, 5])
y = np.power(x, 2) # Effectively y = x**2
e = np.array([1.5, 2.6, 3.7, 4.6, 5.5])

x = sorted(keys, key=lambda x: int(x.split('_')[1]))
print(x)
y_train = []
e_train = []
y_test = []
e_test = []

for key in sorted(keys):
    arrs = exp2data[key]
    if arrs:
        train, test = list(zip(*arrs))

        train_mean = np.mean(train)
        train_std = np.std(train)

        y_train.append(train_mean)
        e_train.append(train_std)

        test_mean = np.mean(test)
        test_std = np.std(test)

        y_test.append(test_mean)
        e_test.append(test_std)
    else:

        y_train.append(0)
        e_train.append(0)

        y_test.append(0)
        e_test.append(0)

plt.errorbar(x, y_train, e_train, linestyle='None', marker='^', label='train')
#plt.errorbar(x, y_test, e_test, linestyle='None', marker='+', label='test')

plt.savefig('tmp.png')
