import numpy as np
import string
from typing import Tuple

from megengine.data.dataset import Dataset, ArrayDataset
from megengine.data import RandomSampler, DataLoader

CHARSET = ['@'] + list(string.ascii_lowercase) + ['-']
MAXLEN = 31
MINLEN = 1
batch_size = 512

def make_string_from_tensor(t):
    assert isinstance(t, np.ndarray)
    assert t.ndim== 3 or t.ndim == 2
    if t.ndim == 3:
        t = np.argmax(t, axis=2)
    ret = []
    for ti in t:
        indexes = ti
        s = []
        for ind in indexes:
            s.append(CHARSET[ind])
        ret.append(''.join(s))

    return ret

pos_to_query = np.eye(MAXLEN+1, dtype='float32')

def get_dataloader(instance_num=102400):
    #instance_num = 102400
    datas = []
    labels = []
    positions = []
    masks = []
    for i in range(instance_num):
        cur_len = np.random.randint(MINLEN, MAXLEN + 1)

        inp_seq = np.zeros((MAXLEN+1, len(CHARSET)), dtype='int32')
        mask = np.zeros((MAXLEN+1,), dtype='int32')
        out_seq = np.zeros((MAXLEN+1, ), dtype='int32')

        pos_encoding = pos_to_query

        inp_seq[cur_len][len(CHARSET)-1] = 1
        out_seq[cur_len] = len(CHARSET)-1
        mask[:cur_len+1] = 1

        for j in range(cur_len):
            pos = np.random.randint(1, len(CHARSET)-1) # not generate '@' and '-'
            inp_seq[j][pos] = 1
            out_seq[cur_len-1-j] = pos

        datas.append(inp_seq)
        labels.append(out_seq)
        positions.append(pos_encoding)
        masks.append(mask)


    reverse_dataset = ArrayDataset(datas, labels, positions, masks)
    random_sampler = RandomSampler(reverse_dataset, batch_size)
    dataloader = DataLoader(reverse_dataset, random_sampler)

    return dataloader

if __name__ == '__main__':
    dataloader = get_dataloader(instance_num=1024)
    for epoch in range(30):
        for step, batch in enumerate(dataloader):
            data, label, position, mask = batch
            inp = make_string_from_tensor(data)
            out = make_string_from_tensor(label)
            print(out)
            for ind, (i, o) in enumerate(zip(inp, out)):
                print(epoch, step, i, o)

