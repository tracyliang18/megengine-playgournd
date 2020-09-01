import math

import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import numpy as np
from megengine.core.tensor_nn import Parameter
from dataset import MAXLEN, CHARSET

L = MAXLEN+1
E = len(CHARSET)
pos_to_query = np.eye(L, dtype='float32')[::-1, :]
eye = mge.tensor()
class TransformerBlock(M.Module):
    # input b, l, e
    def __init__(self, embedding_num=2048):
        self.query_mapping1 = M.Linear(L+E, embedding_num, bias=False)
        self.key_mapping1 = M.Linear(L+E, embedding_num, bias=False)

        #self.query_mapping1 = M.Linear(L, embedding_num, bias=False)
        #self.key_mapping1 = M.Linear(L, embedding_num, bias=False)


    def forward(self, x, position):
        # x: bxlxi
        values = x # bxlxe

        i = F.concat([x, position], axis=2)
        #i = position

        querys = self.query_mapping1(i)
        keys = self.key_mapping1(i)

        attention = F.softmax(F.batched_matrix_mul(querys, keys.dimshuffle(0,2,1)), axis=2) #bxlxl
        out = F.batched_matrix_mul(values.dimshuffle(0, 2, 1), attention.dimshuffle(0, 2, 1))

        out = out.dimshuffle(0, 2, 1)

        return out

class ReverseString(M.Module):
    def __init__(self):
        MID=2048
        self.t1 = TransformerBlock(embedding_num=MID)
        self.fc = M.Linear(L+E, E)

    def get_logit(self, x, pos):
        x = self.t1(x, pos)
        logit = x
        #logit = self.fc(x)

        return logit

    def forward(self, x, pos):
        x = self.get_logit(x, pos)
        flatten = x.reshape(-1, E)

        return flatten



if __name__ == '__main__':
    trans = ReverseString()
    mge.save(trans.state_dict(), 'test.mge')
    from dataset import get_dataloader, make_string_from_tensor
    data_loader = get_dataloader(instance_num=1024)
    inp_sequence = mge.tensor(dtype='float32')
    position = mge.tensor(dtype='float32')
    trans.eval()
    for step, batch in enumerate(data_loader):
        inp, label,pos,mask = batch
        inp_sequence.set_value(inp)
        position.set_value(pos)
        out = trans.get_logit(inp_sequence, position)

        inp = make_string_from_tensor(inp)
        print(inp)
        print(out)
        print(out.shape)
        out = make_string_from_tensor(out.numpy())
        for ind, (i, o) in enumerate(zip(inp, out)):
            print(ind, i, o, mask[ind], mask[ind].sum(), i.index('-'))



