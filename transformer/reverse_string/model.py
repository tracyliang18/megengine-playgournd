import math

import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import numpy as np
from dataset import MAXLEN


L = MAXLEN+1
eye = mge.tensor(np.eye(L, dtype='float32'))
class TransformerBlock(M.Module):
    # input b, l, e
    def __init__(self, i, value_embedding, key_embedding):
        self.key_embedding = key_embedding
        super(TransformerBlock).__init__()
        self.position_encoding = M.Linear(L, key_embedding)
        self.init_map = M.Linear(i, key_embedding)
        self.value_mapping = M.Linear(key_embedding, value_embedding)
        self.key_mapping = M.Linear(key_embedding, key_embedding)
        self.query_mapping = M.Linear(key_embedding, key_embedding)
        self.norm = M.BatchNorm1d(key_embedding)

    def forward(self, x):
        # x: bxlxi
        x = self.init_map(x) # bxlxe
        ori = x
        p = self.position_encoding(eye)
        x = x + p

        values = self.value_mapping(x) #bxlxe
        keys = self.key_mapping(x) #bxlxe
        querys = self.key_mapping(x)

        #print('transformer', values.shape, keys.shape, querys.shape)

        attention = F.softmax(F.batched_matrix_mul(querys, keys.dimshuffle(0,2,1)), axis=1) #bxlxl
        #print(attention[0])
        #print(attention[0].sum(axis=0))
        #print('attention', attention.shape)
        out = F.batched_matrix_mul(values.dimshuffle(0, 2, 1), attention)

        out = out.dimshuffle(0, 2, 1)
        out = out + ori
        out = F.relu(out)
        #a,b,c = out.shape[0], out.shape[1], out.shape[2]
        #tmp = out.reshape(-1, self.key_embedding)
        #i = tmp.shape[0]
        #out = self.norm(tmp)
        #out = out.reshape(a,b,c)


        return out

class ReverseString(M.Module):
    def __init__(self):
        E = 32
        self.t1 = TransformerBlock(28, E, E)
        self.t2 = TransformerBlock(E, E, E)
        self.t3 = TransformerBlock(E, E, E)
        self.t4 = TransformerBlock(E, E, E)
        self.fc = M.Linear(E, 28)


    def forward(self, x):
        x = self.t1(x)
        x = F.relu(x)
        x = self.t2(x)
        x = F.relu(x)
        x = self.t3(x)
        x = F.relu(x)
        x = self.t4(x)
        x = F.relu(x)
        logit = self.fc(x)
        logit = logit.reshape(-1, 28)
        #print('logit', logit.shape)
        #print('logit0', logit[0])
        #print('logit1', logit[1])

        return logit



if __name__ == '__main__':
    trans = TransformerBlock(28, 32, 32)
    from dataset import get_dataloader
    data_loader = get_dataloader()
    inp_sequence = mge.tensor(dtype='float32')
    for step, batch in enumerate(data_loader):
        inp, label, mask = batch
        inp_sequence.set_value(inp)
        out = trans(inp_sequence)
        print(inp_sequence.shape, out.shape)



