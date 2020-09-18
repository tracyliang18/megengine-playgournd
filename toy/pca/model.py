import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

import setting


def get_params(l1, D):

    W1 = mge.Parameter(np.random.randn(l1, D).astype(np.float32))
    B1 = mge.Parameter(np.random.randn(l1).astype(np.float32))

    W2 = mge.Parameter(np.random.randn(D, l1).astype(np.float32))
    B2 = mge.Parameter(np.random.randn(D).astype(np.float32))

    return W1, B1, W2, B2


class TwolayerFC(M.Module):
    def __init__(self, l1, input_d):
        super().__init__()

        self.weight, self.bias, self.weight2, self.bias2 = get_params(l1, input_d)

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        rec = F.linear(x, self.weight2, self.bias2)

        return rec

if __name__ == '__main__':
    emb = 64
    inp = 64
    net1 = TwolayerFC(emb, inp)
    x = mge.tensor(np.random.randn(64, inp).astype(np.float32))
    out1 = net1(x)
    print(out1)

