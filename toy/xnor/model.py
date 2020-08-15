import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

import setting

L = setting.embedding
def get_params(l1):

    W1 = mge.Parameter(np.random.randn(l1, 2).astype(np.float32))
    B1 = mge.Parameter(np.random.randn(l1).astype(np.float32))

    W2 = mge.Parameter(np.random.randn(2, l1).astype(np.float32))
    B2 = mge.Parameter(np.random.randn(2).astype(np.float32))

    return W1, B1, W2, B2

W1, B1, W2, B2 = get_params(L)

class TwolayerFC(M.Module):
    def __init__(self, l1=L):
        super().__init__()

        self.weight, self.bias, self.weight2, self.bias2 = W1, B1, W2, B2




    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        logit = F.linear(x, self.weight2, self.bias2)
        prob = F.softmax(logit)

        return prob


class TwolayerFC2(M.Module):
    def __init__(self, l1):
        super(TwolayerFC2, self).__init__()
        self.fc1 = M.Linear(2, l1)
        self.fc1.weight = W1
        self.fc1.bias = B1
        self.fc2 = M.Linear(l1, 2)
        self.fc2.weight = W2
        self.fc2.bias = B2

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        prob = F.softmax(x)

        return prob

class MorelayerFC2(M.Module):
    def __init__(self, l1):
        super(MorelayerFC2, self).__init__()
        self.fc1 = M.Linear(2, l1)
        self.fc2 = M.Linear(l1, l1)
        self.fc3 = M.Linear(l1, l1)
        self.fc4 = M.Linear(l1, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        prob = F.softmax(x)

        return prob


if __name__ == '__main__':
    net1 = TwolayerFC(L)
    net2 = TwolayerFC2(L)
    x = mge.tensor(np.random.randn(64, 2).astype(np.float32))
    out1 = net1(x)
    out2 = net2(x)

    np.testing.assert_allclose(out1, out2)

