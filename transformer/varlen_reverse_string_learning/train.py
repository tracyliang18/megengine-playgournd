import megengine as mge
import logging
import megengine.functional as F
from megengine.data import RandomSampler, SequentialSampler, DataLoader
from megengine.data.dataset import MNIST
from megengine.data.transform import RandomResizedCrop, Normalize, ToMode, Pad, Compose
import megengine.optimizer as optim

from model import ReverseString
from dataset import get_dataloader
from megengine.functional.utils import zero_grad
from megengine.functional.nn import indexing_one_hot
from megengine.functional.elemwise import log
import megengine._internal as mgb



def cross_entropy_with_softmax(pred, label, axis=1, mask=None):
    n0 = pred.ndim
    n1 = label.ndim
    assert n0 == n1 + 1, (
        "target ndim must be one less than input ndim; input_ndim={} "
        "target_ndim={}".format(n0, n1)
    )

    num_classes = pred.shapeof(axis)

    # Denominator of the softmax
    offset = zero_grad(pred.max(axis=axis, keepdims=True))
    pred = pred - offset
    down = mgb.opr.elem.exp(pred).sum(axis=axis, keepdims=True)

    up = indexing_one_hot(pred, label, axis, keepdims=True)


    if mask is None:
        return (log(down) - up).mean()
    else:
        print(mask.shape)
        print((log(down) - up).shape)
        print(mask.sum())
        return ((log(down) - up) * mask).sum() / mask.sum()

def cross_entropy(pred, label, axis=1):
    n0 = pred.ndim
    n1 = label.ndim
    assert n0 == n1 + 1, (
        "target ndim must be one less than input ndim; input_ndim={} "
        "target_ndim={}".format(n0, n1)
    )

    num_classes = pred.shapeof(axis)

    # Denominator of the softmax
    #offset = zero_grad(pred.max(axis=axis, keepdims=True))
    #pred = pred - offset
    #down = mgb.opr.elem.exp(pred).sum(axis=axis, keepdims=True)

    up = indexing_one_hot(pred, label, axis, keepdims=True)


    return (-log(up+1e-8)).mean()


mge.set_log_file('log.txt')
mge.set_log_level(logging.ERROR)
logger = mge.get_logger(__name__)

net = ReverseString()
optimizer = optim.Adam(
                net.parameters(),
                lr=0.1,
            )

scheduler = optim.multi_step_lr.MultiStepLR(optimizer, (50, 100, 150, 200, 250, 300, 350))

data_loader = get_dataloader()

EPOCH = 400

data = mge.tensor()
label = mge.tensor(dtype='int32')
position = mge.tensor(dtype='float32')
mask = mge.tensor(dtype='int32')

for epoch in range(EPOCH):
    for step, batch in enumerate(data_loader):
        inp, out_gt, pos, m = batch

        data.set_value(inp)
        label.set_value(out_gt.reshape(-1))
        position.set_value(pos)
        mask.set_value(m)

        optimizer.zero_grad()

        logit = net(data, position)

        loss = cross_entropy_with_softmax(logit, label, mask=mask.reshape(-1, 1), axis=1) # 交叉熵损失函数
        #loss = cross_entropy_with_softmax(logit, label, axis=1) # 交叉熵损失函数

        optimizer.backward(loss)
        optimizer.step()
        print("Epoch: {epoch}, step:{step}, lr:{lr}, loss:{loss}".format(epoch=epoch, step=step,
            lr=optimizer.state_dict()['param_groups'][0]['lr'], loss=loss.numpy().item()))
            #lr=0.1, loss=loss.numpy().item()))
    scheduler.step()
    #optimizer.step()


    if epoch % 20 == 0:
        path = 'transformer.{}.mge'.format(epoch)
        mge.save(net.state_dict(), path)
