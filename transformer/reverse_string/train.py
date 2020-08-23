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
        return ((log(down) - up) * mask).sum() / mask.sum()


mge.set_log_file('log.txt')
mge.set_log_level(logging.ERROR)
logger = mge.get_logger(__name__)

net = ReverseString()
optimizer = optim.SGD(
                net.parameters(),
                lr=0.001,
            )

scheduler = optim.multi_step_lr.MultiStepLR(optimizer, (30, 60, 90))

data_loader = get_dataloader()

EPOCH = 120

data = mge.tensor()
label = mge.tensor(dtype='int32')
mask = mge.tensor(dtype='int32')

for epoch in range(EPOCH):
    for step, batch in enumerate(data_loader):
        inp, out_gt, cur_mask = batch

        data.set_value(inp)
        label.set_value(out_gt.reshape(-1))
        mask.set_value(cur_mask.reshape(-1,1))
        optimizer.zero_grad()

        logit = net(data)

        loss = cross_entropy_with_softmax(logit, label, mask=mask, axis=1) # 交叉熵损失函数

        optimizer.backward(loss)
        optimizer.step()
        print("Epoch: {epoch}, step:{step}, lr:{lr}, loss:{loss}".format(epoch=epoch, step=step,
            #lr=optimizer.state_dict()['param_groups'][0]['lr'], loss=loss.numpy().item()))
            lr=scheduler.get_lr(), loss=loss.numpy().item()))
    scheduler.step()

    if epoch % 20 == 0:
        path = 'transformer.{}.mge'.format(epoch)
        mge.save(net.state_dict(), path)
