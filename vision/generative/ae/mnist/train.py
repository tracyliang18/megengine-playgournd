import numpy as np
import tensorflow as tf
from datetime import datetime


import megengine as mge
import megengine.functional as F
from megengine.data import RandomSampler, SequentialSampler, DataLoader
from megengine.data.dataset import MNIST
from megengine.data.transform import RandomResizedCrop, Normalize, ToMode, Pad, Compose
import megengine.optimizer as optim

mge.set_log_file('log.txt')
logger = mge.get_logger(__name__)

#logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")


# dataset
root_dir = '/data/.cache/dataset/MNIST'
mnist_train_dataset = MNIST(root=root_dir, train=True, download=False)

mnist_test_dataset = MNIST(root=root_dir, train=False, download=False)

random_sampler = RandomSampler(dataset=mnist_train_dataset, batch_size=256)
sequential_sampler = SequentialSampler(dataset=mnist_test_dataset, batch_size=256)

mnist_train_dataloader = DataLoader(
    dataset=mnist_train_dataset,
    sampler=random_sampler,
    transform=Compose([
            RandomResizedCrop(output_size=28),
            # mean 和 std 分别是 MNIST 数据的均值和标准差，图片数值范围是 0~255
            #Normalize(mean=0.1307*255, std=0.3081*255),
            #Pad(2),
            # 'CHW'表示把图片由 (height, width, channel) 格式转换成 (channel, height, width) 格式
            #ToMode('CHW'),
        ])
)
mnist_test_dataloader = DataLoader(
    dataset=mnist_test_dataset,
    sampler=sequential_sampler,
)


# model
from model import get_net

net = get_net()

optimizer = optim.SGD(
                net.parameters(),
                lr=0.001,
            )

def get_kl_divergence(mean, var):

    return 1/2 * (mean ** 2 + var - F.log(var) - 1).sum(axis=1).mean()


data = mge.tensor()
label = mge.tensor(dtype="float32")

total_epochs = 256
for epoch in range(total_epochs):
    for step, batch_sample in enumerate(mnist_train_dataloader):
        batch_image, batch_label = batch_sample[0], batch_sample[1]
        batch_image = batch_image.transpose(0,3,1,2)

        data.set_value(batch_image)
        label.set_value(batch_image/255.)

        optimizer.zero_grad()

        net.train()
        reconstruct = net(data)
        reconstruct_loss = ((reconstruct - label).reshape(data.shape[0], -1) ** 2).sum(axis=1).mean()
        loss = reconstruct_loss

        optimizer.backward(loss)
        optimizer.step()

        logger.info("Epoch: {epoch}, step:{step}, lr:{lr}, acc:{acc}, loss:{loss}, rec:{rec_loss}".format(epoch=epoch, step=step,
            lr=optimizer.state_dict()['param_groups'][0]['lr'], acc=0, loss=loss.numpy().item(), rec_loss=reconstruct_loss.numpy()))

    path = 'models/save.ae.{}.mge'.format(epoch)
    mge.save(net.state_dict(), path)
