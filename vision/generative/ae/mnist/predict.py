import megengine as mge
import cv2
import torch
import megengine.functional as F
import numpy as np
import torchvision

from model import get_net

model = get_net()
model.load_state_dict(mge.load('models/save.ae.40.mge'))

model.eval()

# dataset
from megengine.data import RandomSampler, SequentialSampler, DataLoader
from megengine.data.dataset import MNIST
from megengine.data.transform import RandomResizedCrop, Normalize, ToMode, Pad, Compose
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


data = mge.tensor()
code = mge.tensor()
for step, batch_sample in enumerate(mnist_test_dataloader):
    imgs, _ = batch_sample
    imgs = imgs.transpose((0,3,1,2))
    data.set_value(imgs)
    reconstruct = (model(data).numpy() * 255).astype('uint8')
    all_img = np.concatenate([imgs, reconstruct], axis=1).reshape(-1, 1, 28, 28)
    show = torchvision.utils.make_grid(torch.tensor(all_img), 32)
    cv2.imwrite(str(step) + '.png', show.numpy().transpose(1,2,0))
    print(type(show))
    print(show.shape)
