import megengine as mge
import cv2
import torch
import megengine.functional as F
import numpy as np
import torchvision

from model import get_net

model = get_net()
model.load_state_dict(mge.load('models/save.ae.45.mge'))

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
_onehot = mge.tensor()
for step, batch_sample in enumerate(mnist_test_dataloader):
    imgs, target = batch_sample
    imgs = imgs.transpose((0,3,1,2))
    gaussian = np.random.normal(size = (imgs.shape[0], model.layers[-1], 3, 3))
    onehot = np.broadcast_to(np.eye(10)[target][:,:,np.newaxis,np.newaxis], (imgs.shape[0],10,3,3))

    data.set_value(imgs)
    encode_mean, _ = model.encoding(data)
    encode_mean = encode_mean.numpy()
    #_onehot.set_value(onehot)
    h = encode_mean.shape[0]
    #encode_mean  = F.broadcast_to(F.add_axis(F.add_axis(encode_mean, 2), 3), (h, model.layers[-1], 3, 3))
    #encode_mean = np.broadcast_to(encode_mean[:,:,np.newaxis, np.newaxis], (h, model.layers[-1], 3, 3))

    code_ = np.concatenate([encode_mean, onehot], axis=1)

    code.set_value(code_)
    reconstruct_nonoise = model.decoding(code)

    #code.set_value(np.concatenate([gaussian, onehot], axis=1))
    code.set_value(gaussian)
    _onehot.set_value(onehot)
    _, _, reconstruct_noise = model(data, code, _onehot)

    code_ = np.concatenate([gaussian, onehot], axis=1)
    code.set_value(code_)
    reconstruct_random = model.decoding(code)

    reconstruct_nonoise = (reconstruct_nonoise.numpy() * 255).astype('uint8')
    reconstruct_noise = (reconstruct_noise.numpy() * 255).astype('uint8')
    reconstruct_random = (reconstruct_random.numpy() * 255).astype('uint8')

    all_img = np.concatenate([imgs, reconstruct_nonoise, reconstruct_noise, reconstruct_random], axis=1).reshape(-1, 1, 28, 28)
    show = torchvision.utils.make_grid(torch.tensor(all_img), 32)
    cv2.imwrite(str(step) + '.png', show.numpy().transpose(1,2,0))
    print(type(show))
    print(show.shape)
