import numpy as np
from typing import Tuple

# from  https://megengine.org.cn/doc/latest/basic/data_load.html

from megengine.data.dataset import Dataset, ArrayDataset
from megengine.data import RandomSampler, DataLoader

class XORDataset(Dataset):
    def __init__(self, num_points):
        """
        生成如图1所示的二分类数据集，数据集长度为 num_points
        """
        super().__init__()

        # 初始化一个维度为 (50000, 2) 的 NumPy 数组。
        # 数组的每一行是一个横坐标和纵坐标都落在 [-1, 1] 区间的一个数据点 (x, y)
        self.data = np.random.rand(num_points, 2).astype(np.float32) * 2 - 1
        # 为上述 NumPy 数组构建标签。每一行的 (x, y) 如果符合 x*y < 0，则对应标签为1，反之，标签为0
        self.label = np.zeros(num_points, dtype=np.int32)
        for i in range(num_points):
            self.label[i] = 1 if np.prod(self.data[i]) < 0 else 0

    # 定义获取数据集中每个样本的方法
    def __getitem__(self, index: int) -> Tuple:
        return self.data[index], self.label[index]

    # 定义返回数据集长度的方法
    def __len__(self) -> int:
        return len(self.data)

class XORDataset2(ArrayDataset):
    def __init__(self, num_points, batch_size=16):
        """
        生成如图1所示的二分类数据集，数据集长度为 num_points
        """

        # 初始化一个维度为 (50000, 2) 的 NumPy 数组。
        # 数组的每一行是一个横坐标和纵坐标都落在 [-1, 1] 区间的一个数据点 (x, y)
        # np.random.seed(2020)
        self.data = np.random.rand(num_points, 2).astype(np.float32) * 2 - 1
        # 为上述 NumPy 数组构建标签。每一行的 (x, y) 如果符合 x*y < 0，则对应标签为1，反之，标签为0
        self.label = np.zeros(num_points, dtype=np.int32)
        for i in range(num_points):
            self.label[i] = 1 if np.prod(self.data[i]) < 0 else 0

        super().__init__(self.data, self.label)
        self.random_sampler = RandomSampler(dataset=self, batch_size=batch_size, seed=1024)
        self.dataloader = DataLoader(dataset=self, sampler=self.random_sampler)

    def instance_generator(self):
        for (X, Y) in self:
            yield (X, Y)

    def batch_generator(self):
        for batch in self.dataloader:
            yield batch




if __name__ == '__main__':
    np.random.seed(2020)
    # 构建一个包含 30000 个点的训练数据集
    #xor_train_dataset = XORDataset(30000)
    xor_train_dataset = XORDataset2(30000)
    print("The length of train dataset is: {}".format(len(xor_train_dataset)))

    # 通过 for 遍历数据集中的每一个样本
    for cor, tag in xor_train_dataset:
        print("The first data point is: {}, {}".format(cor, tag))
        break

    for cor, tag in xor_train_dataset:
        print("The first data point is: {}, {}".format(cor, tag))
        break

    print("The second data point is: {}".format(xor_train_dataset[1]))


    # iterate batch from random sampler
    for batch in xor_train_dataset.batch_generator():
        print(batch)
        break

    xxx
    # iterate array dataset
    if True:
        gen = xor_train_dataset.instance_generator()
        for inst in gen:
            print(inst)
