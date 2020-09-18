import numpy as np
from typing import Tuple

# from  https://megengine.org.cn/doc/latest/basic/data_load.html

from megengine.data.dataset import Dataset, ArrayDataset
from megengine.data import RandomSampler, DataLoader


class PCADataset(ArrayDataset):
    def __init__(self, input_dimension, num_points, batch_size=16, istrain=True):
        """
        生成如图1所示的二分类数据集，数据集长度为 num_points
        """

        means = [0.1*n for n in range(input_dimension)]
        scales =[1 for n in range(input_dimension)]

        deviation = [0.05 * (-1 if n % 2 ==0 else 1) for n in range(input_dimension)]
        sd = [0.1 * (-1 if n % 2 ==0 else 1) for n in range(input_dimension)]

        alls = []
        for i in range(input_dimension):
            m,s = means[i], scales[i]
            if not istrain:
                m += deviation[i]
                s += sd[i]

            cur = np.random.normal(m, s, num_points).astype(np.float32).reshape(-1, 1)
            print(cur)
            alls.append(cur)

        self.data = np.concatenate(alls, axis=1)

        super().__init__(self.data)
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
    dataset = PCADataset(16, 1024)

    # 通过 for 遍历数据集中的每一个样本
    for item in dataset:
        point = item[0]
        print(point.shape)
        print('tmp')
        break

