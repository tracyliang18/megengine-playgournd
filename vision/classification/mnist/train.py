import megengine as mge
import megengine.functional as F
from megengine.data import RandomSampler, SequentialSampler, DataLoader
from megengine.data.dataset import MNIST
from megengine.data.transform import RandomResizedCrop, Normalize, ToMode, Pad, Compose
import megengine.optimizer as optim

mge.set_log_file('log.txt')
logger = mge.get_logger(__name__)



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
from model import ResNet_MNIST

net = ResNet_MNIST([2, 2])

optimizer = optim.SGD(
                net.parameters(),
                lr=0.01,
            )

def calc_accuracy(pred, label):
    predicted = pred.argmax(axis=1)
    correct = (predicted==label).sum()
    total = label.shape[0]
    return correct/total

class moving_average_metrit(object):

    def __init__(self):
        self.cnt = 0
        self.value = 0

    def tick(self, v, c):
        total = self.cnt + c
        self.value = (self.value * self.cnt + v * c) / total
        self.cnt = total

    def reset(self):
        self.cnt = 0
        self.value = 0

data = mge.tensor()
label = mge.tensor(dtype="int32")

def get_dataset_acc(net, dataset):
    net.eval()
    acc = moving_average_metrit()
    for step, batch_sample in enumerate(dataset):
        batch_image, batch_label = batch_sample[0], batch_sample[1]
        batch_image = batch_image.transpose(0,3,1,2)
        data.set_value(batch_image)
        logit = net(data)
        cur_acc = calc_accuracy(logit.numpy(), batch_label)
        acc.tick(cur_acc, batch_image.shape[0])


    return acc.value



total_epochs = 64
for epoch in range(total_epochs):
    for step, batch_sample in enumerate(mnist_train_dataloader):
        batch_image, batch_label = batch_sample[0], batch_sample[1]
        batch_image = batch_image.transpose(0,3,1,2)
        data.set_value(batch_image)
        label.set_value(batch_label)
        optimizer.zero_grad()

        net.train()
        logit = net(data)
        loss = F.cross_entropy_with_softmax(logit, label) # 交叉熵损失函数

        optimizer.backward(loss)
        optimizer.step()

        net.eval()
        acc = calc_accuracy(logit.numpy(), batch_label)

        logger.info("Epoch: {epoch}, step:{step}, lr:{lr}, acc:{acc}, loss:{loss}".format(epoch=epoch, step=step,
            lr=optimizer.state_dict()['param_groups'][0]['lr'], acc=acc, loss=loss.numpy().item()))

    train_acc = get_dataset_acc(net, mnist_train_dataloader)
    test_acc = get_dataset_acc(net, mnist_test_dataloader)
    logger.info("Epoch: {epoch}, train_acc: {train_acc}, test_acc:{test_acc}".format(epoch=epoch, train_acc=train_acc,
        test_acc=test_acc))

path = '/tmp/save.mge'
mge.save(net.state_dict(), path)
