import megengine as mge
import megengine.functional as F
import megengine.optimizer as optim

from dataset import XORDataset2
from model import TwolayerFC2, MorelayerFC2
import setting
from utils import accuracy


if __name__ == '__main__':
    dataset = XORDataset2(setting.points_num, setting.batch_size)
    #model = TwolayerFC2(setting.embedding)
    model = MorelayerFC2(setting.embedding)

    data = mge.tensor()
    label = mge.tensor(dtype="int32")
    optimizer = optim.SGD(
                    model.parameters(), # 参数列表，将指定参数与优化器绑定
                    #lr=0.05,  # 学习速率
                    lr=setting.learning_rate,  # 学习速率
                )
    total_epochs = 10
    for epoch in range(total_epochs):
        total_loss = 0
        batch_generator = dataset.batch_generator()
        accs = 0
        step_count = 0
        for step, (batch_data, batch_label) in enumerate(batch_generator):
            data.set_value(batch_data)
            label.set_value(batch_label)
            optimizer.zero_grad() # 将参数的梯度置零

            prob = model(data)

            loss = F.cross_entropy(prob, label) # 交叉熵损失函数
            total_loss += loss.numpy().item()

            optimizer.backward(loss) # 反传计算梯度
            optimizer.step()  # 根据梯度更新参数值

            acc = accuracy(prob.numpy(), batch_label)
            accs += acc
            step_count += 1
            #print("step: {}, loss: {}, acc: {}".format(step, loss, )
            #print(step, loss)
        print("epoch: {}, average loss {}, dataset len: {}, acc: {}".format(epoch, total_loss/len(dataset), len(dataset), accs / step_count))

    path = '/tmp/save.mge'
    mge.save(model.state_dict(), path)


