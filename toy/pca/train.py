import os
import megengine as mge
import megengine.functional as F
import megengine.optimizer as optim

from dataset import PCADataset
from model import TwolayerFC
import setting
from utils import accuracy, mkdir_p

logger = mge.get_logger(__name__)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding", required=True, type=int, help="")
    parser.add_argument("-i", "--input", required=True, type=int, help="")
    parser.add_argument("-s", "--save", required=True, type=str, help="")
    parser.add_argument("--epoch", required=True, type=int, help="")
    args = parser.parse_args()

    mkdir_p(args.save)
    logfile = os.path.join(args.save, "log.txt")
    #open(logfile, 'w')
    mge.set_log_file(logfile)

    train_dataset = PCADataset(args.input, setting.points_num, setting.batch_size)
    test_dataset = PCADataset(args.input, setting.points_num, batch_size=setting.batch_size, istrain=False)
    model = TwolayerFC(args.embedding, args.input)

    data = mge.tensor(dtype='float32')
    label = mge.tensor(dtype="float32")
    optimizer = optim.SGD(
                    model.parameters(), # 参数列表，将指定参数与优化器绑定
                    lr=setting.learning_rate,  # 学习速率
                )
    total_epochs = args.epoch
    for epoch in range(total_epochs):
        model.train()
        train_batch_generator = train_dataset.batch_generator()
        test_batch_generator = test_dataset.batch_generator()

        total_loss = 0
        step_count = 0
        for step, (batch_data, ) in enumerate(train_batch_generator):
            data.set_value(batch_data)
            label.set_value(batch_data)
            optimizer.zero_grad() # 将参数的梯度置零

            rec = model(data)

            loss = ((rec - label) ** 2).mean()
            print(loss)
            total_loss += loss.numpy().item()

            optimizer.backward(loss) # 反传计算梯度
            optimizer.step()  # 根据梯度更新参数值

            step_count += 1

        train_loss = 0
        test_loss = 0

        train_batch_generator = train_dataset.batch_generator()
        for step, (batch_data, ) in enumerate(train_batch_generator):
            data.set_value(batch_data)
            label.set_value(batch_data)

            rec = model(data)

            loss = ((rec - label) ** 2).mean() * batch_data.shape[0]
            train_loss += loss.numpy().item()



        test_loss = 0

        model.eval()
        for step, (batch_data, ) in enumerate(test_batch_generator):
            data.set_value(batch_data)

            rec = model(data)

            loss = ((rec - data) ** 2).mean() * batch_data.shape[0]
            test_loss += loss.numpy().item()


        logger.info("epoch: {}, average train loss: {}, average test loss: {}, dataset len: {}".format(epoch, train_loss/len(train_dataset), test_loss/len(test_dataset), len(train_dataset)))


    path = os.path.join(args.save, 'model.mge')
    mge.save(model.state_dict(), path)


