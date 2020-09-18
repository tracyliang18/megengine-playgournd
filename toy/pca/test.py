import megengine as mge
from model import TwolayerFC2, MorelayerFC2
import setting
import dataset

#model = TwolayerFC2(setting.embedding)
model = MorelayerFC2(setting.embedding)
model.load_state_dict(mge.load('/tmp/save.mge'))

model.eval()


#test_data = dataset.XORDataset2(6400)
test_data = dataset.XORDataset2(setting.points_num, setting.batch_size)
data = mge.tensor()
correct = 0
total = 0
gnr = test_data.batch_generator()
for idx, (batch_data, batch_label) in enumerate(gnr):
    data.set_value(batch_data)
    prob = model(data)
    predicted = prob.numpy().argmax(axis=1)
    correct += (predicted==batch_label).sum()
    total += batch_label.shape[0]
print("correct: {}, total: {}, accuracy: {}".format(correct, total, float(correct)/total))


