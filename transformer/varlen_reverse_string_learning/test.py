import megengine as mge
from model import ReverseString
from dataset import get_dataloader, make_string_from_tensor, MAXLEN, CHARSET
import sys

model = ReverseString()
model.load_state_dict(mge.load(sys.argv[1]))

model.eval()


test_data = get_dataloader()
data = mge.tensor()
position = mge.tensor()
total = 0
correct = 0
for idx, (batch_data, batch_label, pos, mask) in enumerate(test_data):
    data.set_value(batch_data)
    position.set_value(pos)
    prob = model(data, position)
    prob = prob.reshape(-1, MAXLEN+1, len(CHARSET))
    predicted = prob.numpy().argmax(axis=2)
    inp_str = make_string_from_tensor(batch_data)
    pred_str = make_string_from_tensor(predicted)
    gt_str = make_string_from_tensor(batch_label)

    for i in range(len(inp_str)):
        total += 1
        correct += gt_str[i] == pred_str[i]

    print(correct, total)

