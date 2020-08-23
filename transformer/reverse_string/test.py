import megengine as mge
from model import ReverseString
from dataset import get_dataloader, make_string_from_tensor, MAXLEN


model = ReverseString()
model.load_state_dict(mge.load('transformer.60.mge'))

model.eval()


test_data = get_dataloader()
data = mge.tensor()
for idx, (batch_data, batch_label, batch_mask) in enumerate(test_data):
    data.set_value(batch_data)
    prob = model(data)
    prob = prob.reshape(-1, MAXLEN+1, 28)
    predicted = prob.numpy().argmax(axis=2)
    inp_str = make_string_from_tensor(batch_data)
    pred_str = make_string_from_tensor(predicted)
    gt_str = make_string_from_tensor(batch_label)

    for i in range(len(inp_str)):
        print(inp_str[i], gt_str[i], pred_str[i], batch_mask[i])


