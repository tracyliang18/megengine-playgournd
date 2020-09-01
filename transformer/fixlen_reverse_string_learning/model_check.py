import megengine as mge
from model import ReverseString
from dataset import get_dataloader, make_string_from_tensor, MAXLEN


model = ReverseString()
model.load_state_dict(mge.load('transformer.20.mge'))

from IPython import embed; embed()
