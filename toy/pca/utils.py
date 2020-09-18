import errno
import os
def accuracy(pred, label):
    predicted = pred.argmax(axis=1)
    correct = (predicted==label).sum()
    total = label.shape[0]
    return correct/total



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

