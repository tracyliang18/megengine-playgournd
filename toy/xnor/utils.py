def accuracy(pred, label):
    predicted = pred.argmax(axis=1)
    correct = (predicted==label).sum()
    total = label.shape[0]
    return correct/total
