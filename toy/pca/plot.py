import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, type=str, help="")
    parser.add_argument("-e", "--epoch", default=9, type=int, help="")
    args = parser.parse_args()

    save_path = os.path.join(os.path.dirname(args.file), 'curve.png')

    fig, ax = plt.subplots()

    with open(args.file) as fin:
        lines = fin.readlines()[::-1]
        trains = []
        tests = []
        ok = False
        for line in lines:
            if 'epoch: {}'.format(args.epoch) in line:
                ok=True
            if ok:
                records = line.split(',')
                train_loss = float(records[1].split(':')[1])
                test_loss = float(records[2].split(':')[1])

                trains.append(train_loss)
                tests.append(test_loss)
            if ok and 'epoch: 0' in line:
                break
    print(trains[::-1], tests[::-1])

    ax.plot(np.log(np.array(trains[::-1]) + 1), label="train")
    ax.plot(np.log(np.array(tests[::-1]) + 1), label="test")
    ax.legend()

    plt.savefig(save_path)
