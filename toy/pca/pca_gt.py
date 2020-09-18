from dataset import PCADataset
from sklearn.decomposition import PCA
import setting

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=int, help="")
    args = parser.parse_args()


    train_dataset = PCADataset(args.input, setting.points_num, setting.batch_size)

    print(len(train_dataset))

    print(train_dataset.data.shape)

    pca = PCA(n_components=args.input)
    pca.fit(train_dataset.data)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    from IPython import embed; embed()
