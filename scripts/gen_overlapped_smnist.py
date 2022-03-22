import gzip
import pickle
import numpy as np
import os, sys

sys.path.append(os.getcwd())
from lib.dataset.utils import compute_mean_and_std_for


def merge(a, b):
    """merge two image
    Args:
        a: [H, W]
        b: [H, W]
    Returns:
        [H,W]
    """
    c = np.stack([a, b], axis=0)
    return np.max(c, axis=0)


def load_smnist(path="data/s2_mnist_nr_nr.gz", train=True):
    choice = "train" if train else "test"
    with gzip.open(path, "rb") as f:
        dataset = pickle.load(f)
        data = dataset[choice]["images"][:, None, :, :].astype(np.float32)
        labels = dataset[choice]["labels"].astype(np.int64)
    return data, labels


def gen_overlapped_data(data, labels, N_test=1000):
    N = data.shape[0]

    # select indexes algorithm
    choices = []
    while len(choices) < N_test:
        # generate a choice
        choice = np.random.randint(0, N, (2,))
        choice = list(choice)
        if choice[0] == choice[1]:
            break
        # choice not in indexes and label of choice not equal
        label = labels[choice]
        if choice not in choices and label[0] != label[1]:
            choices.append(choice)
    index = np.stack(choices, axis=0)

    # merge two images into one
    test_data = data[index]
    test_data = np.max(test_data, axis=1).squeeze(axis=1)
    test_labels = labels[index]
    return test_data, test_labels


def write(test_data, test_labels, path="data/s2_mnist_nr_nr.gz", train=True):
    choice = "train" if train else "test"
    with gzip.open(path, "rb") as f:
        dataset = pickle.load(f)
        dataset[choice]["overlap_data"] = test_data
        dataset[choice]["overlap_labels"] = test_labels
    with gzip.open(path, "wb") as f:
        pickle.dump(dataset, f)
    print("write overlap data done!")


if __name__ == "__main__":
    train = False
    path = "data/s2_mnist_nr_r.gz"
    number_generated_data = 100
    print("generating {} on {}".format(number_generated_data, path))
    data, labels = load_smnist(train=train, path=path)
    test_data, test_labels = gen_overlapped_data(
        data, labels, N_test=number_generated_data
    )
    write(test_data, test_labels, train=train, path=path)
