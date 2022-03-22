import time
from unittest import TestCase

from lib.dataset import makeDataLoader
from tqdm import tqdm


class TestMakeLoader(TestCase):
    def test_shrec15(self):
        N, classes, train_dataloader = makeDataLoader(True, 32, 4, 4, "shrec15")
        print(f"total {N} samples")
        start = time.time()
        for i, (X, y) in enumerate(train_dataloader):
            end = time.time()
            print(f"data is loaded using {end-start} seconds")
            start = time.time()

    def test_shrec17(self):
        N, classes, train_dataloader = makeDataLoader(True, 32, 4, 4, "shrec17")
        print(f"total {N} samples")
        start = time.time()
        for i, (X, y) in enumerate(train_dataloader):
            print(X, y)
        end = time.time()
        print(f"all data is loaded using {end-start} seconds")

    def test_modelnet10(self):

        N, classes, train_dataloader = makeDataLoader(True, 32, 4, 4, "modelnet10")
        print(f"total {N} samples")
        start = time.time()
        for i, (X, y) in enumerate(train_dataloader):
            print(X.shape, y.shape)
        end = time.time()
        print(f"all data is loaded using {end-start} seconds")

    def test_modelnet10_multi_input(self):
        N, classes, train_dataloader = makeDataLoader(
            True, [32, 16, 8], 4, 4, "modelnet10"
        )
        print(f"total {N} samples")
        start = time.time()
        for i, (X, y) in enumerate(train_dataloader):
            print(X.shape, y)
        end = time.time()
        print(f"all data is loaded using {end-start} seconds")

    def test_modelnet10_without_normalize(self):
        N, classes, train_dataloader = makeDataLoader(
            True, 32, 4, 4, "modelnet10", normalize=False
        )
        print(f"total {N} samples")
        start = time.time()
        for i, (X, y) in enumerate(train_dataloader):
            print(X, y)
        end = time.time()
        print(f"all data is loaded using {end-start} seconds")


class TestMakeLoaderWithNormalize(TestCase):
    """
    Load the mean and std, normalize models

    """

    def test_modelnet10_without_normalize(self):
        start = time.time()
        N, classes, train_dataloader = makeDataLoader(
            True, 32, 10, 48, "modelnet10", normalize=True
        )
        print(f"total {N} samples")
        for X, y in train_dataloader:
            pass
        end = time.time()
        print(f"totally use {end-start}s")

    def test_modelnet40_without_normalize(self):
        start = time.time()
        N, classes, train_dataloader = makeDataLoader(
            True, 32, 10, 48, "modelnet40", normalize=True
        )
        print(f"total {N} samples")
        for X, y in train_dataloader:
            pass
        end = time.time()
        print(f"totally use {end-start}s")

    def test_shrec15_without_normalize(self):
        start = time.time()
        N, classes, train_dataloader = makeDataLoader(
            True, 32, 10, 48, "shrec15", normalize=True
        )
        print(f"total {N} samples")
        for X, y in train_dataloader:
            pass
        end = time.time()
        print(f"totally use {end-start}s")

    def test_shrec17_without_normalize(self):
        start = time.time()
        N, classes, train_dataloader = makeDataLoader(
            True, 32, 10, 48, "shrec17", normalize=True
        )
        print(f"total {N} samples")
        for X, y in train_dataloader:
            pass
        end = time.time()
        print(f"totally use {end-start}s")


def traverse(dataset_name, train, normalize=False):
    print(f"traversing {dataset_name}")
    start = time.time()
    N, classes, train_dataloader = makeDataLoader(
        train, 32, 10, 48, dataset_name, normalize=normalize
    )
    for X, y in train_dataloader:
        pass
    end = time.time()
    print(f"totally use {end-start}s")


class TestMakeLoaderWithoutNormalize(TestCase):
    """
    Don't normalize

    """

    def test_modelnet10_without_normalize(self):
        traverse(dataset_name="model10", train=True)
        traverse(dataset_name="model10", train=False)

    def test_modelnet40_without_normalize(self):
        start = time.time()
        N, classes, train_dataloader = makeDataLoader(
            True, 32, 10, 48, "modelnet40", normalize=False
        )
        print(f"total {N} samples")
        for X, y in train_dataloader:
            pass
        end = time.time()
        print(f"totally use {end-start}s")

    def test_shrec15_without_normalize(self):
        start = time.time()
        N, classes, train_dataloader = makeDataLoader(
            True, 32, 10, 48, "shrec15", normalize=False
        )
        print(f"total {N} samples")
        for X, y in train_dataloader:
            pass
        end = time.time()
        print(f"totally use {end-start}s")

    def test_shrec17_without_normalize(self):
        start = time.time()
        N, classes, train_dataloader = makeDataLoader(
            True, 32, 10, 48, "shrec17", normalize=False
        )
        print(f"total {N} samples")
        for X, y in train_dataloader:
            pass
        end = time.time()
        print(f"totally use {end-start}s")

    def test_smnist(self):
        start = time.time()
        N, classes, train_dataloader = makeDataLoader(
            True, 32, 10, 48, "smnist", normalize=False
        )
        print(f"total {N} samples")
        for X, y in train_dataloader:
            pass
        end = time.time()
        print(f"totally use {end-start}s")
