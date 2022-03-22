from unittest import TestCase
from lib.dataset.smnist import SMNIST_Dataset


class TestSMNIST_Dataset(TestCase):
    def test_smnist_dataset(self):
        dataset = SMNIST_Dataset()
        self.assertEqual((1, 60, 60), dataset[1][0].shape)
        self.assertEqual((), dataset[1][1].shape)

    def test_smnist_dataset_overlap(self):
        dataset = SMNIST_Dataset(
            train=False, no_rotate_train=True, no_rotate_test=True, overlap=True
        )
        self.assertEqual((1, 60, 60), dataset[1][0].shape)
        self.assertEqual((2,), dataset[1][1].shape)
