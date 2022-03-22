import torch
from unittest import TestCase
from lib.models.model_smnist import SMNIST


class TestSMNIST(TestCase):
    def test_forward(self):
        x = torch.rand(4, 1, 60, 60).cuda()
        model = SMNIST(10).cuda()
        print(model)
        self.assertEqual(model(x).shape, (4, 10))
