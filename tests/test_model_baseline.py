from unittest import TestCase

import torch
from lib.models import ModelBaseline_3d, ModelBaseline_SMNIST


class TestModelBaseline(TestCase):
    def test_forward(self):
        x = torch.rand(4, 6, 64, 64).cuda()
        model = ModelBaseline_3d(10).cuda()
        output = model(x)
        self.assertEqual(output.shape, (4, 10))


class TestModelBaseline_SMNIST(TestCase):
    def test_forward(self):
        x = torch.rand(4, 1, 60, 60).cuda()
        model = ModelBaseline_SMNIST().cuda()
        output = model(x)
        self.assertEqual((4, 10), output.shape)
