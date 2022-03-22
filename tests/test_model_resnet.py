from unittest import TestCase

import torch
from lib.models.model_resnet import ModelResNet


class TestModelResNet(TestCase):
    def test_forward(self):
        x = torch.rand(4, 6, 64, 64).cuda()
        model = ModelResNet(10).cuda()
        self.assertEqual(model(x).shape, (4, 10))
