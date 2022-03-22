from unittest import TestCase

import torch
from torch import nn

from lib.loss import MarginLoss


class TestMarginLoss(TestCase):
    def test_forward(self):
        nclass = 40
        class_capsules = torch.rand(4, nclass, 16)
        labels = torch.randint(0, nclass, (4,))
        criterion = MarginLoss(nclass=nclass)
        loss = criterion(class_capsules, labels)
        self.assertEqual((), loss.shape)

    def test_nll_loss(self):
        criterion = nn.NLLLoss()
        y_hat = torch.rand(4, 10)
        y = torch.randint(0, 10, (4,))
        loss = criterion(y_hat, y)
        self.assertEqual((), loss.shape)
