from unittest import TestCase

import torch
from s2cnn import s2_equatorial_grid
from s2cnn.soft.s2_conv import S2Convolution


class TestS2Convolution(TestCase):
    def test_forward(self):
        input = torch.rand(4, 6, 64, 64).cuda()
        grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * 32, n_beta=1)
        s2 = S2Convolution(6, 100, 32, 32, grid).cuda()
        self.assertEqual(s2(input).shape, (4, 100, 64, 64, 64))
