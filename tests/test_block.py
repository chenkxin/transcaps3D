from unittest import TestCase

import torch
from lib.models.block import CapsuleResidualBlock


class Test(TestCase):
    def test_for_capsule_residual_block(self):
        # conv1 = conv2 = SO3 and we don't want to change the size of the sphere
        x = torch.rand(1, 5 * 20, 32, 32, 32).to("cuda:0")  # normal features
        model = CapsuleResidualBlock(
            in_features=100, out_features=100, b_in=16, b_out=16
        ).to("cuda:0")
        output = model(x)  # features of capsule we want
        self.assertEqual(output.shape, (1, 100, 32, 32, 32))

    def test_for_capsule_residual_block_lower_b_out(self):
        # conv1 = conv2 = SO3 and we don't want to change the size of the sphere
        x = torch.rand(1, 5 * 20, 32, 32, 32).to("cuda:0")  # normal features
        model = CapsuleResidualBlock(
            in_features=100, out_features=100, b_in=16, b_out=8
        ).to("cuda:0")
        output = model(x)  # features of capsule we want
        self.assertEqual(output.shape, (1, 100, 16, 16, 16))

    def test_get_inital_block(self):
        from lib.models.block import get_inital_block

        feature_extractor = get_inital_block(
            6, 100, 16, 16, use_residual_block=True
        ).to("cuda:0")
        x = torch.rand(1, 6, 32, 32).to("cuda:0")
        output = feature_extractor(x)
        self.assertEqual(output.shape, (1, 100, 32, 32, 32))

    def test_get_capsule_block(self):
        from lib.models.block import get_capsule_block

        feature_extractor = get_capsule_block(
            100, 100, 16, 16, use_residual_block=True
        ).to("cuda:0")
        x = torch.rand(1, 100, 32, 32, 32).to("cuda:0")
        output = feature_extractor(x)
        self.assertEqual(output.shape, (1, 100, 32, 32, 32))


class TestPrimaryCapsuleLayer(TestCase):
    def test_forward(self):
        from lib.models.block import PrimaryCapsuleLayer

        x = torch.rand(1, 5 * 20, 16, 16, 16).to("cuda:0")  # normal features
        model = PrimaryCapsuleLayer(
            in_features=100, num_out_capsules=10, capsule_dim=15, b_in=8, b_out=8
        ).to("cuda:0")
        output = model(x)
        self.assertEqual(output.shape, (1, 10, 15, 16, 16, 16))


class TestConvolutionalCapsuleLayer(TestCase):
    def test_forward(self):
        from lib.models.block import ConvolutionalCapsuleLayer

        x = torch.rand(4, 5, 20, 16, 16, 16).to("cuda:0")
        layer = ConvolutionalCapsuleLayer(5, 20, 10, 10, 8, 4).to("cuda:0")
        self.assertEqual(layer(x).shape, (4, 10, 10, 8, 8, 8))
