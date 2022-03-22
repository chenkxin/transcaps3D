from unittest import TestCase

import torch
from lib.models import ModelCaps, makeModel
from lib.models.model_caps import ModelSphericalCaps


class TestCapsModel(TestCase):
    def test_forward(self):
        x = torch.rand(4, 6, 64, 64).cuda()
        model = ModelCaps(10).cuda()
        self.assertEqual(model(x).shape, (4, 10))

    def test_make_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        LAST_EPOCH, model = makeModel(
            "caps",
            "../logs/task1/",
            10,
            device=device,
            use_residual_block=True,
            load=False,
        )
        self.assertEqual(LAST_EPOCH, -1)

    def test_make_model_continue_training(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        LAST_EPOCH, model = makeModel(
            "caps",
            "../logs/task1/",
            10,
            device=device,
            use_residual_block=True,
            load=True,
        )

        # x = torch.rand(4, 6, 64, 64).to(device)
        # output=model(x)
        # print(model)
        # print(output.shape)


class TestModelCaps(TestCase):
    def test_forward(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ModelCaps(10).to(device)
        x = torch.rand(4, 6, 64, 64).to(device)
        output = model(x)
        print(model)
        print(output.shape)


class TestModelSphericalCaps(TestCase):
    def test__init_primary_layers(self):
        pass

    def test__init_caps_conv_layers(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = ModelSphericalCaps(
            b_in=32,
            primary=[
                (50, 32),  # (d_out, b_out) S^2 conv block or residual block
                (5, 10, 22),  # (n_out_caps, d_out_caps, b_out)
            ],
            hidden=[(5, 10, 22), (5, 10, 7), (5, 10, 7), (10, 10, 7)],
        ).to(device)
        for i in range(10):
            x = torch.rand(4, 6, 64, 64).to(device)
            self.assertEqual((4, 10), model(x).shape)

    def test_compare_with_model_caps(self):
        model = ModelCaps(10)
        print(model)
