import unittest
import importlib
from unittest import mock

import torch

from reconstruct_4dmm import (
    _project_code_norm_,
    optimize_motion_batch,
    optimize_shape,
)


Decoder = importlib.import_module("networks.4dmm_decoder").Decoder


class LatentCodeConstraintTests(unittest.TestCase):
    def test_projection_keeps_each_code_inside_bound(self):
        codes = torch.tensor([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])
        _project_code_norm_(codes, 1.0)
        norms = torch.linalg.vector_norm(codes, dim=-1)
        self.assertTrue(torch.all(norms <= 1.0 + 1e-6))
        self.assertAlmostEqual(float(norms[0]), 1.0, places=6)
        self.assertAlmostEqual(float(norms[1]), 0.5, places=6)
        self.assertEqual(float(norms[2]), 0.0)

    def test_none_bound_preserves_codes(self):
        codes = torch.tensor([[3.0, 4.0]])
        original = codes.clone()
        _project_code_norm_(codes, None)
        self.assertTrue(torch.equal(codes, original))

    def test_shape_optimizer_projects_after_each_update(self):
        class ShapeCodeDecoder:
            def __call__(self, coords, time, motion_code, shape_code):
                prediction = shape_code[:, None, :1].expand(
                    coords.shape[0], coords.shape[1], 1
                ).reshape(-1, 1)
                return coords, prediction

        samples = torch.zeros(4, 4)
        shape_code = torch.tensor([[4.0, 3.0]])
        motion_codes = torch.zeros(2, 2)
        specs = {
            "CodeBound": 1.0,
            "SamplesPerScene": 4,
            "ClampingDistance": 10.0,
        }
        with mock.patch(
            "reconstruct_4dmm.get_sdf_samples_test",
            return_value=(samples, 0.0),
        ):
            optimize_shape(
                ShapeCodeDecoder(),
                {"npz": "unused"},
                shape_code,
                motion_codes,
                specs,
                num_iterations=2,
                device="cpu",
            )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(shape_code.detach())), 1.0 + 1e-6
        )

    def test_motion_optimizer_projects_non_reference_codes_only(self):
        class MotionCodeDecoder:
            def __call__(self, coords, time, motion_code, shape_code):
                prediction = motion_code[:, :1, None].expand(
                    coords.shape[0], coords.shape[1], 1
                )
                return coords, prediction

        samples = torch.zeros(4, 4)
        shape_code = torch.zeros(1, 2)
        motion_codes = torch.tensor([[7.0, 0.0], [4.0, 3.0]])
        reference_before = motion_codes[0].clone()
        specs = {
            "CodeBound": 1.0,
            "SamplesPerScene": 4,
            "ClampingDistance": 10.0,
        }
        frames = [
            {
                "npz": "unused",
                "internal_phase": 1,
                "original_phase": 1,
            }
        ]
        with mock.patch(
            "reconstruct_4dmm.get_sdf_samples_test",
            return_value=(samples, 0.5),
        ):
            optimize_motion_batch(
                MotionCodeDecoder(),
                frames,
                shape_code,
                motion_codes,
                specs,
                num_iterations=2,
                device="cpu",
            )
        self.assertTrue(torch.equal(motion_codes[0], reference_before))
        self.assertLessEqual(
            float(torch.linalg.vector_norm(motion_codes[1])), 1.0 + 1e-6
        )

    def test_reference_time_ignores_motion_code(self):
        decoder = Decoder(
            motionnet_kargs={
                "dim": 4,
                "in_features": 2,
                "out_features": 3,
                "num_filters": 4,
            },
            shapenet_kargs={
                "latent_size": 2,
                "dims": [4],
                "dropout": None,
                "dropout_prob": 0.0,
                "norm_layers": [],
                "latent_in": [],
                "weight_norm": False,
                "xyz_in_all": False,
                "use_tanh": False,
                "latent_dropout": False,
            },
        )
        coords = torch.randn(1, 8, 3)
        time = torch.zeros(1)
        first = decoder.deform_points(coords, time, torch.zeros(1, 2))
        second = decoder.deform_points(coords, time, torch.full((1, 2), 100.0))
        self.assertTrue(torch.equal(first, coords))
        self.assertTrue(torch.equal(second, coords))


if __name__ == "__main__":
    unittest.main()
