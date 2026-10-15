import unittest
from unittest import mock

from anatomical_mirror import infer_auto_mirror


def candidate_scores(normal: float, mirrored: float, valid_slices: int = 5) -> dict:
    return {
        "valid_rv_slices": valid_slices,
        "template_bins": 12,
        "candidates": {
            "normal": {
                "projection_error": normal + 0.1,
                "shape_error": normal,
                "scale_fit_error": normal + 0.01,
                "registration_score": normal + 0.2,
            },
            "mirrored": {
                "projection_error": mirrored + 0.1,
                "shape_error": mirrored,
                "scale_fit_error": mirrored + 0.01,
                "registration_score": mirrored + 0.2,
            },
        },
    }


class RVTopviewAutoMirrorTests(unittest.TestCase):
    @mock.patch("anatomical_mirror.score_topview_candidates")
    def test_accepts_lower_centered_trajectory_score(self, score):
        score.return_value = candidate_scores(normal=0.20, mirrored=0.08)

        decision = infer_auto_mirror("contours", "registrar")

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["mirror"], "mirrored")
        self.assertEqual(decision["status"], "accepted")
        self.assertEqual(decision["source"], "rv_topview_auto_v1")
        self.assertEqual(decision["camera"], "legacy_norm_top")
        self.assertEqual(decision["valid_rv_slices"], 5)
        score.assert_called_once_with("contours", "registrar", template_bins=12)

    @mock.patch("anatomical_mirror.score_topview_candidates")
    def test_rejects_low_normalized_margin(self, score):
        score.return_value = candidate_scores(normal=0.100, mirrored=0.104)

        decision = infer_auto_mirror("contours", "registrar")

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["mirror"], "normal")
        self.assertEqual(decision["status"], "low_confidence")
        self.assertAlmostEqual(decision["normalized_margin"], 0.004 / 0.104)
        self.assertIn("below", decision["reason"])

    @mock.patch("anatomical_mirror.score_topview_candidates")
    def test_converts_insufficient_rv_to_explicit_decision(self, score):
        score.side_effect = ValueError("Only 1 RV-bearing slices; require at least 2")

        decision = infer_auto_mirror("contours", "registrar")

        self.assertFalse(decision["accepted"])
        self.assertIsNone(decision["mirror"])
        self.assertEqual(decision["status"], "insufficient_evidence")
        self.assertEqual(decision["min_rv_slices"], 2)

    def test_validates_threshold(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            infer_auto_mirror("contours", "registrar", confidence_threshold=1.1)


if __name__ == "__main__":
    unittest.main()
