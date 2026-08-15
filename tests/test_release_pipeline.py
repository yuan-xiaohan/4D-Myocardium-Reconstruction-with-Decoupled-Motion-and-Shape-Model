import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import SimpleITK as sitk

from case_config import CaseConfig, load_case_configs
from preprocess_mask import (
    RegistrationResult,
    choose_registration,
    discover_frames,
    load_registration_assets,
    phase_mapping,
    run_preprocessing,
    validate_labels,
)
from run_pipeline import run_case
from train_4dmm import _resolve_training_devices, load_training_config, run_all_training


STAGES_DISABLED = {
    "preprocess": True,
    "inference": False,
    "postprocess": False,
    "video": False,
}


def write_mask(path: Path, include_rv: bool, origin=(0.0, 0.0, 0.0)) -> None:
    array = np.zeros((6, 64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[:64, :64]
    for z in range(6):
        radius = 9 + z
        cavity = (xx - 31) ** 2 + (yy - 31) ** 2 <= radius**2
        outer = (xx - 31) ** 2 + (yy - 31) ** 2 <= (radius + 5) ** 2
        array[z, outer & ~cavity] = 2
        array[z, cavity] = 3
        if include_rv:
            array[z, (xx - 49) ** 2 + (yy - 31) ** 2 <= 7**2] = 1
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(origin)
    image.SetSpacing((1.2, 1.3, 6.0))
    sitk.WriteImage(image, str(path))


class CaseConfigTests(unittest.TestCase):
    def test_auto_discovery_and_recursive_case_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "inputs" / "CaseA" / "masks").mkdir(parents=True)
            config_path = root / "cases.json"
            config_path.write_text(json.dumps({
                "default": {
                    "num_iterations": 20,
                    "stages": STAGES_DISABLED,
                },
                "cases": {
                    "CaseA": {
                        "reference_phase": 2,
                        "phase_batch_size": 2,
                        "stages": {"preprocess": False},
                    }
                },
            }))
            configs = load_case_configs(
                config_path=config_path,
                input_root=root / "inputs",
                output_root=root / "outputs",
            )
            self.assertEqual(len(configs), 1)
            case = configs[0]
            self.assertEqual(case.case_id, "CaseA")
            self.assertEqual(case.reference_phase, 2)
            self.assertEqual(case.phase_batch_size, 2)
            self.assertFalse(case.stages["preprocess"])
            self.assertFalse(case.stages["video"])

    def test_unknown_override_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "inputs" / "CaseA" / "masks").mkdir(parents=True)
            config_path = root / "cases.json"
            config_path.write_text(json.dumps({
                "default": {"stages": STAGES_DISABLED},
                "cases": {"CaseA": {"unknown": 1}},
            }))
            with self.assertRaisesRegex(ValueError, "Unknown fields"):
                load_case_configs(config_path, root / "inputs", root / "outputs")

    def test_explicit_disabled_case_is_reportable_without_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "cases.json"
            config_path.write_text(json.dumps({
                "default": {"stages": STAGES_DISABLED},
                "cases": {"DeferredCase": {"enabled": False}},
            }))
            configs = load_case_configs(config_path, root / "inputs", root / "outputs")
            self.assertEqual([item.case_id for item in configs], ["DeferredCase"])
            self.assertFalse(configs[0].enabled)

    def test_each_case_keeps_independent_checkpoint_and_stage_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case_id in ("CaseA", "CaseB"):
                (root / "inputs" / case_id / "masks").mkdir(parents=True)
            config_path = root / "cases.json"
            config_path.write_text(json.dumps({
                "default": {"stages": STAGES_DISABLED},
                "cases": {
                    "CaseB": {
                        "checkpoints": {"endo": "custom/endo"},
                        "stages": {"video": True},
                    }
                },
            }))
            configs = {
                item.case_id: item for item in load_case_configs(
                    config_path, root / "inputs", root / "outputs"
                )
            }
            self.assertNotEqual(
                configs["CaseA"].checkpoints["endo"],
                configs["CaseB"].checkpoints["endo"],
            )
            self.assertFalse(configs["CaseA"].stages["video"])
            self.assertTrue(configs["CaseB"].stages["video"])


class TrainingScheduleTests(unittest.TestCase):
    def test_public_training_config_enables_both_surfaces_and_target(self):
        config = load_training_config()
        self.assertEqual(config["target_epoch"], 50)
        self.assertEqual(config["batch_size"], 16)
        self.assertTrue(config["surfaces"]["endo"]["enabled"])
        self.assertTrue(config["surfaces"]["epi"]["enabled"])
        self.assertEqual(config["surfaces"]["endo"]["initialization_mode"], "scratch")
        self.assertEqual(config["surfaces"]["epi"]["initialization_mode"], "scratch")

    def test_training_config_rejects_unknown_fields_and_missing_resume_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((Path(__file__).parents[1] / "configs" / "training.json").read_text())
            config["unexpected"] = True
            path = root / "unknown.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_training_config(path)

            config.pop("unexpected")
            path.write_text(json.dumps(config))
            loaded = load_training_config(path)
            self.assertIsNone(loaded["surfaces"]["endo"]["resume_from"])

    @staticmethod
    def _config(mode="auto", devices="auto", endo=True, epi=True):
        return {
            "execution": {"mode": mode, "devices": devices},
            "surfaces": {
                "endo": {"enabled": endo},
                "epi": {"enabled": epi},
            },
        }

    def test_auto_scheduler_serializes_without_cuda_or_with_one_device(self):
        with mock.patch("train_4dmm.torch.cuda.is_available", return_value=False):
            parallel, devices = _resolve_training_devices(self._config())
        self.assertFalse(parallel)
        self.assertEqual(devices, {"endo": "cpu", "epi": "cpu"})

        with mock.patch("train_4dmm.torch.cuda.is_available", return_value=True), \
             mock.patch("train_4dmm.torch.cuda.device_count", return_value=1):
            parallel, devices = _resolve_training_devices(self._config())
        self.assertFalse(parallel)
        self.assertEqual(devices, {"endo": "cuda:0", "epi": "cuda:0"})

    def test_auto_scheduler_assigns_distinct_devices_with_two_gpus(self):
        with mock.patch("train_4dmm.torch.cuda.is_available", return_value=True), \
             mock.patch("train_4dmm.torch.cuda.device_count", return_value=2):
            parallel, devices = _resolve_training_devices(self._config())
        self.assertTrue(parallel)
        self.assertEqual(devices, {"endo": "cuda:0", "epi": "cuda:1"})

    def test_disabling_one_surface_keeps_single_job(self):
        with mock.patch("train_4dmm.torch.cuda.is_available", return_value=True), \
             mock.patch("train_4dmm.torch.cuda.device_count", return_value=2):
            parallel, devices = _resolve_training_devices(self._config(endo=False))
        self.assertFalse(parallel)
        self.assertEqual(devices, {"epi": "cuda:0"})

    def test_surface_failure_does_not_hide_the_other_surface(self):
        config = {
            "execution": {"mode": "serial", "devices": ["cpu"]},
            "target_epoch": 3,
            "batch_size": 1,
            "samples_per_scene": 1,
            "max_batches_per_epoch": 1,
            "checkpoint_every": 1,
            "continue_on_surface_error": True,
            "surfaces": {
                "endo": {"enabled": True},
                "epi": {"enabled": True},
            },
        }
        failure = {"surface": "endo", "status": "failed"}
        success = {"surface": "epi", "status": "completed"}
        with mock.patch("train_4dmm._run_surface_from_config", side_effect=[failure, success]) as runner:
            summary = run_all_training(config, "scheduler-failure-test")
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["surfaces"]["epi"]["status"], "completed")


class PreprocessTests(unittest.TestCase):
    def test_pipeline_writes_one_report_and_cleans_optional_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = CaseConfig(
                case_id="Compact",
                input_mask_dir=root / "masks",
                output_dir=root / "output",
                enabled=True,
                reference_phase=0,
                mirror="normal",
                slice_order="auto",
                num_iterations=1,
                phase_batch_size=1,
                resolution=32,
                preprocess_mode="mask",
                prepared_root=None,
                device="cpu",
                skip_existing=False,
                retain_artifacts=False,
                video_order="original",
                checkpoints={"endo": root, "epi": root},
                stages={"preprocess": True, "inference": True, "postprocess": True, "video": True},
            )
            preprocess = {
                "case_id": "Compact",
                "all_passed": True,
                "original_phases": [0],
                "phase_mapping": [{"original_phase": 0, "internal_phase": 0, "t": 0.0}],
                "orientation": {},
                "manifests": {"endo": "temporary", "epi": "temporary"},
                "transform": {},
            }
            with mock.patch("run_pipeline.run_preprocessing", return_value=preprocess), \
                 mock.patch("run_pipeline.run_case_inference", return_value={
                     "all_passed": True,
                     "results": [{"mesh": "temporary", "error": 0.1}],
                 }), \
                 mock.patch("run_pipeline.run_case_postprocess", return_value={
                     "all_passed": True, "results": [],
                 }), \
                 mock.patch("run_pipeline.run_case_video", return_value={"all_passed": True}):
                report = run_case(config)
            self.assertTrue((root / "output" / "report.json").is_file())
            self.assertFalse((root / "output" / "resolved_config.json").exists())
            self.assertFalse((root / "output" / "artifacts").exists())
            self.assertEqual(report["inference"]["results"][0], {"error": 0.1})

    def test_reference_phase_rotation(self):
        mapping = phase_mapping([0, 1, 2, 3], 2)
        self.assertEqual([item["original_phase"] for item in mapping], [2, 3, 0, 1])
        self.assertEqual([item["internal_phase"] for item in mapping], [0, 1, 2, 3])
        self.assertAlmostEqual(mapping[0]["t"], 0.0)
        self.assertAlmostEqual(mapping[-1]["t"], 1.0)

    def test_mask_validation_and_nonzero_reference_preprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            masks = root / "masks"
            masks.mkdir()
            write_mask(masks / "00.nii.gz", include_rv=False)
            write_mask(masks / "01.nii.gz", include_rv=True)
            frames = discover_frames(masks)
            self.assertEqual([frame.phase for frame in frames], [0, 1])
            self.assertEqual(validate_labels(masks / "00.nii.gz", (2, 3)), [0, 2, 3])
            config = CaseConfig(
                case_id="Synthetic",
                input_mask_dir=masks,
                output_dir=root / "output",
                enabled=True,
                reference_phase=1,
                mirror="normal",
                slice_order="auto",
                num_iterations=1,
                phase_batch_size=1,
                resolution=32,
                preprocess_mode="mask",
                prepared_root=None,
                device="cpu",
                skip_existing=False,
                retain_artifacts=True,
                video_order="original",
                checkpoints={"endo": root, "epi": root},
                stages=STAGES_DISABLED,
            )
            report = run_preprocessing(config)
            self.assertTrue(report["all_passed"])
            self.assertEqual(
                [item["original_phase"] for item in report["phase_mapping"]], [1, 0]
            )
            self.assertLess(report["transform"]["inverse_error"], 1e-5)
            for surface in ("endo", "epi"):
                manifest = json.loads(
                    (root / "output" / "artifacts" / "preprocess" / f"data_load_{surface}.json").read_text()
                )
                instances = manifest["test"]["case"][f"Synthetic-{surface}"]["instance_list"]
                self.assertIn("Synthetic_01", instances[0])
                with np.load(instances[0]) as archive:
                    self.assertEqual(set(archive.files), {"pcd", "t", "P", "Pi", "offset", "scale"})
                    self.assertAlmostEqual(float(archive["t"]), 0.0)
                    self.assertTrue(np.allclose(archive["P"] @ archive["Pi"], np.eye(4)))
                    self.assertTrue(np.allclose(archive["pcd"][:, 3], 0.0))

            # A truncated/corrupt archive is not skipped: it is regenerated
            # through the same atomic output path.
            damaged = root / "output" / "artifacts" / "preprocess" / "npz" / "Synthetic_01-endo.npz"
            damaged.write_bytes(b"not-an-npz")
            resumed = run_preprocessing(replace(config, skip_existing=True))
            self.assertTrue(resumed["all_passed"])
            with np.load(damaged, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"pcd", "t", "P", "Pi", "offset", "scale"})

    def test_phase_label_and_header_validation_fail_fast(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_mask(root / "00.nii.gz", include_rv=True)
            write_mask(root / "02.nii.gz", include_rv=True)
            with self.assertRaisesRegex(ValueError, "Non-contiguous"):
                discover_frames(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_mask(root / "00.nii.gz", include_rv=True)
            write_mask(root / "01.nii.gz", include_rv=True, origin=(1.0, 0.0, 0.0))
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                discover_frames(root)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "00.nii.gz"
            write_mask(path, include_rv=True)
            image = sitk.ReadImage(str(path))
            array = sitk.GetArrayFromImage(image)
            array[0, 0, 0] = 4
            invalid = sitk.GetImageFromArray(array)
            invalid.CopyInformation(image)
            sitk.WriteImage(invalid, str(path))
            with self.assertRaisesRegex(ValueError, "Unexpected labels"):
                validate_labels(path, (1, 2, 3))

    def test_auto_mirror_uses_minimum_score_and_reports_low_confidence(self):
        curve = np.column_stack((np.linspace(0.0, 1.0, 8), np.zeros(8), np.zeros(8)))
        assets = SimpleNamespace(insertion_curves=(curve, curve))

        class Registrar:
            def __init__(self, translated):
                self.assets = assets
                self.translated = translated

            def register(self, contours, mode):
                matrix = np.eye(4)
                if self.translated and mode == "mirrored":
                    matrix[1, 3] = 10.0
                return RegistrationResult(
                    rigid=matrix,
                    inverse_rigid=np.linalg.inv(matrix),
                    scale=1.0,
                    offset=np.zeros(3),
                    mode=mode,
                    score=0.0,
                )

        insertion = (curve, curve, {"valid_slices": [0, 1]})
        with mock.patch("preprocess_mask.extract_insertion_curves", return_value=insertion):
            selected, report = choose_registration(Registrar(True), None, Path("unused"), "auto")
            self.assertEqual(selected.mode, "normal")
            self.assertLess(report["score_normal"], report["score_mirrored"])
            selected, report = choose_registration(Registrar(False), None, Path("unused"), "auto")
            self.assertEqual(selected.mode, "mirrored")
            self.assertTrue(report["low_confidence"])

    def test_registration_asset_is_small_and_valid(self):
        assets = load_registration_assets()
        self.assertGreater(len(assets.centerline), 1)
        self.assertEqual(len(assets.insertion_curves), 2)


if __name__ == "__main__":
    unittest.main()
