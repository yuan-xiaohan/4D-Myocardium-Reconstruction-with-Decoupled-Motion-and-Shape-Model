import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyvista as pv

from postprocess import contract
from postprocess import diagnostic
from postprocess import geometry as bridge
from postprocess import runner as postprocess_runner
from postprocess import video as postprocess_video


def _merge_disconnected(*meshes):
    merged = meshes[0].copy()
    for mesh in meshes[1:]:
        merged = merged.merge(mesh, merge_points=False)
    return merged.extract_surface().triangulate()


def _closed_surface(radius=2.0, center=(0.0, 0.0, 0.0)):
    return pv.Sphere(
        radius=radius,
        center=center,
        theta_resolution=16,
        phi_resolution=16,
    ).triangulate()


def _open_endo_surface():
    return (
        _closed_surface(radius=1.0)
        .clip(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.25))
        .extract_surface()
        .triangulate()
    )


def _valid_combined_mesh():
    epi = pv.Sphere(
        radius=2.0, theta_resolution=16, phi_resolution=16
    ).triangulate()
    endo = pv.Sphere(
        radius=1.0, theta_resolution=16, phi_resolution=16
    ).triangulate()
    epi, _ = bridge._orient_surface(epi, outward=True)
    endo, _ = bridge._orient_surface(endo, outward=False)
    combined = bridge._merge_two_surfaces(epi, endo)
    combined = combined.compute_normals(
        cell_normals=True,
        point_normals=True,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=False,
        non_manifold_traversal=True,
    )
    return bridge._assign_surface_ids(combined, epi, endo)


class PostprocessValidationTests(unittest.TestCase):
    def test_valid_mesh_reports_complete_geometry_contract(self):
        report = contract.validate_combined_mesh(_valid_combined_mesh())
        self.assertTrue(report["finite_points"])
        self.assertTrue(report["triangular_faces"])
        self.assertTrue(report["valid_face_indices"])
        self.assertEqual(report["n_open_edges"], 0)
        self.assertTrue(report["is_manifold"])
        self.assertGreater(report["epi_outward_score"], 0.0)
        self.assertLess(report["endo_outward_score"], 0.0)

    def test_previous_strict_algorithm_version_remains_readable(self):
        mesh = _valid_combined_mesh()
        mesh.field_data["postprocess_algorithm_version"] = np.asarray(
            [3], dtype=np.int32
        )
        report = contract.validate_combined_mesh(mesh)
        self.assertEqual(report["n_open_edges"], 0)

    def test_nonfinite_vertex_is_rejected(self):
        mesh = _valid_combined_mesh()
        points = np.asarray(mesh.points).copy()
        points[0, 0] = np.nan
        mesh.points = points
        with self.assertRaisesRegex(bridge.GeometryError, "NaN or infinite"):
            contract.validate_combined_mesh(mesh)


class RawSurfaceCleanupTests(unittest.TestCase):
    def test_single_components_are_preserved(self):
        epi = _closed_surface()
        endo = _open_endo_surface()
        clean_epi, clean_endo, diagnostics = (
            bridge._remove_raw_surface_artifacts(epi, endo)
        )
        self.assertEqual(clean_epi.n_cells, epi.n_cells)
        self.assertEqual(clean_endo.n_cells, endo.n_cells)
        self.assertEqual(
            diagnostics["combined_removed_component_count"], 0
        )
        bridge._validate_raw_surface_roles(clean_epi, clean_endo)

    def test_tiny_closed_components_are_removed(self):
        epi = _merge_disconnected(
            _closed_surface(),
            _closed_surface(radius=0.03, center=(4.0, 0.0, 0.0)),
        )
        endo = _merge_disconnected(
            _open_endo_surface(),
            _closed_surface(radius=0.01, center=(-4.0, 0.0, 0.0)),
        )
        clean_epi, clean_endo, diagnostics = (
            bridge._remove_raw_surface_artifacts(epi, endo)
        )
        self.assertEqual(len(bridge._surface_components(clean_epi)), 1)
        self.assertEqual(len(bridge._surface_components(clean_endo)), 1)
        self.assertEqual(
            diagnostics["combined_removed_component_count"], 2
        )
        bridge._validate_raw_surface_roles(clean_epi, clean_endo)

    def test_non_negligible_secondary_component_is_rejected(self):
        epi = _merge_disconnected(
            _closed_surface(),
            _closed_surface(radius=0.2, center=(4.0, 0.0, 0.0)),
        )
        with self.assertRaisesRegex(
            bridge.GeometryError, "non-negligible secondary components"
        ):
            bridge._remove_raw_surface_artifacts(epi, _open_endo_surface())

    def test_too_many_secondary_components_are_rejected(self):
        islands = [
            _closed_surface(
                radius=0.005,
                center=(4.0 + index * 0.1, 0.0, 0.0),
            )
            for index in range(bridge.MAX_RAW_ARTIFACT_COMPONENTS + 1)
        ]
        epi = _merge_disconnected(_closed_surface(), *islands)
        with self.assertRaisesRegex(
            bridge.GeometryError, "too many secondary components"
        ):
            bridge._remove_raw_surface_artifacts(epi, _open_endo_surface())

    def test_open_secondary_component_is_rejected(self):
        open_island = pv.Plane(
            center=(4.0, 0.0, 0.0),
            i_size=0.05,
            j_size=0.05,
            i_resolution=1,
            j_resolution=1,
        ).triangulate()
        epi = _merge_disconnected(_closed_surface(), open_island)
        with self.assertRaisesRegex(
            bridge.GeometryError, "must be closed and manifold"
        ):
            bridge._remove_raw_surface_artifacts(epi, _open_endo_surface())


class RawConcatFallbackTests(unittest.TestCase):
    def _write_raw_pair(self, root: Path, case_id="Fallback", phase=0):
        paths = {}
        for surface, mesh in (
            ("epi", _closed_surface()),
            ("endo", _open_endo_surface()),
        ):
            path = (
                root / "artifacts" / "raw" / surface
                / f"{case_id}-{surface}-{phase:02d}.obj"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            mesh.save(path)
            paths[surface] = path
        return paths

    def test_raw_concat_preserves_surface_counts_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._write_raw_pair(root)
            epi = pv.read(paths["epi"]).extract_surface().triangulate()
            endo = pv.read(paths["endo"]).extract_surface().triangulate()
            combined, diagnostics = diagnostic.concatenate_raw_surface_pair(
                paths["epi"], paths["endo"]
            )
            self.assertEqual(combined.n_points, epi.n_points + endo.n_points)
            self.assertEqual(combined.n_cells, epi.n_cells + endo.n_cells)
            self.assertEqual(diagnostics["epi_cells"], epi.n_cells)
            self.assertEqual(diagnostics["endo_cells"], endo.n_cells)
            self.assertEqual(
                set(np.asarray(combined.cell_data["surface_id"]).tolist()),
                {0, 1},
            )
            self.assertEqual(
                set(np.asarray(combined.cell_data["label"]).tolist()),
                {contract.EPI_LABEL, contract.ENDO_LABEL},
            )

    def test_strict_failure_writes_raw_concat_vtk(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            self._write_raw_pair(output)
            config = SimpleNamespace(
                case_id="Fallback",
                output_dir=output,
                skip_existing=False,
            )
            with mock.patch(
                "postprocess.runner.clip_and_merge_surface_pair",
                side_effect=bridge.GeometryError("strict failure"),
            ):
                result = postprocess_runner.process_phase(config, 0)
            self.assertEqual(
                result["status"], "strict_failed_raw_concat_written"
            )
            self.assertFalse(result["strict_postprocess_passed"])
            candidate = pv.read(result["output"]).extract_surface().triangulate()
            validation = contract.validate_raw_concatenated_mesh(candidate)
            self.assertTrue(validation["finite_points"])
            self.assertEqual(
                validation["epi_cells"] + validation["endo_cells"],
                candidate.n_cells,
            )
            surfaces, video_diagnostics = (
                postprocess_video._load_stitched_phase(config, 0)
            )
            self.assertEqual(set(surfaces), {"epi", "endo"})
            self.assertFalse(
                video_diagnostics["strict_postprocess_passed"]
            )
            self.assertEqual(
                video_diagnostics["output_kind"],
                "raw_concat",
            )
            config.skip_existing = True
            reused = postprocess_runner.process_phase(config, 0)
            self.assertEqual(reused["status"], "skipped_raw_concat")
            self.assertEqual(reused["output_kind"], "raw_concat")
            self.assertFalse(reused["strict_postprocess_passed"])
            self.assertTrue(reused["reused_existing"])

    def test_raw_concat_video_frame_is_visibly_marked(self):
        image = np.full((80, 120, 4), 255, dtype=np.uint8)
        normal = postprocess_video._labeled_frame(
            image, "Fallback", 3, strict_postprocess_passed=True
        )
        diagnostic = postprocess_video._labeled_frame(
            image, "Fallback", 3, strict_postprocess_passed=False
        )
        self.assertFalse(np.array_equal(normal, diagnostic))
        self.assertGreater(
            int(diagnostic[:, :, 2].sum()), int(diagnostic[:, :, 0].sum())
        )

    def test_raw_concat_failure_is_reported_without_raising(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            self._write_raw_pair(output)
            config = SimpleNamespace(
                case_id="Fallback",
                output_dir=output,
                skip_existing=False,
            )
            with mock.patch(
                "postprocess.runner.clip_and_merge_surface_pair",
                side_effect=bridge.GeometryError("strict failure"),
            ), mock.patch(
                "postprocess.runner.concatenate_raw_surface_pair",
                side_effect=bridge.GeometryError("raw failure"),
            ):
                result = postprocess_runner.process_phase(config, 0)
            self.assertEqual(
                result["status"], "strict_failed_raw_concat_failed"
            )
            self.assertIsNone(result["output"])
            self.assertEqual(result["strict_error"], "strict failure")
            self.assertEqual(result["fallback_error"], "raw failure")

    def test_case_continues_after_one_phase_uses_raw_concat(self):
        config = SimpleNamespace(case_id="Fallback", output_dir=Path("unused"))

        def fake_process_phase(_config, phase):
            if phase == 1:
                return {
                    "phase": phase,
                    "status": "strict_failed_raw_concat_written",
                    "strict_postprocess_passed": False,
                    "output_kind": "raw_concat",
                    "output": f"frame_{phase:02d}.vtk",
                }
            return {
                "phase": phase,
                "status": "processed",
                "strict_postprocess_passed": True,
                "output_kind": "strict",
                "output": f"frame_{phase:02d}.vtk",
            }

        with mock.patch(
            "postprocess.runner.process_phase", side_effect=fake_process_phase
        ) as patched:
            report = postprocess_runner.run_case_postprocess(
                config, {"original_phases": [0, 1, 2]}
            )
        self.assertEqual(patched.call_count, 3)
        self.assertFalse(report["all_passed"])
        self.assertTrue(report["all_outputs_written"])
        self.assertEqual(report["strict_passed_count"], 2)
        self.assertEqual(report["raw_concat_count"], 1)

if __name__ == "__main__":
    unittest.main()
