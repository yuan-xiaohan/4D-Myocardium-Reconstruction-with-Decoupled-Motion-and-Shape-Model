import unittest

import numpy as np
import pyvista as pv

import postprocess_4dmm_bridge as bridge


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
        report = bridge.validate_combined_mesh(_valid_combined_mesh())
        self.assertTrue(report["finite_points"])
        self.assertTrue(report["triangular_faces"])
        self.assertTrue(report["valid_face_indices"])
        self.assertEqual(report["n_open_edges"], 0)
        self.assertTrue(report["is_manifold"])
        self.assertGreater(report["epi_outward_score"], 0.0)
        self.assertLess(report["endo_outward_score"], 0.0)

    def test_nonfinite_vertex_is_rejected(self):
        mesh = _valid_combined_mesh()
        points = np.asarray(mesh.points).copy()
        points[0, 0] = np.nan
        mesh.points = points
        with self.assertRaisesRegex(bridge.GeometryError, "NaN or infinite"):
            bridge.validate_combined_mesh(mesh)


if __name__ == "__main__":
    unittest.main()
