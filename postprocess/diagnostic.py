"""Diagnostic output for frames that fail strict postprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from .contract import (
    ENDO_LABEL,
    EPI_LABEL,
    RAW_CONCAT_ALGORITHM_NAME,
    RAW_CONCAT_ALGORITHM_VERSION,
    GeometryError,
    triangle_faces,
    validate_raw_concatenated_mesh,
)


def _load_raw_surface(path):
    """Load an inferred surface without cleaning or changing its orientation."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = pv.read(str(path)).extract_surface().triangulate()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise GeometryError(f"Empty mesh: {path}")
    points = np.asarray(mesh.points, dtype=np.float64)
    faces = triangle_faces(mesh)
    if not np.isfinite(points).all():
        raise GeometryError(f"Non-finite raw surface points: {path}")
    if faces.min() < 0 or faces.max() >= len(points):
        raise GeometryError(f"Invalid raw surface face indices: {path}")
    return mesh


def concatenate_raw_surface_pair(epi_path, endo_path):
    """Concatenate raw network surfaces without geometric postprocessing."""
    epi = _load_raw_surface(epi_path)
    endo = _load_raw_surface(endo_path)
    epi_faces = triangle_faces(epi)
    endo_faces = triangle_faces(endo) + epi.n_points
    points = np.vstack(
        (
            np.asarray(epi.points, dtype=np.float64),
            np.asarray(endo.points, dtype=np.float64),
        )
    )
    faces = np.vstack((epi_faces, endo_faces))
    vtk_faces = np.column_stack(
        (np.full(len(faces), 3, dtype=np.int64), faces)
    ).reshape(-1)
    combined = pv.PolyData(points, vtk_faces)
    combined.cell_data["surface_id"] = np.concatenate(
        (
            np.zeros(epi.n_cells, dtype=np.int32),
            np.ones(endo.n_cells, dtype=np.int32),
        )
    )
    combined.cell_data["label"] = (
        np.asarray(combined.cell_data["surface_id"], dtype=np.int32) + EPI_LABEL
    )
    combined.point_data["label"] = np.concatenate(
        (
            np.full(epi.n_points, EPI_LABEL, dtype=np.int32),
            np.full(endo.n_points, ENDO_LABEL, dtype=np.int32),
        )
    )
    combined.field_data["postprocess_algorithm_version"] = np.asarray(
        [RAW_CONCAT_ALGORITHM_VERSION], dtype=np.int32
    )
    combined.field_data["postprocess_algorithm_name"] = np.asarray(
        [RAW_CONCAT_ALGORITHM_NAME]
    )
    diagnostics = {
        "processing_mode": RAW_CONCAT_ALGORITHM_NAME,
        "postprocess_algorithm_version": RAW_CONCAT_ALGORITHM_VERSION,
        "raw_epi_points": int(epi.n_points),
        "raw_epi_cells": int(epi.n_cells),
        "raw_endo_points": int(endo.n_points),
        "raw_endo_cells": int(endo.n_cells),
        **validate_raw_concatenated_mesh(combined),
    }
    return combined, diagnostics
